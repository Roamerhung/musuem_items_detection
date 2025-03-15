import torch
import clip
from PIL import Image
from torchvision import transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
import requests
from io import BytesIO

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load DETR Model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

def load_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    return image

def detect_objects(image_path):
    """Detect objects in an image using DETR."""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = detr_model(**inputs)
    
    # Process output
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() > 0.5:  # Threshold for confidence
            detected_objects.append((detr_model.config.id2label[label.item()], score.item(), box.tolist()))
    
    return detected_objects

def zero_shot_classification(image_path, text_prompts):
    """Perform zero-shot classification using CLIP."""
    image = preprocess(load_image(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Get embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)
    
    # Compute similarity scores
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    best_match = text_prompts[similarity.argmax().item()]
    
    return best_match, similarity.max().item()

def main(image_path, text_prompts):
    """Run zero-shot object detection and classification."""
    detected_objects = detect_objects(image_path)
    
    print("Objects detected:")
    for obj, conf, bbox in detected_objects:
        print(f"{obj} (Confidence: {conf:.2f})")
    
    print("\nZero-Shot Classification:")
    best_match, score = zero_shot_classification(image_path, text_prompts)
    print(f"Best Match: {best_match} (Score: {score:.2f})")

if __name__ == "__main__":
    test_image = "sample.jpg"  # Replace with an actual image path
    test_prompts = ["a cat", "a dog", "a bicycle", "a human"]
    main(test_image, test_prompts)
