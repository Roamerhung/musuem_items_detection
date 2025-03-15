import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open_clip

# ✅ Load Pretrained DETR Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model.eval()

# ✅ Load CLIP Model for Scene Description
clip_model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ✅ Load Test Image from Your Dataset
image_path = "/Users/roamerhung/Desktop/Dissertation Project/final dissertaion project/test_image.jpg"
image = Image.open(image_path).convert("RGB")

# ✅ Run DETR Object Detection
inputs = processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)

# ✅ Process DETR Output
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# ✅ Display Image with DETR Bounding Boxes
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score.item() > 0.5:  # Confidence threshold
        x1, y1, x2, y2 = map(int, box.tolist())
        
        # ✅ Draw Bounding Box
        rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # ✅ Add Label Text
        ax.text(x1, y1, f"{model.config.id2label[label.item()]}: {score.item():.2f}", 
                color="white", fontsize=8, bbox=dict(facecolor="red", alpha=0.5))

plt.axis("off")
plt.show()

# ✅ Use CLIP to Generate Scene Description
image_tensor = preprocess(image).unsqueeze(0).to(device)
text_prompts = [
    "An ancient museum artifact.",
    "A historical relic.",
    "A collection of fossils and bones.",
    "A rare skull from an ancient species."
]
text_inputs = tokenizer(text_prompts).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(image_tensor)
    text_features = clip_model.encode_text(text_inputs)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

# ✅ Select Best Scene Description
best_scene_description = text_prompts[similarity.argmax()]
print(f"📝 CLIP Scene Description: {best_scene_description}")
