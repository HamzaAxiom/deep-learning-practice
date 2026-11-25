# %%
import torch
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT

model = keypointrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()
print("Model loaded to device: ", device)


# %%
from PIL import Image

image = Image.open("person.jpg")

transform = weights.transforms()

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

output = output[0]

print("Boxes:", output["boxes"].shape)
print("Keypoints:", output["keypoints"].shape)
print("Scores:", output["scores"].shape)
# %% draw 
from torchvision.utils import draw_keypoints
import matplotlib.pyplot as plt

# Keep only confident detections' X and Y, and move to CPU
keep = output["scores"] > 0.7
keypoints = output["keypoints"][keep, :, :2].cpu()

# Convert image to uint8 tensor
from torchvision.transforms.functional import pil_to_tensor
img_tensor = pil_to_tensor(image.convert("RGB"))

# Scale keypoint radius to image resolution so dots remain visible when plotted
radius = max(4, min(img_tensor.shape[1:]) // 150)

# Draw keypoints
result = draw_keypoints(
    img_tensor,
    keypoints,
    radius=radius,
    colors="red"
)

plt.figure(figsize=(8, 8))
plt.imshow(result.permute(1, 2, 0))
plt.axis("off")
plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.imshow(image)

# First detected person's keypoints
points = output["keypoints"][0, :, :2].cpu()

plt.scatter(
    points[:, 0],
    points[:, 1],
    c="red",
    s=80
)

plt.axis("off")
plt.show()
# %% printing the COCO indices 5, 7, and 9 for left shoulder, elbow, and wris


# Select the first detected person
person_keypoints = output["keypoints"][0]

# Get the indices for left shoulder, left elbow, left wrist
l_shoulder_idx = 5
l_elbow_idx = 7
l_wrist_idx = 9

# Extract the keypoints
l_shoulder = person_keypoints[l_shoulder_idx, :2].cpu()
l_elbow = person_keypoints[l_elbow_idx, :2].cpu()
l_wrist = person_keypoints[l_wrist_idx, :2].cpu()

# Draw on image
import matplotlib.patches as patches

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

# Plot joints
ax.scatter(l_shoulder[0], l_shoulder[1], c="red", s=100, label="L Shoulder (5)")
ax.scatter(l_elbow[0], l_elbow[1], c="green", s=100, label="L Elbow (7)")
ax.scatter(l_wrist[0], l_wrist[1], c="blue", s=100, label="L Wrist (9)")

# Draw lines
ax.plot([l_shoulder[0], l_elbow[0]], [l_shoulder[1], l_elbow[1]], "--r")
ax.plot([l_elbow[0], l_wrist[0]], [l_elbow[1], l_wrist[1]], "--g")

ax.legend()
plt.show()
# %% 
person_keypoints = output["keypoints"][0]
l_shoulder = person_keypoints[5].cpu()

print(l_shoulder)
# %%  checking confidence for person
for i, (box, score) in enumerate(zip(output["boxes"], output["scores"])):
    print(i, f"score={score:.3f}", box.int().tolist())
# %%
blank = torch.zeros_like(img_tensor)        # black background
# or: torch.full_like(img_tensor, 255)     # white background

skeleton = [
    [5, 7], [7, 9],    # Left arm
    [6, 8], [8, 10],   # Right arm
    [5, 6], [11, 12],  # Shoulders & hips
    [5, 11], [6, 12],  # Torso
    [11, 13], [13, 15],# Left leg (hip -> knee -> ankle)
    [12, 14], [14, 16] # Right leg (hip -> knee -> ankle)
]
result = draw_keypoints(
    blank,
    keypoints,
    connectivity=skeleton,
    radius=radius,
    width=radius // 2,
    colors="red",
)
plt.figure(figsize=(8, 8))
plt.imshow(result.permute(1, 2, 0))
plt.axis("off")
plt.show()
# %%
from torchview import draw_graph
from IPython.display import display

with torch.no_grad():
    images, _ = model.transform(input_tensor)

# 1. Backbone + Feature Pyramid Network (FPN)
graph_backbone = draw_graph(model.backbone, input_data=images.tensors, depth=2)
display(graph_backbone.visual_graph)

# 2. Keypoint Head (takes 256-channel 14x14 RoI features)
dummy_roi = torch.randn(1, 256, 14, 14, device=device)
graph_kp = draw_graph(model.roi_heads.keypoint_head, input_data=dummy_roi, depth=2)
display(graph_kp.visual_graph)
# %%
