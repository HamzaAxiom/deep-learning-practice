# %% 
from ultralytics import YOLO
import torch
import cv2 
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load a pretrained YOLO26n pose model
model = YOLO('yolo26n-pose.pt').to(device)
results = model("https://ultralytics.com/images/bus.jpg", device=device)
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
# %% own image
from PIL import Image
img = Image.open("../person.jpg")
results = model(img, device=device)
kpt_radius = max(5, min(img.size) // 150)
plotted = results[0].plot(kpt_radius=kpt_radius, line_width=kpt_radius // 2)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# %% Inspect architecture

print(model.model)
model.info(detailed=True)

# %% real-time video pose (inline in VS Code)
from IPython.display import clear_output, display

cap = cv2.VideoCapture("../video_test.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Could not open ../video_test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=device, verbose=False)
    kpt_radius = max(4, min(frame.shape[:2]) // 150)
    annotated = results[0].plot(kpt_radius=kpt_radius, line_width=max(2, kpt_radius // 2))

    clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    display(plt.gcf())
    plt.close()

cap.release()
# %%
# %% benchmark (whole video, streaming so RAM stays low)
import time
import numpy as np

cap = cv2.VideoCapture("../video_test.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open ../video_test.mp4")

if device == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

latencies_ms = []
det_counts = []
kpt_confs = []
warmup_frames = 5
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    r = model(frame, device=device, verbose=False)[0]

    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000

    # Skip first few frames for GPU warmup
    if frame_idx >= warmup_frames:
        latencies_ms.append(dt_ms)
        det_counts.append(len(r.boxes))
        if r.keypoints is not None and r.keypoints.conf is not None and len(r.keypoints.conf) > 0:
            kpt_confs.append(r.keypoints.conf.mean().item())
    frame_idx += 1

cap.release()

lat = np.array(latencies_ms)
print(f"Frames tested : {len(lat)} (+{warmup_frames} warmup)")
print(f"FPS           : {1000.0 / lat.mean():.2f}")
print(f"Mean latency  : {lat.mean():.2f} ms (p95: {np.percentile(lat, 95):.2f} ms)")
print(f"Avg detections: {np.mean(det_counts):.2f} people/frame")
if kpt_confs:
    print(f"Avg kpt conf  : {np.mean(kpt_confs):.3f}")
if device == "cuda":
    print(f"Peak VRAM     : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
# %%
