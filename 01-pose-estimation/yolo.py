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
# %% benchmark (whole video, batched GPU inference)
import time
import numpy as np

print("YOLO device:", next(model.model.parameters()).device)

cap = cv2.VideoCapture("../video_test.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open ../video_test.mp4")

batch_size = 16
warmup_batches = 2

if device == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

batch_latencies_ms = []
det_counts = []
kpt_confs = []
batch = []
batch_idx = 0
total_frames = 0

while True:
    ret, frame = cap.read()
    if ret:
        batch.append(frame)
        if len(batch) < batch_size:
            continue
    elif not batch:
        break

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = model(batch, device=device, half=(device == "cuda"), verbose=False)

    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000

    if batch_idx >= warmup_batches:
        batch_latencies_ms.append(dt_ms)
        total_frames += len(batch)
        for r in results:
            det_counts.append(len(r.boxes))
            if r.keypoints is not None and r.keypoints.conf is not None and len(r.keypoints.conf) > 0:
                kpt_confs.append(r.keypoints.conf.mean().item())

    batch.clear()
    batch_idx += 1
    if not ret:
        break

cap.release()

total_sec = sum(batch_latencies_ms) / 1000.0
print(f"Frames tested : {total_frames} (batch_size={batch_size}, +{warmup_batches} warmup batches)")
print(f"FPS           : {total_frames / total_sec:.2f}")
print(f"Mean latency  : {(total_sec * 1000) / total_frames:.2f} ms/frame")
print(f"Avg detections: {np.mean(det_counts):.2f} people/frame")
if kpt_confs:
    print(f"Avg kpt conf  : {np.mean(kpt_confs):.3f}")
if device == "cuda":
    print(f"Peak VRAM     : {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
# %%
