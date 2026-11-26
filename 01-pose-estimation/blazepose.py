# %% setup & model download
import os
import time
import urllib.request

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from IPython.display import clear_output, display
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading BlazePose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# BlazePose 33-landmark body connections
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms & shoulders
    (11, 23), (12, 24), (23, 24),                      # Torso
    (23, 25), (25, 27), (27, 29), (29, 31),            # Left leg & foot
    (24, 26), (26, 28), (28, 30), (30, 32),            # Right leg & foot
]


def draw_pose(rgb_img: np.ndarray, pose_landmarks_list) -> np.ndarray:
    """Draw BlazePose landmarks and skeleton on an RGB image with scaled size."""
    out = rgb_img.copy()
    h, w = out.shape[:2]
    radius = max(4, min(h, w) // 150)
    thickness = max(2, radius // 2)

    for landmarks in pose_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for i, j in POSE_CONNECTIONS:
            cv2.line(out, pts[i], pts[j], (0, 255, 0), thickness)
        for x, y in pts:
            cv2.circle(out, (x, y), radius, (255, 0, 0), -1)
    return out


def create_landmarker(delegate):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=delegate,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


try:
    landmarker = create_landmarker(mp_python.BaseOptions.Delegate.GPU)
    print("BlazePose loaded on: GPU")
except Exception as exc:
    landmarker = create_landmarker(mp_python.BaseOptions.Delegate.CPU)
    print(f"BlazePose loaded on: CPU (GPU delegate unavailable: {exc})")

# %% own image
bgr = cv2.imread("../person.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

result = landmarker.detect(mp_image)
plotted = draw_pose(rgb, result.pose_landmarks)

plt.figure(figsize=(8, 8))
plt.imshow(plotted)
plt.axis("off")
plt.show()

# %% real-time video pose (inline in VS Code)
cap = cv2.VideoCapture("../video_test.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open ../video_test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    annotated = draw_pose(rgb, result.pose_landmarks)

    clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(annotated)
    plt.axis("off")
    display(plt.gcf())
    plt.close()

cap.release()

# %% benchmark (whole video, streaming so RAM stays low)
cap = cv2.VideoCapture("../video_test.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open ../video_test.mp4")

latencies_ms = []
det_counts = []
kpt_visibilities = []
warmup_frames = 5
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    t0 = time.perf_counter()
    result = landmarker.detect(mp_image)
    dt_ms = (time.perf_counter() - t0) * 1000

    if frame_idx >= warmup_frames:
        latencies_ms.append(dt_ms)
        det_counts.append(len(result.pose_landmarks))
        for landmarks in result.pose_landmarks:
            kpt_visibilities.append(np.mean([lm.visibility for lm in landmarks]))
    frame_idx += 1

cap.release()

lat = np.array(latencies_ms)
print(f"Frames tested : {len(lat)} (+{warmup_frames} warmup)")
print(f"FPS           : {1000.0 / lat.mean():.2f}")
print(f"Mean latency  : {lat.mean():.2f} ms (p95: {np.percentile(lat, 95):.2f} ms)")
print(f"Avg detections: {np.mean(det_counts):.2f} people/frame")
if kpt_visibilities:
    print(f"Avg visibility: {np.mean(kpt_visibilities):.3f}")
# %%
