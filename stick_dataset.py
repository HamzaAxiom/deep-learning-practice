import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def make_gaussian_heatmaps(
    keypoints: np.ndarray,
    img_size: int,
    heatmap_size: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Convert (K, 2) image-space keypoints to (K, H, W) Gaussian heatmaps."""
    num_keypoints = keypoints.shape[0]
    scale = heatmap_size / img_size

    yy, xx = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing="ij",
    )

    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        hx, hy = x * scale, y * scale
        heatmaps[i] = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma ** 2))

    return torch.from_numpy(heatmaps)


class StickFigureDataset(Dataset):
    """Synthetic stick-figure pose dataset with 17 COCO-style keypoints."""

    SKELETON = [
        (5, 7), (7, 9),      # Left arm
        (6, 8), (8, 10),     # Right arm
        (5, 6), (11, 12),    # Shoulders & hips
        (5, 11), (6, 12),    # Torso
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]

    def __init__(
        self,
        num_samples: int = 300,
        img_size: int = 224,
        heatmap_size: int = 56,
        sigma: float = 2.0,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.seed = seed
        self.num_keypoints = 17

    def __len__(self) -> int:
        return self.num_samples

    def _sample_keypoints(self, rng: np.random.Generator) -> np.ndarray:
        s = self.img_size
        cx = rng.uniform(0.35 * s, 0.65 * s)
        cy = rng.uniform(0.38 * s, 0.52 * s)
        scale = rng.uniform(0.75, 1.15)

        kp = np.zeros((17, 2), dtype=np.float32)

        # Head (0: nose, 1-2: eyes, 3-4: ears)
        head_y = cy - 48 * scale
        kp[0] = [cx, head_y]
        kp[1] = [cx - 5 * scale, head_y - 3 * scale]
        kp[2] = [cx + 5 * scale, head_y - 3 * scale]
        kp[3] = [cx - 10 * scale, head_y]
        kp[4] = [cx + 10 * scale, head_y]

        # Shoulders (5: left, 6: right)
        sh_y = cy - 25 * scale
        kp[5] = [cx - 18 * scale, sh_y]
        kp[6] = [cx + 18 * scale, sh_y]

        # Elbows (7, 8) & Wrists (9, 10) with random arm angles
        for side, sh_idx, el_idx, wr_idx in [(-1, 5, 7, 9), (1, 6, 8, 10)]:
            a1 = rng.uniform(-0.8, 0.8)
            a2 = a1 + rng.uniform(-0.7, 0.7)
            kp[el_idx] = kp[sh_idx] + np.array([side * 22 * np.cos(a1), 22 * np.sin(a1)]) * scale
            kp[wr_idx] = kp[el_idx] + np.array([side * 20 * np.cos(a2), 20 * np.sin(a2)]) * scale

        # Hips (11: left, 12: right)
        hip_y = cy + 20 * scale
        kp[11] = [cx - 12 * scale, hip_y]
        kp[12] = [cx + 12 * scale, hip_y]

        # Knees (13, 14) & Ankles (15, 16) with slight leg variation
        for side, hip_idx, kn_idx, an_idx in [(-1, 11, 13, 15), (1, 12, 14, 16)]:
            dx1 = rng.uniform(-8, 10) * side
            dx2 = rng.uniform(-6, 8) * side
            kp[kn_idx] = kp[hip_idx] + np.array([dx1, 28]) * scale
            kp[an_idx] = kp[kn_idx] + np.array([dx2, 28]) * scale

        return np.clip(kp, 4, s - 5)

    def _render_image(self, kp: np.ndarray) -> torch.Tensor:
        img = Image.new("RGB", (self.img_size, self.img_size), (20, 20, 20))
        draw = ImageDraw.Draw(img)

        # Draw head circle around nose
        nx, ny = kp[0]
        r_head = 12
        draw.ellipse([nx - r_head, ny - r_head, nx + r_head, ny + r_head], outline=(230, 230, 230), width=3)

        # Draw limbs
        for i, j in self.SKELETON:
            draw.line([tuple(kp[i]), tuple(kp[j])], fill=(230, 230, 230), width=3)

        # Draw joint dots
        for x, y in kp:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 100, 100))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        keypoints = self._sample_keypoints(rng)
        img_tensor = self._render_image(keypoints)
        heatmaps = make_gaussian_heatmaps(
            keypoints, self.img_size, self.heatmap_size, self.sigma
        )
        return img_tensor, heatmaps
