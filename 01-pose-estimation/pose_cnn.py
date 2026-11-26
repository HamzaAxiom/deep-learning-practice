# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseCNN(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseCNN, self).__init__()
        
        self.num_keypoints = num_keypoints
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.head = nn.Conv2d(128, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.deconv(x))
        x = self.head(x)
        return x

        
        

# %% training

from torch.utils.data import Dataset, DataLoader
from stick_dataset import StickFigureDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseCNN(num_keypoints=17).to(device)
dataset = StickFigureDataset(num_samples=400)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# %% training loop
n_epoch = 30
for epoch in range(n_epoch):
    for i, (images, heatmaps) in enumerate(dataloader):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Epoch [{epoch}/{n_epoch}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')
# %%
# %% evaluate & visualize
import matplotlib.pyplot as plt

test_ds = StickFigureDataset(num_samples=5, seed=999)
img, target_hm = test_ds[0]

model.eval()
with torch.no_grad():
    pred_hm = model(img.unsqueeze(0).to(device))[0].cpu()  # (17, 56, 56)

# Extract (x, y) from each predicted heatmap via argmax
flat_idx = pred_hm.view(17, -1).argmax(dim=1)
pred_y = (flat_idx // 56) * (224 / 56)
pred_x = (flat_idx % 56) * (224 / 56)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img.permute(1, 2, 0))
axes[0].scatter(pred_x, pred_y, c="cyan", s=40)
axes[0].set_title("Input + Predicted Joints")

axes[1].imshow(target_hm.sum(dim=0), cmap="hot")
axes[1].set_title("Target Heatmaps")

axes[2].imshow(pred_hm.sum(dim=0), cmap="hot")
axes[2].set_title("Predicted Heatmaps")
plt.show()
# %%
# %% accuracy (PCK & Mean Pixel Error)
def hm_to_coords(hm, img_size=224):
    # hm: (B, 17, 56, 56) -> (B, 17, 2) in pixel space
    B, K, H, W = hm.shape
    idx = hm.view(B, K, -1).argmax(dim=-1)
    y = (idx // W).float() * (img_size / H)
    x = (idx % W).float() * (img_size / W)
    return torch.stack([x, y], dim=-1)

test_ds = StickFigureDataset(num_samples=50, seed=999)
test_loader = DataLoader(test_ds, batch_size=32)

model.eval()
dists = []
with torch.no_grad():
    for imgs, target_hms in test_loader:
        pred_hms = model(imgs.to(device)).cpu()
        pred_xy = hm_to_coords(pred_hms)
        true_xy = hm_to_coords(target_hms)
        dists.append(torch.norm(pred_xy - true_xy, dim=-1))

dists = torch.cat(dists, dim=0)  # (N, 17)
print(f"Mean Pixel Error: {dists.mean():.2f} px")
print(f"PCK @ 10px: {(dists < 10).float().mean() * 100:.1f}%")
print(f"PCK @  5px: {(dists < 5).float().mean() * 100:.1f}%")
# %%
