import numpy as np
from PIL import Image
from pathlib import Path

def load_image(path: Path):
    img = Image.open(path)
    img_np = np.array(img)
    if img_np.ndim == 2:  # grayscale
        img_np = img_np[:, :, np.newaxis]
    return img_np.astype(np.float32)  # for numerical stability

# Load image
image_path = Path("download.jpeg")
image = load_image(image_path)

H, W, C = image.shape
f_h, f_w, f_c = 3, 3, 3
stride = 1
padding = 0

# Create random filter of shape (3, 3, 3)
kernel = np.random.rand(f_h, f_w, f_c).astype(np.float32)

# Calculate output feature map shape
out_h = (H - f_h + 2 * padding) // stride + 1
out_w = (W - f_w + 2 * padding) // stride + 1
output = np.zeros((out_h, out_w), dtype=np.float32)

# Apply convolution
for j in range(out_h):       # height axis (rows)
    for i in range(out_w):   # width axis (cols)
        patch = image[j:j+f_h, i:i+f_w, :]  # shape: (3, 3, 3)
        output[j, i] = np.sum(patch * kernel)

print("Feature map shape:", output.shape)
print(output)
