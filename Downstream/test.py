import torch
from torchvision.utils import save_image

# Image size
H, W = 224, 224

# -----------------------
# Gaussian noise
# -----------------------
# N(0,1)
gaussian = torch.randn(3, H, W)

# Normalize to [0,1] for visualization
gaussian_vis = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

save_image(gaussian_vis, "gaussian_noise.png")

# -----------------------
# Uniform noise
# -----------------------
# U(0,1)
uniform = torch.rand(3, H, W)

save_image(uniform, "uniform_noise.png")

print("Saved:")
print("  gaussian_noise.png")
print("  uniform_noise.png")
