"""
EuroSAT RGB CNN Application Script
================================

Author:
    Teng Zhang (mi-geo.github.io)
    Applied the eurosat_best_cnn.pth model to selected images.

This script:
    - Loads the trained EuroSAT CNN model
    - Applies the same preprocessing transform used during training
    - Performs prediction on a single image or a folder of images
    - Plots and saves a grid of predicted images

Used ChatGPT to clean up and annotate the code.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -----------------------
# 1. Define EuroSAT classes (fixed 10-class taxonomy)
# -----------------------
classes = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop',
    'Residential', 'River', 'SeaLake'
]

num_classes = len(classes)

# Detect GPU if available; otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# 2. Load trained model
# -----------------------
# NOTE: SimpleEuroSATCNN must be defined or imported before this block.
model = SimpleEuroSATCNN(num_classes=num_classes).to(device)

# Load weights from your trained checkpoint.
# map_location=device ensures correct loading even if model was trained on a GPU.
model.load_state_dict(torch.load("Spatial/eurosat_best_cnn.pth", map_location=device))

# Set model to inference mode — disables dropout, uses running batchnorm stats, etc.
model.eval()


# -----------------------
# 3. Transform for the input image
# -----------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),       # Resize to training-size input
    transforms.ToTensor(),             # Convert PIL → (C,H,W) tensor in [0,1]
    transforms.Normalize(              # Normalize using same stats as training
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -----------------------
# 4. Prediction function
# -----------------------

def predict_image(path):
    """
    Load an image, apply transform, run model inference, and return label.
    """

    # Load with PIL and ensure RGB (important because EuroSAT tiles are RGB)
    img = Image.open(path).convert("RGB")

    # Apply transforms → gives tensor of shape (3, 64, 64)
    img_t = transform(img).unsqueeze(0).to(device)
    # unsqueeze(0) converts (3,64,64) → (1,3,64,64) to make a batch of size 1

    with torch.no_grad():  # ensures no gradient computation during inference
        outputs = model(img_t)  # shape (1, num_classes)
        _, predicted = torch.max(outputs, 1)  # index of highest logit
        label = classes[predicted.item()]     # convert to class name

    return label


# -----------------------
# 5. Quick test on one image
# -----------------------

image_path = "Spatial/sample/tile3.png"
pred = predict_image(image_path)
print("Predicted class:", pred)


# -----------------------
# 6. Predict a whole folder
# -----------------------
from pathlib import Path
tiles_dir = Path("Spatial/sample")  # folder containing .png tiles
png_paths = sorted(tiles_dir.glob("*.png"))

images = []   # raw PIL images (for plotting)
labels = []   # predicted labels

with torch.no_grad():
    for path in png_paths:
        img = Image.open(path).convert("RGB")  # load each tile
        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)

        # Get predicted class index
        pred_idx = logits.argmax(dim=1).item()
        label = classes[pred_idx]

        images.append(img)
        labels.append(label)


# -----------------------
# 7. Plot images + predicted labels in a grid
# -----------------------
import math
import matplotlib.pyplot as plt


def plot_image_grid(image_paths, labels, output="image_grid.png", cols=4):
    """
    Display a grid of images with labels.
    'image_paths' should be a list of file paths.
    """

    images = []

    # Load all images safely (avoids crashing if an image is corrupted)
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: could not load {p} ({e})")

    n = len(images)
    if n == 0:
        print("No valid images found!")
        return

    # Determine number of rows based on number of columns
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten axes whether 1D or 2D
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = axes.flatten()

    # Plot each image with its predicted label
    for ax, img, label in zip(axes, images, labels):
        img = img.resize((256, 256))  # upsample for display clarity
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    # Turn off unused axes if images < grid size
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()

    # Save figure BEFORE showing
    fig.savefig(output, dpi=150, bbox_inches="tight")

    print(f"Saved grid to: {output}")


# Use paths + labels from earlier prediction loop
image_paths = png_paths

plot_image_grid(image_paths, labels, output="eurosat_grid.png", cols=4)


# -----------------------
# 8. Display one image (sanity check)
# -----------------------
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(png_paths[0])
plt.imshow(img)
plt.axis("off")
plt.show()