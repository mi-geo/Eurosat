"""
EuroSAT RGB CNN Training Script
================================

Train a simple convolutional neural network (CNN) on the EuroSAT RGB dataset.
This script is part of my personal deep learning 101 learning exercises.

Author:
    Teng Zhang (mi-geo.github.io)

Based on ideas learned from:
    https://www.kaggle.com/code/nilesh789/land-cover-classification-with-eurosat-dataset

Used ChatGPT to clean up and comment the code for clarity.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ================================================================
# 1. Configuration
# ================================================================
DATA_DIR = "C:/Users/46798566/Downloads/EuroSAT/2750"   # Directory containing 10-class EuroSAT RGB tiles
BATCH_SIZE = 64
NUM_EPOCHS = 10
VAL_SPLIT = 0.2
LR = 1e-3  # learning rate for Adam optimizer

# Auto-detect GPU when available; otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================================================================
# 2. Dataset & Transforms
# ================================================================

# Define preprocessing for each image.
# Resize ensures uniform 64×64 tiles; Normalize matches the inference transform.
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # EuroSAT tiles are already 64×64, but forcing this helps avoid mismatches
    transforms.ToTensor(),        # Convert PIL image → PyTorch Tensor (C,H,W) in range [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Simple mean/std normalization; improves training stability
                         std=[0.5, 0.5, 0.5]),
])

# ImageFolder automatically assigns labels based on folder names:
# DATA_DIR/
#   ├── AnnualCrop/
#   ├── Forest/
#   ├── ...
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
num_classes = len(dataset.classes)

print("Found classes:", dataset.classes)
print("Total images:", len(dataset))

# Train/validation split—commonly used ratio: 80/20.
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

# random_split creates two sub-datasets referencing the same ImageFolder
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoaders handle batching, shuffling, and parallel loading
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ================================================================
# 3. Model Definition — Simple CNN
# ================================================================
class SimpleEuroSATCNN(nn.Module):
    """
    A compact CNN designed for 64×64 EuroSAT RGB imagery.
    Follows the standard pattern:
        conv → relu → pool → conv → relu → pool → conv → relu → pool → FC layers
    """

    def __init__(self, num_classes):
        super().__init__()

        # Convolution layers:
        #   Input channels: 3 (RGB)
        #   Output channels: 32 → 64 → 128
        # kernel_size=3 with padding=1 keeps spatial size unchanged before pooling.
        self.conv1 = nn.Conv2d(3,   32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Max pooling reduces width/height by half each time.
        # After 3 pooling layers:
        #     64 → 32 → 16 → 8 pixels
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers:
        # Flattened feature map size = 128 channels * 8 * 8 spatial size
        self.fc1 = nn.Linear(128 * 8 * 8, 256)     # Dense layer (feature extraction)
        self.fc2 = nn.Linear(256, num_classes)     # Final output logits

        # Dropout helps regularize and avoid overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Each conv → ReLU → pool reduces image size
        x = self.pool(F.relu(self.conv1(x)))  # Output shape: (32, 32×32)
        x = self.pool(F.relu(self.conv2(x)))  # Output shape: (64, 16×16)
        x = self.pool(F.relu(self.conv3(x)))  # Output shape: (128, 8×8)

        # Flatten tensor from (batch, 128, 8, 8) → (batch, 8192)
        x = x.view(x.size(0), -1)

        # Dense layers with dropout for regularization
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # No softmax is applied here because CrossEntropyLoss expects raw logits
        return x


# Instantiate model + optimizer + loss function
model = SimpleEuroSATCNN(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()  # good default for multi-class classification

print(model)


# ================================================================
# 4. Training & Validation
# ================================================================

def train_one_epoch(epoch):
    """
    Executes one full pass over the training dataset.
    """
    model.train()  # set model to training mode (enables dropout)
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()        # Always reset gradients before computing new ones
        outputs = model(images)      # Forward pass
        loss = criterion(outputs, labels)  # Compute CE loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Apply weight updates

        # Multiply by batch size because DataLoader returns mini-batch loss
        running_loss += loss.item() * images.size(0)

        # Occasionally print batch status
        if (batch_idx + 1) % 50 == 0:
            print(f"  [Batch {batch_idx+1}/{len(train_loader)}] "
                  f"loss: {loss.item():.4f}")

    # Return average loss over all training samples
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate():
    """
    Evaluate the model on the validation dataset.
    No gradients required. Computes:
        - Average validation loss
        - Classification accuracy
    """
    model.eval()  # sets dropout/batchnorm into inference mode
    correct, total = 0, 0
    eval_loss = 0.0

    with torch.no_grad():  # disable grad tracking for speed + memory savings
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * images.size(0)

            # Predictions from logits
            preds = outputs.argmax(dim=1)  # highest-probability class
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = eval_loss / len(val_loader.dataset)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


# ================================================================
# 5. Main Training Loop
# ================================================================
best_val_acc = 0.0  # Track best accuracy across epochs

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    train_loss = train_one_epoch(epoch)
    val_loss, val_acc = evaluate()

    print(f"Epoch {epoch+1}: "
          f"train loss = {train_loss:.4f}, "
          f"val loss = {val_loss:.4f}, "
          f"val acc = {val_acc:.4f}")

    # Save checkpoint if validation improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "eurosat_best_cnn.pth")
        print(f"  ✓ New best model saved with val acc = {best_val_acc:.4f}")

print("\nTraining complete.")
print("Best validation accuracy:", best_val_acc)
print("Class labels:", dataset.classes)