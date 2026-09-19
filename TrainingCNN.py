# import pandas as pd
import numpy as np
# from Net import Net
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
# from sklearn.metrics import accuracy_score
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from torch.optim.lr_scheduler import CosineAnnealingLR

# --------------------------------------------------------------------------------------------
# 1. Transforms
# --------------------------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], 
        p=0.5
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomGrayscale(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0) # type: ignore
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------------------------------------------------
# 2. Config & Reproducibility
# --------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 72
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

generator = torch.Generator().manual_seed(SEED)

# --------------------------------------------------------------------------------------------
# 3. Datasets & Split
# --------------------------------------------------------------------------------------------
class TransformedSubset(Dataset):
    """Wraps a Dataset subset to dynamically apply transformations."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Load raw dataset once WITHOUT any transforms
raw_base_dataset = ImageFolder(root='dataset/round2')

train_size = int(0.8 * len(raw_base_dataset))
test_size = len(raw_base_dataset) - train_size

# Perform split on raw dataset
raw_train_subset, raw_test_subset = random_split(raw_base_dataset, [train_size, test_size], generator=generator)

# Wrap each subset with its dedicated transform
train_dataset = TransformedSubset(raw_train_subset, transform=train_transform)
test_dataset = TransformedSubset(raw_test_subset, transform=test_transform)

# --------------------------------------------------------------------------------------------
# Class Balancing
# --------------------------------------------------------------------------------------------
targets = [raw_base_dataset.samples[i][1] for i in raw_train_subset.indices]
class_counts = np.bincount(targets, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# --------------------------------------------------------------------------------------------
# 4. DataLoaders
# --------------------------------------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=sampler) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total: {len(raw_base_dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# --------------------------------------------------------------------------------------------
# 5. Model Architecture
# --------------------------------------------------------------------------------------------
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Unfreeze weights so model can fine-tune properly
for param in model.parameters():
    param.requires_grad = True

# Access classifier input features safely
in_features = getattr(model.classifier[1], 'in_features', 1280)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
EPOCHS = 15
optimizer = Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda') # type: ignore

# --------------------------------------------------------------------------------------------
# Early Stopping Helper Class
# --------------------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# --------------------------------------------------------------------------------------------
# 6. Training & Validation Loop
# --------------------------------------------------------------------------------------------
early_stopping = EarlyStopping(patience=4, min_delta=0.001)

print(f"Starting training on {device}...")

for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", mininterval=20.0)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 1. Cast forward pass to float16 using autocast
        with torch.amp.autocast('cuda'): # type: ignore
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)

        # 2. Backward pass & Optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 3. Accumulate metrics entirely on GPU
        batch_size = labels.size(0)
        running_loss += loss.detach() * batch_size
        total += batch_size
        correct += predicted.eq(labels).sum()

    scheduler.step()
    
    train_acc = (correct.item() / total) * 100
    epoch_loss = (running_loss.item() / total)

    # Validation Phase
    model.eval()
    val_loss_running = torch.tensor(0.0, device=device)
    val_correct = torch.tensor(0, device=device)
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
           
            with torch.amp.autocast('cuda'): # type: ignore
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                _, val_pred = val_outputs.max(1)
                
            batch_size = val_labels.size(0)
            val_loss_running += v_loss.detach() * batch_size
            val_total += batch_size
            val_correct += val_pred.eq(val_labels).sum()

    val_acc = (val_correct.item() / val_total) * 100
    val_loss = (val_loss_running.item() / val_total)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} -> Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"\n[Early Stopping Triggered] Stopping early at epoch {epoch+1}.")
        break

# Restore best weights once after training ends
if early_stopping.best_model_wts is not None:
    model.load_state_dict(early_stopping.best_model_wts)
    print(f"Loaded best model weights (Best Val Loss: {early_stopping.best_loss:.4f})")

# --------------------------------------------------------------------------------------------
# 7. Checkpoint Saving
# --------------------------------------------------------------------------------------------
checkpoint = {
    'model_state': model.state_dict(),
    'class_to_idx': raw_base_dataset.class_to_idx
}

torch.save(checkpoint, 'model.pt')
print("Model saved successfully as model.pt!")