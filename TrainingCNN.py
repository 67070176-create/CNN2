#import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
#from sklearn.metrics import accuracy_score
from tqdm import tqdm

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
ensure_rgb = transforms.Lambda(lambda img: img.convert('RGB'))

def get_dynamic_transform(current_acc):
    """Dynamically scales augmentation intensity based on training accuracy."""
    if current_acc < 60.0:
        # Easy: Light affine only
        return transforms.Compose([
            ensure_rgb,
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif current_acc < 85.0:
        # Medium: Rotation + moderate scaling
        return transforms.Compose([
            ensure_rgb,
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Hard: Stronger affine + perspective warp
        return transforms.Compose([
            ensure_rgb,
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.90, 1.10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

test_transform = transforms.Compose([
    ensure_rgb,
    transforms.Resize((224, 224)),
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
# 3. Datasets & Split (Fixed: Separate transforms for train and test)
# --------------------------------------------------------------------------------------------
# Load datasets separately so test set doesn't get augmentations
class TransformedSubset(Dataset):
    """Wraps a Dataset subset to dynamically apply transformations."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
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
train_dataset = TransformedSubset(raw_train_subset, transform=get_dynamic_transform(0.0))
test_dataset = TransformedSubset(raw_test_subset, transform=test_transform)

# --------------------------------------------------------------------------------------------
# Class Balancing (Uncomment if classes are imbalanced) Using weight
# --------------------------------------------------------------------------------------------
targets = [raw_base_dataset.samples[i][1] for i in raw_train_subset.indices]
class_counts = np.bincount(targets)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# --------------------------------------------------------------------------------------------
# 4. DataLoaders
# --------------------------------------------------------------------------------------------
# If using sampler above, set sampler=sampler and replace shuffle=True with shuffle=False
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=sampler) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,)

print(f"Total: {len(raw_base_dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# --------------------------------------------------------------------------------------------
# 5. Model Architecture
# --------------------------------------------------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 1. Freeze ALL pre-trained weights
for param in model.parameters():
    param.requires_grad = False

#Add Drop-out 30% to the model
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(  # type: ignore
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Reduced learning rate to 1e-4 for transfer learning
EPOCHS = 15
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
# --------------------------------------------------------------------------------------------
# 6. Training Loop
# --------------------------------------------------------------------------------------------

print(f"Starting training on {device}...")
last_acc = 0.0

for epoch in range(EPOCHS):
    train_dataset.transform = get_dynamic_transform(last_acc)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", mininterval=20.0)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 50 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    last_acc = epoch_acc
    
    print(f"Epoch {epoch+1} Results -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

# --------------------------------------------------------------------------------------------
# 7. Evaluation & Model Saving
# --------------------------------------------------------------------------------------------
print("\nEvaluating on Test Set...")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = (test_correct / test_total) * 100
print(f"Final Test Accuracy: {test_acc:.2f}%")

torch.save(model.state_dict(), 'model.pt')
print("Model saved successfully as model.pt!")