import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
#from Net import Net

# 1. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Define Custom Dataset Wrapper to safely handle RGB conversion
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = x.convert('RGB')  # Simple, clean, and works in both scripts
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# 3. Load dataset & set transforms
SEED = 42
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load raw dataset without transforms
raw_base_dataset = ImageFolder(root='archive/synthetic_test_set')  # Ensure this points to your dataset path

train_size = int(0.8 * len(raw_base_dataset))
test_size = len(raw_base_dataset) - train_size

generator = torch.Generator().manual_seed(SEED)
_, raw_test_subset = random_split(raw_base_dataset, [train_size, test_size], generator=generator)

# Wrap test subset
test_dataset = TransformedSubset(raw_test_subset, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Load the model architecture
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
num_classes = 72

# Update Dropout to 0.5 to match Training script
in_features = getattr(model.classifier[1], 'in_features', 1280)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)

# 5. Load dictionary checkpoint correctly
checkpoint = torch.load('model.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state'])  # Key change: Load 'model_state'
model = model.to(device)

# 6. Run Evaluation
model.eval()

correct = 0
total = 0

print("Evaluating model.pt on test set...")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = (correct / total) * 100
print(f"\nFinal Test Accuracy: {test_acc:.2f}% ({correct}/{total} correct)")