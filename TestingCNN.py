import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
import torch.nn as nn

# 1. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Recreate the dataset split (using the SAME seed as training)
SEED = 42
ensure_rgb = transforms.Lambda(lambda img: img.convert('RGB'))
transform = transforms.Compose([
    ensure_rgb,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = ImageFolder(root='archive/synthetic_test_set', transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(SEED)
_, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Load the model architecture and saved weights
# If you used ResNet18:
model = resnet18()
num_ftrs = model.fc.in_features
num_classes = 72

model.fc = nn.Sequential(  # type: ignore
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, num_classes)
)

model.load_state_dict(torch.load('model.pt', map_location=device))
model = model.to(device)

# 4. Run Evaluation
model.eval()  # Disables dropout/batchnorm updates

correct = 0
total = 0

print("Evaluating model.pt on test set...")

with torch.no_grad():  # Disables gradient computation (faster & uses less GPU memory)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = (correct / total) * 100
print(f"\nFinal Test Accuracy: {test_acc:.2f}% ({correct}/{total} correct)")