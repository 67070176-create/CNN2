import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

# 1. Define Custom Dataset Wrapper to safely handle RGB conversion
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

# 2. Evaluation helper function
def evaluate(model, loader, device, description):
    correct = 0
    total = 0
    print(f"Evaluating {description}...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description, mininterval=20.0):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = (correct / total) * 100
    print(f"{description} Accuracy: {acc:.2f}% ({correct}/{total} correct)\n")
    return acc

# 3. Guard entry point for Windows Multiprocessing
if __name__ == '__main__':
    # Setup device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Check GPU drivers or PyTorch CUDA installation.")

    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Dataset & Transforms
    SEED = 42
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_dataset = ImageFolder(root='dataset/round2')

    train_ratio = 0.8
    train_size = int(train_ratio * len(raw_dataset))
    test_size = len(raw_dataset) - train_size

    train_subset, test_subset = random_split(
        raw_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset = TransformedSubset(train_subset, transform=transform)
    test_dataset = TransformedSubset(test_subset, transform=transform)

    # Multi-worker DataLoaders (now safely supported on Windows)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # Model architecture & Checkpoint setup
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    num_classes = 72

    model.fc = nn.Sequential( # type: ignore
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    checkpoint = torch.load('model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # Run Evaluations
    model.eval()

    #train_acc = evaluate(model, train_loader, device, "Train Set")
    test_acc = evaluate(model, test_loader, device, "Validate Set")

    print("--- Final Results ---")
    #print(f"Train Accuracy:    {train_acc:.2f}%")
    print(f"Validate Accuracy: {test_acc:.2f}%")