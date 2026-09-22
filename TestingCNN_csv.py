import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# 1. Device Setup & Paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 72

TRAIN_DIR = "E:/move work/year3/deeplearn/CNN/dataset/round2"
CSV_PATH = "E:/move work/year3/deeplearn/CNN/archive2/test/test.csv"
IMG_DIR = "E:/move work/year3/deeplearn/CNN/archive2/test"
OUTPUT_PATH = "E:/move work/year3/deeplearn/CNN/out2.csv"

# 2. Build Class Map (Handles gaps in folder numbers automatically)
class_folders = sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
idx_to_class = {idx: int(folder_name) for idx, folder_name in enumerate(class_folders)}

# 3. Custom Dataset
class TestCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_id = str(row['id']).strip()
        if not img_id.endswith('.png'):
            img_id = f"{img_id}.png"
            
        # Clean filename to prevent path resolution errors
        img_filename = os.path.basename(img_id)
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract Ground Truth Label from prefix (e.g. "161_1.png" -> 161)
        filename_without_ext = os.path.splitext(img_filename)[0]
        ground_truth_label = int(filename_without_ext.rsplit('_', 1)[0])

        return image, ground_truth_label

# 4. Transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Load Checkpoint
print("Loading model checkpoint...")
checkpoint = torch.load('model.pt', map_location=device, weights_only=True)

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# 6. DataLoader & Inference
test_dataset = TestCSVDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_targets = []

print("Generating predictions and computing accuracy...")
with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

# Map predictions to true labels
final_labels = [idx_to_class[p] for p in all_preds]

# Calculate Accuracy
correct_count = sum(p == t for p, t in zip(final_labels, all_targets))
total_count = len(all_targets)
acc_percentage = (correct_count / total_count) * 100

print("\n" + "=" * 45)
print(f"Total Samples    : {total_count}")
print(f"Correct Count    : {correct_count}")
print(f"Overall Accuracy : {acc_percentage:.2f}%")
print("=" * 45 + "\n")

outXls = pd.read_csv(CSV_PATH)
outXls['predicted_label'] = final_labels
outXls['ground_truth'] = all_targets
outXls.to_csv(OUTPUT_PATH, index=False)

print(f"Finished! Output written to: {OUTPUT_PATH}")