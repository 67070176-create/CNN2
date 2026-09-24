import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# --------------------------------------------------------------------------------------------
# 1. Device Setup & Configuration
# --------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")
NUM_CLASSES = 72

CSV_PATH = "E:/move work/year3/deeplearn/CNN/test_dataset/test.csv"
IMG_DIR = "E:/move work/year3/deeplearn/CNN/test_dataset"
OUTPUT_PATH = "E:/move work/year3/deeplearn/CNN/out.csv"


# --------------------------------------------------------------------------------------------
# 2. Complete Folder Name -> 3-Digit Class ID Mapping
# --------------------------------------------------------------------------------------------
FOLDER_TO_CLASS_ID = {
    '161': 101, '162': 102, '163': 103, '164': 104, '167': 105, '168': 106, '169': 107,
    '170': 108, '171': 109, '173': 110, '175': 111, '176': 112, '177': 113, '178': 114,
    '179': 115, '180': 116, '181': 117, '182': 118, '183': 119, '184': 120, '185': 121,
    '186': 122, '187': 123, '188': 124, '189': 125, '190': 126, '191': 127, '192': 128,
    '193': 129, '194': 130, '195': 131, '196': 132, '197': 133, '199': 134, '200': 135,
    '201': 136, '202': 137, '203': 138, '204': 139, '205': 140, '206': 141,
    '207': 201, '209': 202, '210': 203, '212': 204, '213': 205, '214': 206, '215': 207,
    '216': 208, '217': 209, '224': 210, '225': 211, '226': 212, '227': 213, '228': 214,
    '229': 215, '230': 216, '231': 217, '232': 218, '233': 219, '234': 220, '236': 221,
    '240': 301, '241': 302, '242': 303, '243': 304, '244': 305, '245': 306, '246': 307,
    '247': 308, '248': 309, '249': 310
}


# --------------------------------------------------------------------------------------------
# 3. Custom Dataset
# --------------------------------------------------------------------------------------------
class TestCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        raw_path = str(row['gt_path']).strip()
        img_filename = os.path.basename(raw_path)
        if not img_filename.endswith('.jpg'):
            img_filename = f"{img_filename}.jpg"

        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        gt_class_id = int(row['gt_class_id']) if 'gt_class_id' in row and pd.notna(row['gt_class_id']) else -1

        return image, gt_class_id


# --------------------------------------------------------------------------------------------
# 4. Standard Transforms
# --------------------------------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --------------------------------------------------------------------------------------------
# 5. Load Checkpoint & Auto-Map Model Index -> Folder -> Class ID
# --------------------------------------------------------------------------------------------
print("Loading model checkpoint...")
checkpoint = torch.load('model.pt', map_location=device, weights_only=False)

# Read training class_to_idx directly from checkpoint
class_to_idx = checkpoint['class_to_idx']

# STEP 1: Auto-invert model dictionary (0..71 -> '161', '162', ...)
idx_to_folder = {idx: str(folder_name) for folder_name, idx in class_to_idx.items()}

# STEP 2: Map model index to official 3-digit class_id
idx_to_class_id = {
    idx: FOLDER_TO_CLASS_ID[folder] 
    for idx, folder in idx_to_folder.items()
}

print("\n--- Model Index Auto-Mapping Sample ---")
for idx in range(min(5, len(idx_to_class_id))):
    print(f"Model Pred Index {idx:2d} -> Folder Name '{idx_to_folder[idx]}' -> Official Class ID {idx_to_class_id[idx]}")

# Re-build ResNet-18 model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()


# --------------------------------------------------------------------------------------------
# 6. DataLoader & Inference
# --------------------------------------------------------------------------------------------
test_dataset = TestCSVDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_targets = []

print("\nGenerating predictions...")
with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

# Map raw model output indices to official 3-digit class_ids
final_class_ids = [idx_to_class_id[p] for p in all_preds]

# Calculate Accuracy
if len(all_targets) > 0 and all_targets[0] != -1:
    correct_count = sum(p == t for p, t in zip(final_class_ids, all_targets))
    total_count = len(all_targets)
    acc_percentage = (correct_count / total_count) * 100

    print("\n" + "=" * 45)
    print(f"Total Samples    : {total_count}")
    print(f"Correct Count    : {correct_count}")
    print(f"Overall Accuracy : {acc_percentage:.2f}%")
    print("=" * 45 + "\n")

# Save output CSV formatted for submission
outXls = pd.read_csv(CSV_PATH)
outXls['prediction'] = final_class_ids
outXls.to_csv(OUTPUT_PATH, index=False)

print(f"Finished! Output written to: {OUTPUT_PATH}")