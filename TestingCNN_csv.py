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

# --------------------------------------------------------------------------------------------
# 1. Device Setup & Config
# --------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 72

# --------------------------------------------------------------------------------------------
# 2. Custom Dataset for CSV-based Inference
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
        
        # Build image filename (appends .png if missing)
        img_id = str(row['id'])
        if not img_id.endswith('.png'):
            img_id = f"{img_id}.png"
            
        img_path = os.path.join(self.img_dir, img_id)

        # Force 3-channel RGB (compatible with ResNet)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# --------------------------------------------------------------------------------------------
# 3. Test Transforms (Matches ResNet Requirements)
# --------------------------------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --------------------------------------------------------------------------------------------
# 4. Load Saved Checkpoint & Reconstruct Model
# --------------------------------------------------------------------------------------------
print("Loading model checkpoint...")
checkpoint = torch.load('model.pt', map_location=device, weights_only=True)

# Instantiate ResNet18 architecture
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

# Load saved weights (handles both wrapped state dict and direct dict)
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
    class_to_idx = checkpoint.get('class_to_idx', None)
else:
    model.load_state_dict(checkpoint)
    class_to_idx = None

model = model.to(device)
model.eval()


# --------------------------------------------------------------------------------------------
# 5. Data Loader Initialization
# --------------------------------------------------------------------------------------------
CSV_PATH = "E:/move work/year3/deeplearn/CNN/archive2/test/test.csv"
IMG_DIR = 'E:/move work/year3/deeplearn/CNN/archive2/test'

test_dataset = TestCSVDataset(
    csv_file=CSV_PATH,
    img_dir=IMG_DIR,
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --------------------------------------------------------------------------------------------
# 6. Inference Loop & CSV Export
# --------------------------------------------------------------------------------------------
all_preds = []

print("Generating predictions...")
with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        all_preds.extend(preds)

# Map integer prediction back to original class string label if mapped during training
# if class_to_idx is not None:
#     idx_to_class = {v: k for k, v in class_to_idx.items()}
#     all_preds = [idx_to_class[p] for p in all_preds]
final_labels = [p + 161 for p in all_preds]

# Save output to C:/out.csv
outXls = pd.read_csv(CSV_PATH)
outXls['label'] = final_labels
outXls.to_csv('E:/move work/year3/deeplearn/CNN/out2.csv', index=False)

print("Finished! Predictions successfully saved to E:/move work/year3/deeplearn/CNN/out.csv")