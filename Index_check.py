import torch
import pandas as pd

# Load checkpoint
CHECKPOINT_PATH = 'model.pt'
CSV_PATH = "E:/move work/year3/deeplearn/CNN/test_dataset/test.csv"

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

print("=" * 60)
print("1. CHECKPOINT KEYS & DICTIONARY INSPECTION")
print("=" * 60)

if isinstance(checkpoint, dict):
    print("Checkpoint Keys Found:", list(checkpoint.keys()))
    
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        print("\n[SUCCESS] 'class_to_idx' found in checkpoint!")
        
        # Invert class_to_idx to get idx -> class folder name
        idx_to_folder = {v: k for k, v in class_to_idx.items()}
        
        print("\nSample Training Class Mappings (First 10 indices):")
        print("Index -> Folder Name in Training Dataset")
        for i in range(min(10, len(idx_to_folder))):
            print(f"  Index {i:2d} -> Folder '{idx_to_folder[i]}'")
    else:
        print("\n[WARNING] 'class_to_idx' is MISSING from checkpoint dictionary.")
        print("Model was likely saved with torch.save(model.state_dict()) only.")
else:
    print("\n[WARNING] Checkpoint is a raw state_dict, not a dictionary with metadata.")

print("\n" + "=" * 60)
print("2. CSV GROUND TRUTH vs PREDICTION FORMAT CHECK")
print("=" * 60)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"CSV Columns: {list(df.columns)}")
    print("\nFirst 3 Rows of Test CSV:")
    print(df.head(3))
    
    if 'gt_class_id' in df.columns:
        print("\nGround Truth Column 'gt_class_id' Sample Values:", df['gt_class_id'].head(5).tolist())
    if 'gt_path' in df.columns:
        print("Path Column 'gt_path' Sample Values:", df['gt_path'].head(5).tolist())
        
except Exception as e:
    print(f"Could not load CSV at {CSV_PATH}: {e}")

print("\n" + "=" * 60)
print("3. DIAGNOSTIC SUMMARY")
print("=" * 60)
print("If 'class_to_idx' is NOT saved in checkpoint:")
print("-> PyTorch assigned class indices based on how train folders were structured on disk.")
print("-> Print 'idx_to_folder' above to verify if Index 0 matches your expected starting class.")