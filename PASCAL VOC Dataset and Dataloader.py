import torch
import os
import cv2
import glob
import numpy as np
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
IMG_SIZE = 320     
GRID_SIZE = 20      
NUM_ANCHORS = 3
NUM_CLASSES = 7
ANCHORS = [(0.05, 0.15), (0.30, 0.30), (0.60, 0.60)]

# --- PATHS (Update these!) ---
# Where are your .jpg files? 
# (Assuming they are in 'images' folder inside Downloads, or VOCdevkit. Update this!)
IMAGE_DIR = r"C:\Users\Parth\Downloads\archive(1)\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"

# Where are your .txt files? (We verified this path works)
LABEL_DIR = r"C:\Users\Parth\Downloads\yolo_labels"

class AutoDiscoveryDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        # 1. Auto-Discovery: Find all JPGs
        print(f" Scanning {self.image_dir} for images...")
        all_images = glob.glob(os.path.join(self.image_dir, "*.jpg"))
        
        self.valid_pairs = []
        
        # 2. Pair them with Labels
        for img_path in all_images:
            # Get filename without extension (e.g., "000123")
            file_id = os.path.splitext(os.path.basename(img_path))[0]
            
            # Check if corresponding label exists
            label_path = os.path.join(self.label_dir, file_id + ".txt")
            
            if os.path.exists(label_path):
                self.valid_pairs.append((img_path, label_path))
        
        if len(self.valid_pairs) == 0:
            print(f"❌ CRITICAL: Found 0 valid image-label pairs!")
            print(f"   Check your IMAGE_DIR: {self.image_dir}")
            print(f"   Check your LABEL_DIR: {self.label_dir}")
        else:
            print(f"✅ Success! Found {len(self.valid_pairs)} valid training pairs.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]

        # 1. Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0 
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # 2. Load Label
        target = torch.zeros((NUM_ANCHORS, GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES))           #important
        
        with open(label_path) as f:
            lines = f.readlines()
            
        for line in lines:
            data = list(map(float, line.strip().split()))
            if len(data) < 5: continue
            
            cls_id = int(data[0])
            x, y, w, h = data[1], data[2], data[3], data[4]
            
            if cls_id >= NUM_CLASSES: continue

            gx = int(x * GRID_SIZE)
            gy = int(y * GRID_SIZE)
            gx = min(max(0, gx), GRID_SIZE - 1)
            gy = min(max(0, gy), GRID_SIZE - 1)

            # Anchor Math
            best_iou = 0.0
            best_anchor_idx = 0
            gt_area = w * h
            
            for i, (aw, ah) in enumerate(ANCHORS):
                inter = min(w, aw) * min(h, ah)
                union = gt_area + (aw * ah) - inter
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor_idx = i

            target[best_anchor_idx, gy, gx, 0] = 1.0 
            target[best_anchor_idx, gy, gx, 1:5] = torch.tensor([x, y, w, h])
            target[best_anchor_idx, gy, gx, 5 + cls_id] = 1.0

        return img, target

# --- CREATE DATASET ---
# Note: We don't use train.txt anymore!
train_dataset = AutoDiscoveryDataset(IMAGE_DIR, LABEL_DIR)

# Split into Train/Val (Optional, but good practice)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset,
                          batch_size=8, 
                          shuffle=True, 
                          drop_last=True)

# We already created 'val_subset' in the previous step
test_loader = DataLoader(
    val_subset, 
    batch_size=8, 
    shuffle=False,      # No need to shuffle for testing
    drop_last=True
)

print(f"✅ FINAL STATUS: Data Pipeline Ready.")
print(f"   - Training Batches: {len(train_loader)}")
print(f"   - Testing Batches:  {len(test_loader)}")
print("Dataloader ready.")

#  Scanning C:\Users\Parth\Downloads\archive(1)\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages for images...
# ✅ Success! Found 3519 valid training pairs.

# ✅ FINAL STATUS: Data Pipeline Ready.
#    - Training Batches: 395
#    - Testing Batches:  44
# ✅ DataLoader ready.
