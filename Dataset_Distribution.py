import os
import glob

# Path to your labels (from your previous scripts)
LABEL_DIR = r"C:\Users\Parth\Downloads\yolo_labels"
CLASSES = [
    "person", "chair", "table", "bottle", 
    "sofa", "pottedplant", "tvmonitor"
]
# Note: I kept the display name "table" but mapped it to ID 2 (diningtable)

def count_classes():
    counts = {name: 0 for name in CLASSES}
    txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    
    print(f"Scanning {len(txt_files)} files...")
    
    for fpath in txt_files:
        with open(fpath, "r") as f:
            for line in f:
                try:
                    # Line format: class_id x y w h
                    cls_id = int(line.split()[0])
                    if 0 <= cls_id < len(CLASSES):
                        counts[CLASSES[cls_id]] += 1
                except ValueError:
                    continue

    print("-" * 30)
    print("DATASET DISTRIBUTION")
    print("-" * 30)
    for cls, count in counts.items():
        print(f"{cls:<10}: {count}")
    print("-" * 30)

    if counts['bottle'] == 0:
        print("🚨 CRITICAL ERROR: You have 0 bottle labels. Re-run Script 5 (XML Converter)!")
    elif counts['person'] > 10 * counts['bottle']:
        print("⚠️ WARNING: Massive imbalance. 'Person' dominates the dataset.")

count_classes()

# Scanning 6956 files...
# ------------------------------
# DATASET DISTRIBUTION
# ------------------------------
# person    : 2587
# chair     : 1554
# table     : 421
# bottle    : 974
# sofa      : 487
# pottedplant: 994
# tvmonitor : 632
# ------------------------------
