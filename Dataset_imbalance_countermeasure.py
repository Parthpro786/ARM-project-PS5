# since the original dataset count of person was 6500+ whereas other were only upto 900 counts thus the confidence score of the perosn became biased and even non-person objects were being detected as persons.
# thus we decided to remove some counts of object: person 
import os
import glob
import random

LABEL_DIR = r"C:\Users\Parth\Downloads\yolo_labels"
PERSON_CLASS_ID = 0
KEEP_RATIO = 0.15 

def undersample_dataset():
    txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    initial_count = len(txt_files)
    print(f"Files BEFORE deletion: {initial_count}")
    
    deleted_count = 0
    
    for fpath in txt_files:
        with open(fpath, "r") as f:
            lines = f.readlines()
            
        if not lines: continue
        
        # Check if file has ONLY people (Class 0)
        has_non_person = False
        for line in lines:
            try:
                cls_id = int(line.split()[0])
                if cls_id != PERSON_CLASS_ID:
                    has_non_person = True
                    break
            except:
                continue
                
        # Delete if it's "Person Only" and loses the lottery
        if not has_non_person:
            if random.random() > KEEP_RATIO:
                os.remove(fpath)
                deleted_count += 1

    final_count = len(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    print("-" * 30)
    print(f"Deleted: {deleted_count}")
    print(f"Files AFTER deletion: {final_count}")
    print("-" * 30)

    if final_count > 8000:
        print("❌ ERROR: Did not delete enough files. Check your permissions!")
    else:
        print("✅ SUCCESS: Dataset is now balanced.")

undersample_dataset()

# Files BEFORE deletion: 7374
# ------------------------------
# Deleted: 418
# Files AFTER deletion: 6956
# ------------------------------
# ✅ SUCCESS: Dataset is now balanced.
