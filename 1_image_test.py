# use this code for object detecting on a specific image with its relative filepath (here it is supposed that the image is in downloads folder)
import cv2
import torch
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
# MUST match what you just trained with (320 or 416)
IMG_SIZE = 320      
S = 20              # 416/16 = 26. (Use 20 if you trained on 320)
A = 3               
C = 7   
CONF_THRESH = 0.15  # Low threshold for boxes

VOC_CLASSES = [
    "person", "chair", "table", "bottle", 
    "sofa", "pottedplant", "tvmonitor"
]

# Path to your Downloads folder
DOWNLOADS_DIR = r"C:\Users\Parth\Downloads"

device = torch.device("cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = TinyYOLOv2_Brevitas(num_classes=C, num_anchors=A)
try:
    state = torch.load("tinyyolo.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval() # Keep Batch Norm active
    model.to(device)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def preprocess(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return None, None
        
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        print("❌ Could not read image. Check file format.")
        return None, None
        
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device), img_raw

def analyze_image(img_path):
    tensor_img, original_img = preprocess(img_path)
    if tensor_img is None: return

    with torch.no_grad():
        preds = model(tensor_img)

    preds = preds.squeeze(0) 
    current_S = preds.size(1) 
    preds = preds.permute(1, 2, 0).view(current_S, current_S, A, 5 + C)

    # ---------------------------------------------------------
    # "FORCED CLASSIFIER" LOGIC
    # We look for the absolute highest signal in the grid,
    # regardless of whether it meets the threshold for a box.
    # ---------------------------------------------------------
    best_overall_conf = -1.0
    best_class_name = "Unknown"
    best_grid_loc = (0,0)
    
    # Store valid detections for drawing
    detections = []

    for gy in range(current_S):
        for gx in range(current_S):
            for a in range(A):
                # Raw scores
                obj_raw = torch.sigmoid(preds[gy, gx, a, 0]).item()
                cls_logits = preds[gy, gx, a, 5:]
                cls_probs = torch.sigmoid(cls_logits)
                
                # Get the winner for this specific anchor
                current_cls_id = torch.argmax(cls_probs).item()
                current_cls_score = cls_probs[current_cls_id].item()
                
                total_conf = obj_raw * current_cls_score
                
                # Update "Best Guess"
                if total_conf > best_overall_conf:
                    best_overall_conf = total_conf
                    best_class_name = VOC_CLASSES[current_cls_id]
                    best_grid_loc = (gx, gy)

                # Store box if it's decent
                if total_conf > CONF_THRESH:
                    tx, ty = preds[gy, gx, a, 1], preds[gy, gx, a, 2]
                    tw, th = preds[gy, gx, a, 3], preds[gy, gx, a, 4]
                    
                    cx = (gx + torch.sigmoid(tx).item()) / current_S
                    cy = (gy + torch.sigmoid(ty).item()) / current_S
                    bw = torch.sigmoid(tw).item() 
                    bh = torch.sigmoid(th).item()
                    
                    h, w = original_img.shape[:2]
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    detections.append((x1, y1, x2, y2, current_cls_id, total_conf))

    # -----------------------------
    # PRINT RESULTS
    # -----------------------------
    print("-" * 40)
    print(f"🖼️ ANALYZING: {os.path.basename(img_path)}")
    print("-" * 40)
    
    print(f"🔍 BEST GUESS (Forced): {best_class_name}")
    print(f"📊 Confidence Signal:   {best_overall_conf:.4f}")
    print(f"📍 Grid Location:       {best_grid_loc}")

    if best_overall_conf < 0.05:
        print("\n⚠️ VERDICT: The model sees NOTHING. (Signal is noise)")
    elif best_overall_conf < 0.20:
        print("\n⚠️ VERDICT: It sees a faint shape, but is unsure.")
    else:
        print("\n✅ VERDICT: Strong detection found!")

    # -----------------------------
    # DRAW & SHOW
    # -----------------------------
    for (x1, y1, x2, y2, cid, conf) in detections:
        color = (0, 255, 0)
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
        label = f"{VOC_CLASSES[cid]} {conf:.2f}"
        cv2.putText(original_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Analysis Result", original_img)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# RUN IT HERE
# -----------------------------
# Change this to your file name!
IMAGE_FILENAME = "bottle_test.jpg" 

full_path = os.path.join(DOWNLOADS_DIR, IMAGE_FILENAME)
analyze_image(full_path)
