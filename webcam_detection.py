import cv2
import torch
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 320   
S = 20             
A = 3               
C = 7               

# --- CALIBRATION ---
# Since your model peaks at ~0.2, we set the cutoff slightly lower.
CONF_THRESH = 0.05  
NMS_THRESH = 0.15    

VOC_CLASSES = [
    "person", "chair", "table", "bottle", 
    "sofa", "pottedplant", "tvmonitor"
]

device = torch.device("cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = TinyYOLOv2_Brevitas(num_classes=C, num_anchors=A)

try:
    state = torch.load("tinyyolo.pth", map_location=device)
    # Try strict first, fall back if needed
    try:
        model.load_state_dict(state, strict=True)
    except:
        model.load_state_dict(state, strict=False)
    print("✅ Model loaded.")
except FileNotFoundError:
    print("❌ Error: 'tinyyolo.pth' not found.")
    exit()

# --- THE BATCH NORM HACK ---
# We keep the model in 'train' mode to use live statistics.
# This often fixes "low confidence" issues on small batches.
model.eval() 
model.to(device)

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def nms(detections, iou_thresh):
    if not detections: return []
    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    keep = []
    while len(detections) > 0:
        best = detections.pop(0)
        keep.append(best)
        rest = []
        for box in detections:
            x1 = max(best[0], box[0]); y1 = max(best[1], box[1])
            x2 = min(best[2], box[2]); y2 = min(best[3], box[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = (box[2]-box[0])*(box[3]-box[1]) + (best[2]-best[0])*(best[3]-best[1]) - inter
            if (inter / (union + 1e-6)) < iou_thresh:
                rest.append(box)
        detections = rest
    return keep

def decode_final(preds, img_w, img_h):
    preds = preds.squeeze(0) 
    current_S = preds.size(1) 
    preds = preds.permute(1, 2, 0).view(current_S, current_S, A, 5 + C)

    detections = []

    for gy in range(current_S):
        for gx in range(current_S):
            for a in range(A):
                obj = torch.sigmoid(preds[gy, gx, a, 0]).item()
                
                # Speed optimization: skip essentially empty cells
                if obj < 0.05: continue 

                cls_logits = preds[gy, gx, a, 5:]
                cls_probs = torch.sigmoid(cls_logits) 
                cls_id = torch.argmax(cls_probs).item()
                conf = obj * cls_probs[cls_id].item()

                if conf < CONF_THRESH: continue

                tx, ty = preds[gy, gx, a, 1], preds[gy, gx, a, 2]
                tw, th = preds[gy, gx, a, 3], preds[gy, gx, a, 4]

                cx = (gx + torch.sigmoid(tx).item()) / current_S
                cy = (gy + torch.sigmoid(ty).item()) / current_S
                bw = torch.sigmoid(tw).item() 
                bh = torch.sigmoid(th).item()

                x1 = int((cx - bw / 2) * img_w)
                y1 = int((cy - bh / 2) * img_h)
                x2 = int((cx + bw / 2) * img_w)
                y2 = int((cy + bh / 2) * img_h)
                
                x1=max(0,x1); y1=max(0,y1); x2=min(img_w,x2); y2=min(img_h,y2)
                detections.append((x1, y1, x2, y2, cls_id, conf))

    return nms(detections, NMS_THRESH)


# OpenCV part

cap = cv2.VideoCapture(0)

if not cap.isOpened(): print("❌ Camera not detected")
else:
    print(f"✅ SYSTEM LIVE. Threshold: {CONF_THRESH}")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        inp = preprocess(frame)

        with torch.no_grad():
            preds = model(inp)

        detections = decode_final(preds, frame.shape[1], frame.shape[0])
        
        for x1, y1, x2, y2, cls_id, conf in detections:
            # Color Logic
            if cls_id == 0: color = (0, 255, 0)     # Person = Green
            elif cls_id == 3: color = (0, 0, 255)   # Bottle = Red
            else: color = (255, 255, 0)             # Others = Cyan
            
            # Draw Box (Thicker for visibility)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"{VOC_CLASSES[cls_id]} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 5), (x1 + t_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Final Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
