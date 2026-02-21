import cv2
import torch
import numpy as np
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_SIZE = 320      # Ensure this matches your training
S = 20              # 26 for 416, 20 for 320
A = 3               
C = 7   
CONF_THRESH = 0.25  # Threshold for display

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
    model.load_state_dict(state, strict=False)
    
    # Use .eval() for inference. 
    # If detections vanish, you can try .train() (Batch Norm Hack), 
    # but .eval() is correct for a well-trained model.
    model.eval() 
    model.to(device)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# -----------------------------
# UTILITIES
# -----------------------------
def decode_boxes(preds, img_w, img_h):
    preds = preds.squeeze(0) 
    current_S = preds.size(1) 
    preds = preds.permute(1, 2, 0).view(current_S, current_S, A, 5 + C)

    detections = []

    for gy in range(current_S):
        for gx in range(current_S):
            for a in range(A):
                obj = torch.sigmoid(preds[gy, gx, a, 0]).item()
                cls_probs = torch.sigmoid(preds[gy, gx, a, 5:])
                cls_id = torch.argmax(cls_probs).item()
                cls_score = cls_probs[cls_id].item()
                
                conf = obj * cls_score

                if conf > CONF_THRESH:
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
    return detections

# -----------------------------
# SMART TEST LOOP (Only Valid Images)
# -----------------------------
print("\n STARTING SMART TEST (Showing 10 Valid Images)...\n")
print("Press any key to advance to the next image.")

count = 0
MAX_TESTS = 10

# Iterate through the test_loader
for batch_idx, (images, targets) in enumerate(test_loader):
    if count >= MAX_TESTS: break
    
    # Process each image in the batch
    for i in range(len(images)):
        if count >= MAX_TESTS: break
        
        # --- THE SMART FILTER ---
        # We look at the Ground Truth (targets) for this specific image.
        # If your Dataset is working correctly, it ONLY contains 1.0s for the 7 classes.
        # Everything else (sheep, bus, car) was already filtered out or ignored.
        
        tgt_tensor = targets[i] 
        # Check if ANY object exists in this image
        has_relevant_object = (tgt_tensor[..., 0] == 1.0).any()
        
        if not has_relevant_object:
            # Skip this image silently
            continue 
            
        # --- RUN INFERENCE ---
        img_tensor = images[i].unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            preds = model(img_tensor)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        count += 1
        
        # --- VISUALIZE ---
        img_vis = images[i].permute(1, 2, 0).numpy()
        img_vis = (img_vis * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        h, w = img_vis.shape[:2]
        detections = decode_boxes(preds, w, h)
        
        print(f" Image {count}: Found {len(detections)} Predictions |  Latency: {latency_ms:.2f} ms")
        
        # Draw Boxes
        for (x1, y1, x2, y2, cid, conf) in detections:
            # Color Coding
            if cid == 3: color = (0, 0, 255)      # Bottle = Red
            elif cid == 0: color = (0, 255, 0)    # Person = Green
            else: color = (255, 255, 0)           # Others = Cyan
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{VOC_CLASSES[cid]} {conf:.2f}"
            
            # Label Background
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img_vis, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
            cv2.putText(img_vis, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
        cv2.imshow(f"Valid Test Image {count}", img_vis)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

print(f"\n✅ Finished. Displayed {count} images containing your specific objects.")

# sample
# ✅ Model loaded.

# STARTING SMART TEST (Showing 10 Valid Images)...

# Press any key to advance to the next image.
#  Image 1: Found 3 Predictions |  Latency: 189.44 ms
#  Image 2: Found 5 Predictions |  Latency: 268.91 ms
#  Image 3: Found 3 Predictions |  Latency: 222.27 ms
#  Image 4: Found 4 Predictions |  Latency: 243.35 ms
#  Image 5: Found 9 Predictions |  Latency: 256.09 ms
#  Image 6: Found 3 Predictions |  Latency: 185.83 ms
#  Image 7: Found 6 Predictions |  Latency: 262.01 ms
#  Image 8: Found 5 Predictions |  Latency: 174.04 ms
#  Image 9: Found 3 Predictions |  Latency: 267.24 ms
#  Image 10: Found 3 Predictions |  Latency: 200.03 ms

# ✅ Finished. Displayed 10 images containing your specific objects.
