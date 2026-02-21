# to know whether the model has the right dataset to learn we first need to check the whether the dataset code is segregating the required objects ( according to VOC CLASS array) or not
#run this code prior to training
import matplotlib.pyplot as plt  # for plotting rectangle simply
import torch

# 1. Grab one batch from your train loader
try:
    images, targets = next(iter(train_loader))
    print(f"✅ Successfully grabbed a batch of {len(images)} images.")
except NameError:
    print("❌ Error: 'train_loader' is not defined. Run the AutoDiscoveryDataset code first!")
    exit()
except StopIteration:
    print("❌ Error: DataLoader is empty. Check your image paths!")
    exit()

# 2. Pick the first image in the batch
img_tensor = images[0]
tgt_tensor = targets[0]                       # Shape: [Anchors, Grid, Grid, 5+C]

# 3. Prepare Image for Display
# (Permute: Channels First -> Channels Last)
img_display = img_tensor.permute(1, 2, 0).numpy()

# 4. Set up the Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img_display)

# 5. Decode the Target Tensor manually
# We loop through every grid cell and anchor to find the "1.0" flag
S = tgt_tensor.shape[1] # Grid Size (here  20)
found_objects = 0

print(f" Scanning {S}x{S} grid for labels...")

for a in range(NUM_ANCHORS):
    for y in range(S):
        for x in range(S):
            # Check the "Object Confidence" index (Index 0)
            if tgt_tensor[a, y, x, 0] == 1.0: 
                found_objects += 1
                
                # Extract coordinates (They are normalized relative to the image)
                # Format in Tensor: [x_center, y_center, width, height]
                box = tgt_tensor[a, y, x, 1:5] 
                gx, gy, gw, gh = box.tolist()
                
                # Convert normalized coords (0-1) to Pixel Coords
                img_h, img_w = IMG_SIZE, IMG_SIZE
                
                # Math: Center -> Top-Left Corner
                x1 = int((gx - gw/2) * img_w)
                y1 = int((gy - gh/2) * img_h)
                w_px = int(gw * img_w)
                h_px = int(gh * img_h)
                
                # Find the Class ID (Index 5 onwards)
                cls_vec = tgt_tensor[a, y, x, 5:]
                cls_id = torch.argmax(cls_vec).item()
                
                # Draw Red Box (Ground Truth)
                rect = plt.Rectangle((x1, y1), w_px, h_px, 
                                     linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add Label Text
                label_text = f"Class {cls_id}"
                ax.text(x1, y1 - 5, label_text, color='white', 
                        backgroundcolor='red', fontsize=10, weight='bold')

plt.title(f"Ground Truth Check | Found: {found_objects} Objects", fontsize=14)
plt.axis('off')
plt.show()

# --- FINAL DIAGNOSIS ---
if found_objects == 0:
    print("\n❌ CRITICAL WARNING: No objects found in this image.")
    print("Possibilities:")
    print("1. This specific image happens to be empty (Try running the cell again).")
    print("2. The Dataloader is reading files but failing to parse the text lines.")
else:
    print(f"\n✅ SUCCESS! Found {found_objects} objects.")
    print("The Red Boxes show exactly what the model will try to learn.")

