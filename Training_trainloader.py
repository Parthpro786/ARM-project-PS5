import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. SETUP ---
print("Initializing Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Initialize the corrected model from Script 1
# Ensure the model definition script is run/imported before this
model = TinyYOLOv2_Brevitas(num_classes=7, num_anchors=3) 
model = model.to(device)

# --- 2. CONFIGURATION ---
num_epochs = 40
learning_rate = 1e-3

# Optimizer (Adam is generally faster/stable for YOLO than SGD)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Optional: Scheduler to lower LR if loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

print(f"Training images: {len(train_loader.dataset)}")
# print(f"Test images: {len(test_loader.dataset)}") # Uncomment if you want to verify test set size

# --- 3. TRAINING LOOP ---
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    
    print(f"--- Epoch {epoch+1} Started ---")
    
    for batch_i, (images, targets) in enumerate(train_loader):
        
        # Move data to GPU/CPU
        images = images.to(device)
        targets = targets.to(device)

        # --- USER DEBUG REQUEST ---
        # Summing index 0 (Objectness) counts how many anchors have an object assigned
        # If this prints '0.0' consistently, something is wrong with Script 2 (Anchors)
        # We perform the check here before the forward pass
        num_objects_in_batch = targets[..., 0].sum().item()
        
        # Forward pass
        output_y = model(images)
        
        # Calculate Loss
        loss = yolo_loss(output_y, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Optional: Print progress every 10 batches to avoid clutter
        if batch_i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_i} | Objects: {num_objects_in_batch:.0f} | Loss: {loss.item():.4f}")

    # Calculate average loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    
    # Update Learning Rate Scheduler
    scheduler.step(avg_train_loss)

    print(f">>> Epoch [{epoch+1}/{num_epochs}] Completed | Avg Loss: {avg_train_loss:.4f}")

# --- 4. SAVE MODEL ---
print("Training Finished. Saving model...")
torch.save(model.state_dict(), "tinyyolo.pth")
print("Saved as 'tinyyolo.pth'")
