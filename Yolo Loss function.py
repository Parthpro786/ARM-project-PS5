import torch
import torch.nn.functional as F

# --- FINAL LOSS CONFIGURATION ---
# We use these weights to force the model to be confident.
LAMBDA_COORD = 5.0
LAMBDA_OBJ   = 5.0   # <--- CRITICAL: Keeps detections strong
LAMBDA_NOOBJ = 0.1   # <--- CRITICAL: Stops background from killing the signal
CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 3.0, 5.0, 5.0, 5.0, 3.0])

def yolo_loss(preds, targets):
    
    B = preds.size(0)
    device = preds.device
    class_weights = CLASS_WEIGHTS.to(device)
    
    # Auto-detect Grid Size (Works for both 320 and 416)
    current_grid_size = preds.size(2) 
    
    # 1. Reshape Input
    preds = preds.permute(0, 2, 3, 1).contiguous()
    preds = preds.view(B, current_grid_size, current_grid_size, NUM_ANCHORS, 5 + NUM_CLASSES)

    targets = targets.permute(0, 2, 3, 1, 4).contiguous()

    # 2. Extract Predictions
    pred_obj = torch.sigmoid(preds[..., 0])    
    pred_xy  = torch.sigmoid(preds[..., 1:3])  
    pred_wh  = torch.sigmoid(preds[..., 3:5])  
    pred_cls = torch.sigmoid(preds[..., 5:])   

    # 3. Extract Targets
    tgt_obj = targets[..., 0]   
    
    # Coordinate Math (Grid Offset)
    grid_x = torch.arange(current_grid_size, device=device).repeat(current_grid_size, 1).view([1, current_grid_size, current_grid_size, 1])
    grid_y = torch.arange(current_grid_size, device=device).repeat(current_grid_size, 1).t().view([1, current_grid_size, current_grid_size, 1])
    
    tgt_tx = (targets[..., 1] * current_grid_size) - grid_x
    tgt_ty = (targets[..., 2] * current_grid_size) - grid_y
    tgt_xy = torch.stack([tgt_tx, tgt_ty], dim=-1)
    
    tgt_wh = targets[..., 3:5] 
    tgt_cls = targets[..., 5:]

    # 4. Masks
    obj_mask   = tgt_obj 
    noobj_mask = 1.0 - tgt_obj 

    # 5. Calculate Losses
    # Coordinates (Only penalized if object exists)
    loss_xy = F.mse_loss(pred_xy * obj_mask.unsqueeze(-1), 
                         tgt_xy  * obj_mask.unsqueeze(-1), reduction="sum")

    loss_wh = F.mse_loss(pred_wh * obj_mask.unsqueeze(-1),
                         tgt_wh  * obj_mask.unsqueeze(-1), reduction="sum")

    # Objectness (The Signal Booster)
    loss_obj = F.mse_loss(pred_obj * obj_mask, tgt_obj, reduction="sum")
    
    # Background (The Noise Filter)
    loss_noobj = F.mse_loss(pred_obj * noobj_mask, tgt_obj * 0.0, reduction="sum")

    # Classification
    raw_cls_loss = F.mse_loss(pred_cls * obj_mask.unsqueeze(-1), tgt_cls, reduction="none")
    loss_cls = (raw_cls_loss * class_weights).sum()

    # Total Weighted Sum
    total_loss = (LAMBDA_COORD * (loss_xy + loss_wh) + 
                  LAMBDA_OBJ * loss_obj +       
                  LAMBDA_NOOBJ * loss_noobj +   
                  loss_cls)

    return total_loss / B
