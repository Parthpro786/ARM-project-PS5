# 1. Switch to Eval Mode (Crucial for Quantization)
model.eval()

# 2. (Optional but good) Run one forward pass with dummy data
# This forces Brevitas to calculate final scale factors if they were pending
dummy_input = torch.randn(1, 3, 320, 320).to(device)
with torch.no_grad():
    model(dummy_input)

# 3. Now Save
torch.save(model.state_dict(), "trained_model.pth")
print("✅ Model saved correctly in Eval mode.")

import torch
from brevitas.export import export_qonnx

model = TinyYOLOv2_Brevitas(num_classes=7, num_anchors=3)
model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
model.eval()
model.cpu()

dummy_inp = torch.randn(1, 3, 320, 320)

onnx_name = "tinyyolo_export_qonnx_2.onnx"

with torch.no_grad():
    export_qonnx(model, dummy_inp, onnx_name)

print("✅ Exported:", onnx_name)
