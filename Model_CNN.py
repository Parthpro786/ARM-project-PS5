#----------------------------------------the main starting point----------------------
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint

class QuantConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1. Input Quantizer: Ensures input is always Int8 
        # (Crucial because we are now passing Floats between blocks)
        self.quant_inp = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            bit_width=8, 
            return_quant_tensor=True
        )
        
        # 2. Conv: Standard Brevitas Conv
        self.conv = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=3, padding=1, bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=8,
            return_quant_tensor=True 
        )
        
        # 3. BN: Standard PyTorch BN
        self.bn = nn.BatchNorm2d(out_ch)
        
        # 4. Act: Returns FLOAT (return_quant_tensor=False)
        # This allows us to use standard nn.MaxPool2d next
        self.act = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=False # <--- Key change: Output standard Float
        )

    def forward(self, x):
        # Quantize Input -> Conv -> BN -> Act(Float)
        x_q = self.quant_inp(x)
        out_q = self.conv(x_q)
        out_bn = self.bn(out_q)
        return self.act(out_bn)


class TinyYOLOv2_Brevitas(nn.Module):
    def __init__(self, num_classes=7, num_anchors=3):
        super().__init__()
        
        # Note: We replaced qnn.QuantMaxPool2d with nn.MaxPool2d
        # This works because our blocks now output standard Floats.
        
        self.backbone = nn.Sequential(
            QuantConvBlock(3, 16),
            QuantConvBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            QuantConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            QuantConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            QuantConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Detector Head
        # We need an explicit input quantizer here too, because the last MaxPool
        # outputs Floats.
        self.detector_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            bit_width=8, 
            return_quant_tensor=True
        )
        
        self.detector = qnn.QuantConv2d(
            256, num_anchors * (5 + num_classes),
            kernel_size=1,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=8,
            bias=True,
            return_quant_tensor=False # Output Float for Loss Calculation
        )

    def forward(self, x):
        # Backbone (returns float)
        features = self.backbone(x)
        
        # Detector (Quantize -> Conv -> Float)
        features_q = self.detector_quant(features)
        out = self.detector(features_q)
        
        return out
