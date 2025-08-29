#!/usr/bin/env python3
"""
Test the actual CNN output size for our mel spectrogram dimensions
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/Users/gjb/Projects/flow_synthesizer/code')
from models.basic import GatedCNN
import argparse

# Create test args
args = argparse.Namespace(
    kernel=5,
    dilation=3
)

# Our actual mel spectrogram dimensions
input_size = [1, 128, 173]  # [C, H, W]
output_size = 35  # PolyMAX has 35 parameters

print(f"Testing GatedCNN with input_size={input_size}, output_size={output_size}")

# Create model with debug output
model = GatedCNN(input_size, output_size, channels=32, n_layers=4, hidden_size=512, n_mlp=2, type_mod='gated', args=args)

# Create test input
test_input = torch.randn(1, 128, 173)  # batch_size=1, H=128, W=173
print(f"\nTest input shape: {test_input.shape}")

# Forward pass to see actual dimensions
with torch.no_grad():
    try:
        output = model(test_input)
        print(f"\nSuccess! Output shape: {output.shape}")
    except Exception as e:
        print(f"\nError during forward pass: {e}")
        print("This confirms the dimension mismatch issue.")