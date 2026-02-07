#!/usr/bin/env python3
"""
EfficientNet-B5 Architecture Analysis for Grad-CAM++ Layer Selection
=====================================================================
Analyzes the full architecture of EfficientNet-B5 (timm) to identify
optimal layers for Grad-CAM++ in diabetic retinopathy classification.

Focus: Finding layers that balance spatial resolution (for small lesions
like microaneurysms in Mild DR) with semantic richness.
"""

import torch
import torch.nn as nn
import timm
import sys
from collections import OrderedDict

# ============================================================
# 1. Load Model
# ============================================================
print("=" * 90)
print("EFFICIENTNET-B5 ARCHITECTURE ANALYSIS FOR GRAD-CAM++")
print("=" * 90)

model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=5)
model.eval()

# ============================================================
# 2. Print Full Architecture (high-level)
# ============================================================
print("\n" + "=" * 90)
print("SECTION 1: HIGH-LEVEL MODEL STRUCTURE")
print("=" * 90)

for name, module in model.named_children():
    print(f"\n  {name}: {module.__class__.__name__}")
    if hasattr(module, '__len__'):
        try:
            print(f"    (contains {len(module)} sub-modules)")
        except:
            pass

# ============================================================
# 3. Forward Pass with Hooks to Capture Shapes
# ============================================================
print("\n" + "=" * 90)
print("SECTION 2: FORWARD PASS - FEATURE MAP SHAPES (Input: 1x3x456x456)")
print("=" * 90)

# Register hooks on all named modules to capture output shapes
feature_maps = OrderedDict()

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            feature_maps[name] = {
                'shape': tuple(output.shape),
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters()),
            }
    return hook_fn

hooks = []
for name, module in model.named_modules():
    h = module.register_forward_hook(make_hook(name))
    hooks.append(h)

# Forward pass with dummy input (456x456 as per project config)
dummy_input = torch.randn(1, 3, 456, 456)
with torch.no_grad():
    output = model(dummy_input)

# Remove hooks
for h in hooks:
    h.remove()

print(f"\nOutput shape: {output.shape}")
print(f"Total named modules: {len(feature_maps)}")

# ============================================================
# 4. Show All Conv/BN Layers with Feature Map Sizes
# ============================================================
print("\n" + "=" * 90)
print("SECTION 3: ALL CONVOLUTIONAL LAYERS (Conv2d, BatchNorm2d)")
print("=" * 90)
print(f"\n{'Name':<60} {'Type':<20} {'Output Shape':<25} {'Params':>10}")
print("-" * 115)

for name, info in feature_maps.items():
    if info['type'] in ['Conv2d', 'BatchNorm2d', 'SiLU', 'SqueezeExcite',
                         'DepthwiseSeparableConv', 'InvertedResidual',
                         'Sequential', 'Conv2dSame']:
        shape_str = f"{info['shape']}" if len(info['shape']) == 4 else str(info['shape'])
        print(f"  {name:<58} {info['type']:<20} {shape_str:<25} {info['params']:>10,}")

# ============================================================
# 5. Detailed Block-by-Block Analysis
# ============================================================
print("\n" + "=" * 90)
print("SECTION 4: BLOCK-BY-BLOCK ANALYSIS (Backbone)")
print("=" * 90)

# Identify all top-level blocks
block_info = []
for name, info in feature_maps.items():
    # Match block outputs (blocks.X)
    parts = name.split('.')
    if len(parts) == 2 and parts[0] == 'blocks' and parts[1].isdigit():
        block_idx = int(parts[1])
        H, W = info['shape'][2], info['shape'][3]
        C = info['shape'][1]
        block_info.append({
            'name': name,
            'block_idx': block_idx,
            'channels': C,
            'spatial': (H, W),
            'type': info['type'],
            'total_shape': info['shape'],
        })

print(f"\n{'Block':<20} {'Output Shape':<25} {'Channels':>10} {'Spatial':>12} {'Resolution':>12}")
print("-" * 85)

for b in block_info:
    res = f"{b['spatial'][0]}x{b['spatial'][1]}"
    shape_str = f"{b['total_shape']}"
    print(f"  {b['name']:<18} {shape_str:<25} {b['channels']:>10} {res:>12} {b['spatial'][0]/456*100:>10.1f}%")

# ============================================================
# 6. Identify ALL Sub-blocks within Each Block
# ============================================================
print("\n" + "=" * 90)
print("SECTION 5: SUB-BLOCK DETAIL (blocks.X.Y)")
print("=" * 90)

for block_idx in range(7):  # blocks.0 through blocks.6
    print(f"\n  --- blocks.{block_idx} ---")
    print(f"  {'Name':<55} {'Type':<30} {'Output Shape':<25}")
    print(f"  " + "-" * 110)

    for name, info in feature_maps.items():
        if name.startswith(f'blocks.{block_idx}.') and len(info['shape']) == 4:
            parts = name.split('.')
            depth = len(parts)
            if depth <= 3:  # blocks.X.Y level
                shape_str = f"{info['shape']}"
                indent = "  " * (depth - 1)
                print(f"  {indent}{name:<53} {info['type']:<30} {shape_str:<25}")

# ============================================================
# 7. conv_stem, bn1, conv_head, bn2 Analysis
# ============================================================
print("\n" + "=" * 90)
print("SECTION 6: STEM AND HEAD LAYERS")
print("=" * 90)

stem_head_layers = ['conv_stem', 'bn1', 'conv_head', 'bn2', 'global_pool', 'classifier']
for layer_name in stem_head_layers:
    if layer_name in feature_maps:
        info = feature_maps[layer_name]
        print(f"\n  {layer_name}:")
        print(f"    Type: {info['type']}")
        print(f"    Output Shape: {info['shape']}")
        print(f"    Parameters: {info['params']:,}")

# ============================================================
# 8. GRAD-CAM CANDIDATE LAYERS (Comprehensive)
# ============================================================
print("\n" + "=" * 90)
print("SECTION 7: GRAD-CAM++ CANDIDATE LAYERS")
print("=" * 90)

candidates = []

# A) Each block output (blocks.0 through blocks.6)
for b in block_info:
    position = "early" if b['block_idx'] <= 1 else ("mid" if b['block_idx'] <= 4 else "late")
    candidates.append({
        'name': b['name'],
        'channels': b['channels'],
        'spatial': b['spatial'],
        'position': position,
        'description': f"Block {b['block_idx']} output",
    })

# B) conv_head (final conv before classifier)
if 'conv_head' in feature_maps:
    info = feature_maps['conv_head']
    candidates.append({
        'name': 'conv_head',
        'channels': info['shape'][1],
        'spatial': (info['shape'][2], info['shape'][3]),
        'position': 'final',
        'description': 'Final 1x1 conv (channel expansion)',
    })

# C) bn2 (final batch norm)
if 'bn2' in feature_maps:
    info = feature_maps['bn2']
    candidates.append({
        'name': 'bn2',
        'channels': info['shape'][1],
        'spatial': (info['shape'][2], info['shape'][3]),
        'position': 'final',
        'description': 'Final batch norm after conv_head',
    })

# D) Last sub-block in blocks.6
for name, info in feature_maps.items():
    if name.startswith('blocks.6.') and len(info['shape']) == 4:
        parts = name.split('.')
        if len(parts) == 3 and parts[2].isdigit():
            candidates.append({
                'name': name,
                'channels': info['shape'][1],
                'spatial': (info['shape'][2], info['shape'][3]),
                'position': 'late',
                'description': f"Sub-block {parts[2]} in blocks.6",
            })

# E) Last sub-block in blocks.3, 4, 5 (mid-late, good for small lesions)
for blk in [3, 4, 5]:
    for name, info in feature_maps.items():
        if name.startswith(f'blocks.{blk}.') and len(info['shape']) == 4:
            parts = name.split('.')
            if len(parts) == 3 and parts[2].isdigit():
                position = "mid" if blk <= 4 else "late"
                candidates.append({
                    'name': name,
                    'channels': info['shape'][1],
                    'spatial': (info['shape'][2], info['shape'][3]),
                    'position': position,
                    'description': f"Sub-block {parts[2]} in blocks.{blk}",
                })

print(f"\n{'#':<4} {'Layer Name':<25} {'Channels':>10} {'Spatial':>12} {'Position':<10} {'Description':<40}")
print("-" * 105)

for i, c in enumerate(candidates):
    res = f"{c['spatial'][0]}x{c['spatial'][1]}"
    print(f"  {i+1:<2} {c['name']:<25} {c['channels']:>10} {res:>12} {c['position']:<10} {c['description']:<40}")

# ============================================================
# 9. RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 90)
print("SECTION 8: RECOMMENDATIONS FOR GRAD-CAM++ IN DR CLASSIFICATION")
print("=" * 90)

print("""
ANALYSIS SUMMARY:
================

EfficientNet-B5 with input 456x456 has the following block structure:

  INPUT (456x456) 
    -> conv_stem (3->48 channels, stride 2)
    -> blocks.0 (MBConv)  => early features: edges, textures
    -> blocks.1 (MBConv)  => early features: simple patterns
    -> blocks.2 (MBConv)  => mid features: complex textures
    -> blocks.3 (MBConv)  => mid features: object parts
    -> blocks.4 (MBConv)  => mid-late: semantic parts
    -> blocks.5 (MBConv)  => late: high-level semantics
    -> blocks.6 (MBConv)  => late: most abstract features
    -> conv_head (1x1 conv expansion) => final feature representation
    -> bn2 => normalized final features
    -> global_pool => 1x1
    -> classifier => 5 classes

RECOMMENDATIONS FOR GRAD-CAM++ LAYERS:
======================================

1. PRIMARY RECOMMENDATION: 'bn2' (or equivalently 'conv_head')
   - Rationale: This is the STANDARD choice for Grad-CAM on EfficientNet.
     It provides the richest semantic features with reasonable spatial resolution.
   - Best for: Overall classification explanation, moderate/severe DR

2. SECONDARY RECOMMENDATION: 'blocks.4' (last sub-block)
   - Rationale: Higher spatial resolution than bn2. Better for localizing
     SMALL lesions like microaneurysms in Mild DR.
   - Best for: Mild DR where microaneurysms are very small (< 5 pixels)
   - Reference: For fine-grained localization, mid-level features preserve
     spatial detail while still having semantic meaning.

3. TERTIARY RECOMMENDATION: 'blocks.2' (last sub-block)
   - Rationale: Much higher resolution than bn2. Captures low-level features
     like individual microaneurysms and small hemorrhages.
   - Caveat: Features are less semantically meaningful; heatmaps may be noisy.
   - Best for: Layer comparison studies showing how the network "sees" at
     different depths.

4. MULTI-LAYER STRATEGY (RECOMMENDED FOR THIS PROJECT):
   Run Grad-CAM++ on MULTIPLE layers and compare:
   
   a) 'blocks.2'  - Fine spatial detail, early features
   b) 'blocks.4'  - Good balance resolution/semantics
   c) 'bn2'       - Maximum semantic richness
   
   This multi-scale approach is particularly valuable for DR because:
   - Mild DR: Small microaneurysms need fine resolution (blocks.2 or blocks.4)
   - Severe/Proliferative DR: Large lesions are well-captured by bn2
   - The layer comparison itself is a valuable visualization for the thesis

LITERATURE SUPPORT:
==================
- Selvaraju et al. (2017): "We found that the last convolutional layer
  provides the best compromise between high-level semantics and detailed
  spatial information" -> bn2/conv_head

- For fine-grained medical imaging: Mid-level layers (blocks.3-4) often
  outperform the final layer when lesions are small (similar to findings
  in dermoscopy and chest X-ray literature)

- The existing notebook 05_gradcam.ipynb likely uses the default last layer.
  Adding blocks.4 for Mild DR analysis would strengthen the thesis.

NOTE ON LAYER NAMING IN TIMM:
=============================
When using pytorch_grad_cam or similar libraries with timm EfficientNet:
  - model.bn2        -> Final batch norm (most common target)
  - model.blocks[6]  -> Last backbone block 
  - model.blocks[4]  -> Mid-late block (recommended for small lesions)
  - model.conv_head  -> Final 1x1 conv before pooling
""")

# ============================================================
# 10. Print exact module objects for programmatic access
# ============================================================
print("\n" + "=" * 90)
print("SECTION 9: PROGRAMMATIC ACCESS (for pytorch_grad_cam)")
print("=" * 90)

print("""
# How to specify target layers for pytorch_grad_cam:

import timm
from pytorch_grad_cam import GradCAMPlusPlus

model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=5)

# Option 1: Final batch norm (standard, high semantics)
target_layer_bn2 = [model.bn2]

# Option 2: Last backbone block (high semantics)
target_layer_blocks6 = [model.blocks[6]]

# Option 3: Mid-late block (better for small lesions)
target_layer_blocks4 = [model.blocks[4]]

# Option 4: Mid block (fine detail)
target_layer_blocks2 = [model.blocks[2]]

# Option 5: Multi-layer (combine for richer heatmaps)
target_layers_multi = [model.blocks[4], model.bn2]

# Create Grad-CAM++
cam = GradCAMPlusPlus(model=model, target_layers=target_layer_bn2)
""")

# ============================================================
# 11. Parameter count per block
# ============================================================
print("\n" + "=" * 90)
print("SECTION 10: PARAMETER COUNT PER BLOCK")
print("=" * 90)

print(f"\n{'Component':<30} {'Parameters':>15} {'% of Total':>12}")
print("-" * 60)

total_params = sum(p.numel() for p in model.parameters())

for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    pct = params / total_params * 100
    print(f"  {name:<28} {params:>15,} {pct:>11.2f}%")

    # If it's blocks, break down further
    if name == 'blocks':
        for sub_name, sub_module in module.named_children():
            sub_params = sum(p.numel() for p in sub_module.parameters())
            sub_pct = sub_params / total_params * 100
            print(f"    blocks.{sub_name:<22} {sub_params:>15,} {sub_pct:>11.2f}%")

print(f"\n  {'TOTAL':<28} {total_params:>15,} {'100.00%':>12}")

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
