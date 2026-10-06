import sys
import os
import math
import numpy as np

# Fix numpy compatibility issues
np.math = math
if not hasattr(np.emath, 'pi'):
    np.emath.pi = np.pi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face-detection-tflite'))

from PIL import Image
from fdlite.transform import image_to_tensor

# Load image
img = Image.open('assets/portrait.jpg')
img_rgb = img.convert('RGB')
w, h = img.size

print(f"Image: {w}x{h}")

# Python preprocessing
height, width = 128, 128
image_data = image_to_tensor(
    img,
    None,
    output_size=(width, height),
    keep_aspect_ratio=True,
    output_range=(-1, 1))

tensor = image_data.tensor_data
padding = image_data.padding

print(f"Padding: {padding}")

def get_pixel(row, col):
    return tensor[row, col].tolist()

# Check letterbox areas
print(f"Top-left (0,0): {get_pixel(0, 0)}")
print(f"Letterbox area (10, 64): {get_pixel(10, 64)}")
print(f"Letterbox area (20, 64): {get_pixel(20, 64)}")

# Check actual image content areas
print(f"Image area (50, 64): {get_pixel(50, 64)}")
print(f"Image area (64, 64): {get_pixel(64, 64)}")
print(f"Image area (90, 64): {get_pixel(90, 64)}")

# Compare anchoring (if needed)
# Generate anchors in Python
def generate_anchors(opts):
    anchors = []
    num_layers = opts['num_layers']
    strides = opts['strides']
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']

    layer_id = 0
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return anchors

SSD_OPTIONS_SHORT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}

anchors = generate_anchors(SSD_OPTIONS_SHORT)
print(f"Number of anchors: {len(anchors)}")
print(f"First 5 anchors: {anchors[:5]}")