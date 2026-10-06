import sys
import os
import math
import numpy as np

# CRITICAL: This is needed for np.math module compatibility
np.math = math

# Fix numpy compatibility issues (np.emath functions removed in newer numpy)
if not hasattr(np.emath, 'pi'):
    np.emath.pi = np.pi
if not hasattr(np.emath, 'atan2'):
    np.emath.atan2 = np.arctan2
if not hasattr(np.emath, 'floor'):
    np.emath.floor = np.floor
if not hasattr(np.emath, 'log'):
    np.emath.log = np.log
if not hasattr(np.emath, 'sin'):
    np.emath.sin = np.sin
if not hasattr(np.emath, 'cos'):
    np.emath.cos = np.cos
if not hasattr(np.emath, 'sqrt'):
    np.emath.sqrt = np.sqrt

# Add local face-detection-tflite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face-detection-tflite'))

# Import from local fdlite
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi

# Use reference model from local package
MODEL_PATH = 'face-detection-tflite/fdlite/data'

# Load models
detect_faces = FaceDetection(model_path=MODEL_PATH)
detect_face_landmarks = FaceLandmark(model_path=MODEL_PATH)

# Load image
from PIL import Image
img = Image.open('assets/portrait.jpg')
img_rgb = img.convert('RGB')
w, h = img.size

print(f"Image size: {w}x{h}")
print(f"Python center pixel (w//2, h//2): {img_rgb.getpixel((w//2, h//2))}")
print(f"Python top-left (0, 0): {img_rgb.getpixel((0, 0))}")
print(f"Python top-right (w-1, 0): {img_rgb.getpixel((w-1, 0))}")

# Check tensor data using fdlite
from fdlite.transform import image_to_tensor
height, width = 128, 128
image_data = image_to_tensor(
    img,
    None,
    output_size=(width, height),
    keep_aspect_ratio=True,
    output_range=(-1, 1))
print(f"Python tensor first 10 values: {image_data.tensor_data.flatten()[:10]}")
print(f"Python padding: {image_data.padding}")

# Check tensor data at specific locations (not letterbox areas)
tensor = image_data.tensor_data
print(f"Python tensor shape: {tensor.shape}")
# Sample from middle of image where actual image content is
print(f"Python tensor at (64, 32): {tensor[32, 64]}")
print(f"Python tensor at (64, 64): {tensor[64, 64]}")
print(f"Python tensor at (64, 96): {tensor[96, 64]}")

# Detect faces
face_detections = detect_faces(img)

print(f"Python: After NMS (final, letterbox removed): {[{'score': d.score, 'bbox': [d.data[0].tolist(), d.data[1].tolist()]} for d in face_detections]}")

if face_detections:
    # Get ROI
    face_roi = face_detection_to_roi(face_detections[0], img.size)
    
    # Detect landmarks
    face_landmarks = detect_face_landmarks(img, face_roi)
    
    # Log landmark results
    print(f"Landmarks: {len(face_landmarks)}")
    print("First 3 landmarks (Python):")
    for i, lm in enumerate(face_landmarks[:3]):
        print(f"  {i}: x={lm.x:.6f}, y={lm.y:.6f}, z={lm.z:.6f}")
else:
    print("No face detected")
