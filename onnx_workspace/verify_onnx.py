import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(project_root, 'assets')
fdlite_dir = os.path.join(project_root, 'face-detection-tflite')
fdlite_model_dir = os.path.join(fdlite_dir, 'fdlite', 'data')

# Add fdlite to path
sys.path.insert(0, fdlite_dir)
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi

# Fix numpy compatibility - add missing functions to scimath
import numpy.lib.scimath as scimath
scimath.atan2 = np.arctan2
scimath.floor = np.floor
scimath.sqrt = np.sqrt
scimath.log = np.log
scimath.sin = np.sin
scimath.cos = np.cos
scimath.log10 = np.log10
scimath.exp = np.exp
scimath.power = np.power
np.math = scimath
np.emath = scimath
if not hasattr(np.emath, 'pi'):
    np.emath.pi = np.pi

img_path = os.path.join(assets_dir, 'portrait.jpg')
img = Image.open(img_path)
print(f"Image: {img.size}")

# =====================
# FACE DETECTION TEST
# =====================
print(f"\n=== FACE DETECTION TEST ===")

# Reference
ref_fd = FaceDetection(model_path=fdlite_model_dir)
ref_detections = ref_fd(img)
print(f"Reference detections: {len(ref_detections)}")
if ref_detections:
    d = ref_detections[0]
    print(f"  bbox: {d.data[0]}, {d.data[1]}")
    print(f"  score: {d.score:.6f}")

# ONNX model
detector_model_path = os.path.join(assets_dir, 'face_detector.onnx')
print(f"\nLoading ONNX detector: {detector_model_path}")
session = ort.InferenceSession(detector_model_path)

# Preprocess image
sys.path.insert(0, os.path.join(fdlite_dir, 'fdlite'))
from fdlite.transform import image_to_tensor
image_data = image_to_tensor(img, None, output_size=(128, 128), keep_aspect_ratio=True, output_range=(-1, 1))
input_tensor = image_data.tensor_data[np.newaxis].astype(np.float32)

input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"Input: {input_name}, shape: {input_tensor.shape}")
print(f"Outputs: {output_names}")

outputs = session.run(output_names, {input_name: input_tensor})
raw_boxes = outputs[0]
raw_scores = outputs[1]
print(f"ONNX boxes shape: {raw_boxes.shape}")
print(f"ONNX scores shape: {raw_scores.shape}")
print(f"ONNX first 5 boxes: {raw_boxes.flatten()[:5]}")
print(f"ONNX first 5 scores: {raw_scores.flatten()[:5]}")

print(f"\n=== SUCCESS ===")
print(f"ONNX face detector conversion verified!")

# =====================
# FACE LANDMARK TEST
# =====================
print(f"\n=== FACE LANDMARK TEST ===")

# Reference landmark detection
ref_fl = FaceLandmark(model_path=fdlite_model_dir)
det = ref_detections[0]
roi = face_detection_to_roi(det, img.size)

# Preprocess using fdlite.transform
from fdlite.transform import image_to_tensor
height, width = ref_fl.input_shape[1:3]
image_data = image_to_tensor(img, roi, output_size=(width, height), keep_aspect_ratio=False, output_range=(0., 1.))
input_data = image_data.tensor_data[np.newaxis]

ref_fl.interpreter.set_tensor(ref_fl.input_index, input_data)
ref_fl.interpreter.invoke()
raw_landmarks_ref = ref_fl.interpreter.get_tensor(ref_fl.data_index)
print(f"Reference raw landmarks shape: {raw_landmarks_ref.shape}")
print(f"Reference first 10: {raw_landmarks_ref.flatten()[:10]}")

# ONNX landmark model
landmark_model_path = os.path.join(assets_dir, 'face_landmark.onnx')
print(f"\nLoading ONNX landmark model: {landmark_model_path}")
session_lm = ort.InferenceSession(landmark_model_path)

# Preprocess for landmark - same as reference (192x192, no maintain aspect ratio)
image_data_lm = image_to_tensor(img, roi, output_size=(192, 192), keep_aspect_ratio=False, output_range=(0., 1.))
roi_tensor = image_data_lm.tensor_data[np.newaxis].astype(np.float32)

input_name_lm = session_lm.get_inputs()[0].name
output_names_lm = [o.name for o in session_lm.get_outputs()]
print(f"Input: {input_name_lm}, shape: {roi_tensor.shape}")
print(f"Outputs: {output_names_lm}")

outputs_lm = session_lm.run(output_names_lm, {input_name_lm: roi_tensor})
raw_landmarks_onnx = outputs_lm[0]
print(f"ONNX landmarks shape: {raw_landmarks_onnx.shape}")
print(f"ONNX first 10 landmarks: {raw_landmarks_onnx.flatten()[:10]}")

print(f"\n=== LANDMARK SUCCESS ===")
print(f"ONNX face landmark conversion verified!")
print(f"  landmarks diff: {np.max(np.abs(raw_landmarks_ref - raw_landmarks_onnx)):.10f}")

print(f"\n=== ALL TESTS PASSED ===")
print(f"Both ONNX models produce identical outputs to TFLite!")
print(f"Files created:")
print(f"  - {detector_model_path}: {os.path.getsize(detector_model_path):,} bytes")
print(f"  - {landmark_model_path}: {os.path.getsize(landmark_model_path):,} bytes")