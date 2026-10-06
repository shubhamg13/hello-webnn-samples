import os
import sys
import tf2onnx

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'face-detection-tflite', 'fdlite', 'data')
assets_dir = os.path.join(project_root, 'assets')
model_dir = os.path.dirname(__file__)

# Input models
face_detect_input = os.path.join(src_dir, 'face_detection_front.tflite')
face_landmark_input = os.path.join(src_dir, 'face_landmark.tflite')

# Output models  
face_detect_output = os.path.join(assets_dir, 'face_detector.onnx')
face_landmark_output = os.path.join(assets_dir, 'face_landmark.onnx')

print(f"Source directory: {src_dir}")
print(f"Assets directory: {assets_dir}")
print(f"Face detection input: {face_detect_input}, exists: {os.path.exists(face_detect_input)}")
print(f"Face landmark input: {face_landmark_input}, exists: {os.path.exists(face_landmark_input)}")

# Convert face detection model
if os.path.exists(face_detect_input):
    print(f"\nConverting face detection model...")
    try:
        tf2onnx.convert.from_tflite(
            face_detect_input,
            output_path=face_detect_output,
            opset=16
        )
        print(f"SUCCESS: Converted face detection to {face_detect_output}")
    except Exception as e:
        print(f"ERROR converting face detection: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"ERROR: Face detection input not found: {face_detect_input}")

# Convert face landmark model  
if os.path.exists(face_landmark_input):
    print(f"\nConverting face landmark model...")
    try:
        tf2onnx.convert.from_tflite(
            face_landmark_input,
            output_path=face_landmark_output,
            opset=16
        )
        print(f"SUCCESS: Converted face landmark to {face_landmark_output}")
    except Exception as e:
        print(f"ERROR converting face landmark: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"ERROR: Face landmark input not found: {face_landmark_input}")

# Verify outputs
print(f"\n=== Verification ===")
for output_path in [face_detect_output, face_landmark_output]:
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"EXISTS: {output_path} ({size:,} bytes)")
    else:
        print(f"MISSING: {output_path}")