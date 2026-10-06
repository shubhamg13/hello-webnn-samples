import flatbuffers
import struct

def read_tflite_model(model_path):
    with open(model_path, 'rb') as f:
        data = f.read()
    
    print(f"\n=== Inspecting {model_path} ===")
    print(f"File size: {len(data)} bytes")
    
    # Check TFLite magic number
    magic = data[:8]
    if magic == b'TFL3\x00\x00\x00\x00':
        print("Magic number: TFL3 (valid)")
    else:
        print(f"Magic number: {magic}")
    
    # Try to find subgraph info
    # TFLite format uses flatbuffers - look for common patterns
    data_str = data[:2000].decode('latin-1')
    
    # Look for "input" strings
    if 'input' in data_str.lower():
        print("\nFound 'input' references in header")
    
    if 'output' in data_str.lower():
        print("Found 'output' references in header")

if __name__ == "__main__":
    import os
    
    base_path = "assets"
    
    # Inspect face detector model
    detect_model = os.path.join(base_path, "face_detector.tflite")
    if os.path.exists(detect_model):
        read_tflite_model(detect_model)
    
    # Inspect face landmarks detector model
    landmark_model = os.path.join(base_path, "face_landmarks_detector.tflite")
    if os.path.exists(landmark_model):
        read_tflite_model(landmark_model)
