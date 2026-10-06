import tensorflow as tf
import os

def inspect_tflite_model(model_path):
    print(f"\n=== Inspecting {os.path.basename(model_path)} ===")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nInput count: {len(input_details)}")
    for i, inp in enumerate(input_details):
        print(f"  Input {i}:")
        print(f"    name: {inp['name']}")
        print(f"    shape: {inp['shape']}")
        print(f"    dtype: {inp['dtype']}")
    
    print(f"\nOutput count: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"  Output {i}:")
        print(f"    name: {out['name']}")
        print(f"    shape: {out['shape']}")
        print(f"    dtype: {out['dtype']}")

def inspect_tfjs_model(json_path):
    import json
    
    print(f"\n=== Inspecting {os.path.basename(json_path)} ===")
    
    with open(json_path, 'r') as f:
        model_json = json.load(f)
    
    # Try different JSON structures
    user_metadata = model_json.get('userDefinedMetadata', {})
    signature = user_metadata.get('signature', model_json.get('signature', {}))
    
    inputs = signature.get('inputs', {})
    outputs = signature.get('outputs', {})
    
    if not inputs:
        # Try parsing from modelTopology
        print("  Using signature from modelTopology")
        signature = model_json.get('signature', {})
        inputs = signature.get('inputs', {})
        outputs = signature.get('outputs', {})
    
    print(f"\nInput:")
    for name, info in inputs.items():
        shape = info.get('tensorShape', {}).get('dim', [])
        shape_str = '[' + ', '.join([str(d.get('size', '?')) for d in shape]) + ']'
        print(f"  {name}: {info.get('dtype', '?')} {shape_str}")
    
    print(f"\nOutput:")
    for name, info in outputs.items():
        shape = info.get('tensorShape', {}).get('dim', [])
        shape_str = '[' + ', '.join([str(d.get('size', '?')) for d in shape]) + ']'
        print(f"  {name}: {info.get('dtype', '?')} {shape_str}")

if __name__ == "__main__":
    base_path = "assets"
    ref_path = "face-detection-tflite/fdlite/data"
    
    print("=" * 60)
    print("FACE DETECTION MODELS")
    print("=" * 60)
    
    # Inspect face detector TFLite
    detect_model = os.path.join(base_path, "face_detector.tflite")
    if os.path.exists(detect_model):
        inspect_tflite_model(detect_model)
    
    # Inspect face detector TFJS
    detect_tfjs = os.path.join(base_path, "face-detection-tfjs-short-v1/model.json")
    if os.path.exists(detect_tfjs):
        inspect_tfjs_model(detect_tfjs)
    
    print("\n" + "=" * 60)
    print("FACE LANDMARK MODELS")
    print("=" * 60)
    
    # Inspect face landmarks detector TFLite
    landmark_model = os.path.join(base_path, "face_landmarks_detector.tflite")
    if os.path.exists(landmark_model):
        inspect_tflite_model(landmark_model)
    
    # Inspect face landmarks TFJS (facemesh)
    landmark_tfjs = os.path.join(base_path, "facemesh-tfjs-default-v1/model.json")
    if os.path.exists(landmark_tfjs):
        inspect_tfjs_model(landmark_tfjs)
    
    print("\n" + "=" * 60)
    print("REFERENCE MODELS (face-detection-tflite)")
    print("=" * 60)
    
    # Reference face detection
    ref_detect = os.path.join(ref_path, "face_detection_short_range.tflite")
    if os.path.exists(ref_detect):
        inspect_tflite_model(ref_detect)
    
    # Reference face landmark
    ref_landmark = os.path.join(ref_path, "face_landmark.tflite")
    if os.path.exists(ref_landmark):
        inspect_tflite_model(ref_landmark)
