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

if __name__ == "__main__":
    base_path = "assets"
    ref_path = "face-detection-tflite/fdlite/data"
    
    print("Face Landmark TFLite Models")
    print("=" * 50)
    
    # Inspect face detector model
    detect_model = os.path.join(base_path, "face_detector.tflite")
    if os.path.exists(detect_model):
        inspect_tflite_model(detect_model)
    
    # Inspect face landmarks detector model
    landmark_model = os.path.join(base_path, "face_landmarks_detector.tflite")
    if os.path.exists(landmark_model):
        inspect_tflite_model(landmark_model)
    
    # Inspect reference face landmark model
    print("\n" + "=" * 50)
    print("REFERENCE MODELS")
    print("=" * 50)
    ref_landmark_model = os.path.join(ref_path, "face_landmark.tflite")
    if os.path.exists(ref_landmark_model):
        inspect_tflite_model(ref_landmark_model)
    
    # Inspect reference iris landmark model
    iris_landmark_model = os.path.join(ref_path, "iris_landmark.tflite")
    if os.path.exists(iris_landmark_model):
        inspect_tflite_model(iris_landmark_model)
