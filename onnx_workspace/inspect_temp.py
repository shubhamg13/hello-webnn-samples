import tensorflow as tf
import os

base = r'D:\Developments\face_landmarker_tflite\assets'
ref = r'D:\Developments\face_landmarker_tflite\face-detection-tflite\fdlite\data'

models = [
    (os.path.join(base, 'face_detector.tflite'), 'face_detector.tflite'),
    (os.path.join(base, 'face_detector_ref.tflite'), 'face_detector_ref.tflite'),
    (os.path.join(base, 'face_detection_ref.tflite'), 'face_detection_ref.tflite'),
    (os.path.join(base, 'face_landmarks_detector.tflite'), 'face_landmarks_detector.tflite'),
    (os.path.join(base, 'face_landmark_ref.tflite'), 'face_landmark_ref.tflite'),
    (os.path.join(ref, 'face_detection_front.tflite'), 'face_detection_front.tflite'),
    (os.path.join(ref, 'face_detection_back.tflite'), 'face_detection_back.tflite'),
    (os.path.join(ref, 'face_detection_short_range.tflite'), 'face_detection_short_range.tflite'),
    (os.path.join(ref, 'face_detection_full_range.tflite'), 'face_detection_full_range.tflite'),
    (os.path.join(ref, 'face_detection_full_range_sparse.tflite'), 'face_detection_full_range_sparse.tflite'),
    (os.path.join(ref, 'face_landmark.tflite'), 'face_landmark.tflite'),
    (os.path.join(ref, 'iris_landmark.tflite'), 'iris_landmark.tflite'),
]

for path, name in models:
    if not os.path.exists(path):
        print(f'=== {name} (NOT FOUND) ===')
        continue
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()
        sz = os.path.getsize(path)
        print(f'=== {name} ({sz} bytes) ===')
        print(f'  Inputs: {len(inp)}')
        for i, d in enumerate(inp):
            print(f"    [{i}] name={d['name']}, shape={list(d['shape'])}, dtype={d['dtype'].__name__}")
        print(f'  Outputs: {len(out)}')
        for i, d in enumerate(out):
            print(f"    [{i}] name={d['name']}, shape={list(d['shape'])}, dtype={d['dtype'].__name__}")
    except Exception as e:
        print(f'=== {name} (ERROR: {e}) ===')
