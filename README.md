# hello-webnn-samples

A collection of browser demos for running neural-network models with the
[WebNN](https://www.w3.org/TR/webnn/) API, plus related face-detection and ONNX
experiments.

The demos are static HTML pages. Most run inference directly with the native
WebNN API (`navigator.ml` + `MLGraphBuilder`) — no TensorFlow.js, no ONNX
Runtime, no build step. A few use TensorFlow.js / TFLite or ONNX Runtime Web for
comparison.

## Running locally

Serve the repo root with any static file server:

```bash
python3 server.py
# then open http://localhost:8000
```

`server.py` adds CORS headers and the `application/wasm` MIME type needed by
some demos. The homepage (`index.html`) links to every demo.

## WebNN browser support

The pure-WebNN demos require a browser with WebNN enabled (Chromium-based
browsers with the WebNN flag, or a WebNN-capable build). Each demo has a
CPU / GPU / NPU device selector where supported.

## Demos

### Pure WebNN

| Demo | Model | Directory |
|------|-------|-----------|
| Image Classification | SqueezeNet 1.1 (224×224, 1000 ImageNet classes) | [`image_classification/`](image_classification/) |
| Handwritten Digit Recognition | LeNet (28×28 MNIST, draw or upload) | [`lenet/`](lenet/) |
| Selfie Segmentation | MediaPipe Selfie Segmenter (256×256, person mask + background effects) | [`selfie_segmentation/`](selfie_segmentation/) |
| Fast Style Transfer | Feed-forward style transfer (540×540, 6 styles) | [`style_transfer/`](style_transfer/) |
| Face Detection | BlazeFace SSD (128×128, bounding boxes + keypoints) | [`webnn_face_detection/`](webnn_face_detection/) |
| Face Detector (raw) | BlazeFace SSD graph built directly with `MLGraphBuilder` | [`face_detector-webnn/`](face_detector-webnn/) |
| Group Test | Subgraph decomposition of the first N operators (1–5 ops) | [`face_detector_group_test/`](face_detector_group_test/) |
| Graph Explorer | Clickable graph nodes, per-node subgraph comparison across devices | [`graph_explorer/`](graph_explorer/) |
| Face Detection v2 (SSD + YOLO) | Switchable SSD v1/v2 + YOLOv8n-Face backends + landmarks | [`webnn_face_detection_v2/`](webnn_face_detection_v2/) |
| MediaPipe Face Detector | BlazeFace graph via `MLGraphBuilder` | [`mediapipe_face_detector-webnn/`](mediapipe_face_detector-webnn/) |
| Face Landmark | 468-point landmark estimation | [`face_landmark-webnn/`](face_landmark-webnn/) |
| Face Landmarks Detector | Detection + landmark pipeline | [`face_landmarks_detector-webnn/`](face_landmarks_detector-webnn/) |
| YOLOv8n Face | YOLOv8n-face (640×640, DFL + 5 keypoints) | [`yolov8n-face-webnn/`](yolov8n-face-webnn/) |
| YOLOv8s | YOLOv8s object detection | [`yolov8s-webnn/`](yolov8s-webnn/) |
| YOLOv8s Detection | YOLOv8s detection demo | [`yolov8s_detection/`](yolov8s_detection/) |
| Virtual Try-On | Realtime virtual makeup / lip try-on | [`virtual_tryon/`](virtual_tryon/) |

### TensorFlow.js / TFLite

| Demo | Directory |
|------|-----------|
| Face Detection + Landmarks | [`face_detection/`](face_detection/) |
| Model Comparison (TFLite vs TFJS) | [`compare_models/`](compare_models/) |
| Landmarks Verification | [`verify_landmarks/`](verify_landmarks/) |
| Preprocessing Test | [`preprocess_test/`](preprocess_test/) |

### ONNX Runtime

| Demo | Directory |
|------|-----------|
| ONNX Face Landmarker (WASM / WebGL / WebGPU / WebNN) | [`onnx_web/`](onnx_web/) |
| ONNX Runtime Test (browser) | [`onnx_test/`](onnx_test/) |
| ONNX Runtime Test (Node.js) | [`onnx_test_node/`](onnx_test_node/) |
| ONNX Python Workspace (conversion / verification) | [`onnx_workspace/`](onnx_workspace/) |

### Reference / tools

| Item | Path |
|------|------|
| MediaPipe FaceLandmarker reference | [`reference/reference_draw_mesh.js`](reference/reference_draw_mesh.js) |
| Model inspectors / preprocessing helpers | `inspect_model.py`, `inspect_all_models.py`, `inspect_tflite.py`, `preprocess_test.py`, `debug_landmarks.py` |

## Model sources

- **Face detection / landmarks**: MediaPipe BlazeFace SSD + FaceMesh
  (originally TFLite, converted to TFJS / ONNX / WGWT binary weights).
- **SqueezeNet 1.1**: [ONNX Model Zoo](https://github.com/onnx/models),
  weights exported to a single `WGWT` binary + manifest.
- **LeNet**: weights from
  [webnn-samples](https://github.com/webmachinelearning/webnn-samples) (`lenet.bin`).
- **Selfie Segmenter / Fast Style Transfer**: MediaPipe-style weights served by
  webnn-samples, exported to per-demo weight binaries.

## Weight format

The pure-WebNN demos load two kinds of weight files:

- **WGWT** (magic bytes `WGWT`) — a flat float32 binary plus a JSON manifest
  mapping tensor names to `{byteOffset, byteLength, shape}`. Used by
  `image_classification/`, `selfie_segmentation/`, `style_transfer/`, and
  `face_detector-webnn/`.
- **Raw binary** — a single `.bin` with fixed byte offsets (no manifest). Used
  by `lenet/` (and `selfie_segmentation/` uses a similar split weights/biases
  `.bin` + `.json` layout).
