# MediaPipe Face Landmarker — ROI Expansion Logic (Detailed Analysis)

## Overview

The Face Landmarker pipeline uses a **larger crop region** than the face detector's raw bounding box. This is by design — the BlazeFace detector outputs a tight box around the eyes/nose/mouth region, but the mesh model needs to see the full face (forehead, chin, sides). The ROI is intentionally expanded by **1.5×** and made **square** before being cropped and passed into the landmark model.

This document analyzes the full pipeline from source code, covering:

1. [Model Inputs/Outputs](#1-model-inputs-and-outputs)
2. [The Two ROI Generation Paths](#2-the-two-roi-generation-paths)
3. [Step-by-Step ROI Transformation](#3-step-by-step-roi-transformation)
4. [Source Code References](#4-source-code-references)
5. [Scale Factor Decision](#5-scale-factor-decision)
6. [Visual Summary](#6-visual-summary)

---

## 1. Model Inputs and Outputs

### Model 1: Face Detector (`face_detector.tflite`) — BlazeFace Short-Range

| Property | Value |
|---|---|
| **File** | `face_detector.tflite` (inside `.task` bundle) |
| **Architecture** | BlazeFace (lightweight SSD-based detector) |
| **Input** | RGB image (any resolution via ImageToTensorCalculator) |
| **Output** | `Detection` protos: bounding box + 6 keypoints |
| **Keypoints** | 0=left eye, 1=right eye, 2=nose tip, 3=mouth center, 4=left ear tragion, 5=right ear tragion |
| **Source** | [`face_detection_short_range_cpu.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt) |

```
Input:  image (any resolution)
Output: Detection {
  relative_bounding_box: { xmin, ymin, width, height }   // normalized [0,1]
  relative_keypoints: [
    { x, y },   // 0: left eye
    { x, y },   // 1: right eye
    { x, y },   // 2: nose tip
    { x, y },   // 3: mouth center
    { x, y },   // 4: left ear tragion
    { x, y },   // 5: right ear tragion
  ]
}
```

### Model 2: Face Landmark Detector (`face_landmarks_detector.tflite`)

| Property | Value |
|---|---|
| **File** | `face_landmarks_detector.tflite` (inside `.task` bundle) |
| **Input** | RGB image cropped to ROI, letterbox-fitted to **192×192**, normalized `[0.0, 1.0]` |
| **Output (basic)** | 1 landmark tensor (468×3) + 1 face presence flag tensor |
| **Output (with attention)** | 6 tensors merged into **478** landmarks + 1 face presence flag tensor |
| **Face presence** | Sigmoid activation → threshold 0.5 |
| **Source** | [`face_landmark_cpu.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt), [`tensors_to_face_landmarks_with_attention.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt) |

```
Input:  192×192 RGB image, float [0,1], cropped from the expanded ROI
Output without attention:
  tensor[0]: 468 landmarks (x, y, z) normalized to [0, 192] range
  tensor[1]: face presence flag (raw logit)

Output with attention (6 tensors, merged into 478 landmarks):
  tensor[0]: 468 base mesh landmarks
  tensor[1]: 80 refined lip landmarks
  tensor[2]: 71 refined left eye landmarks
  tensor[3]: 71 refined right eye landmarks
  tensor[4]: 5 left iris landmarks (indices 468–472)
  tensor[5]: 5 right iris landmarks (indices 473–477)
  tensor[6]: face presence flag
```

### Model 3: Blendshape Predictor (`face_blendshapes.tflite`) — Optional

| Property | Value |
|---|---|
| **File** | `face_blendshapes.tflite` (only in blendshape variant of `.task`) |
| **Input** | 468/478 face landmarks |
| **Output** | 52 blendshape coefficients (0.0–1.0) |

---

## 2. The Two ROI Generation Paths

The pipeline has **two paths** for generating the ROI fed to the landmark model:

### Path A: From Face Detection (initial detection or re-detection)

Used when:
- Very first frame (no previous landmarks exist)
- Tracking is lost (face moved too much between frames)
- Not in stream mode (IMAGE mode)

```
FaceDetection
    │
    ▼
DetectionsToRectsCalculator       ← align to eye-to-eye vector
    │  rotation: left eye (KP0) → right eye (KP1)
    │  target angle: 0° (horizontal)
    ▼
RectTransformationCalculator
    │  scale_x: 1.5
    │  scale_y: 1.5
    │  square_long: true
    ▼
Expanded ROI → feed to FaceLandmark model
```

### Path B: From Previous Landmarks (tracking mode)

Used when:
- Previous frame's landmarks exist
- Tracking is maintained (face didn't move too far)

```
Previous frame's 468/478 landmarks
    │
    ▼
LandmarksToDetectionCalculator    ← compute tight bbox enclosing ALL landmarks
    │  (min/max of all 468/478 x,y coordinates)
    ▼
DetectionsToRectsCalculator       ← align to eye-to-eye vector
    │  rotation: left eye left edge (KP33) → right eye right edge (KP263)
    │  target angle: 0° (horizontal)
    ▼
RectTransformationCalculator
    │  scale_x: 1.5
    │  scale_y: 1.5
    │  square_long: true
    ▼
Expanded ROI → feed to FaceLandmark model (next frame)
```

> **Why Path B produces a larger area:** The `LandmarksToDetectionCalculator` creates a tight bounding box around **all 468/478 landmarks**, which includes the forehead, chin, and sides of the face. The face detector's box only covers eyes/nose/mouth. Even though both paths apply the same 1.5× expansion, Path B starts from a larger initial rect.

---

## 3. Step-by-Step ROI Transformation

### Step 0: LandmarksToDetectionCalculator (Path B only)

```cpp
// Source: landmarks_to_detection_calculator.cc
float x_min = max_float, x_max = min_float;
float y_min = max_float, y_max = min_float;
for (landmark in all_landmarks) {
    x_min = min(x_min, landmark.x());
    x_max = max(x_max, landmark.x());
    y_min = min(y_min, landmark.y());
    y_max = max(y_max, landmark.y());
}
// Output: tight bounding box enclosing ALL 468/478 landmarks
// relative_bounding_box: { xmin, ymin, width = xmax-xmin, height = ymax-ymin }
```

### Step 1: DetectionsToRectsCalculator (Both paths)

Converts the detection to a normalized rect, optionally with rotation:

```
Configuration:
  rotation_vector_start_keypoint_index: 33  (left side of left eye)
  rotation_vector_end_keypoint_index:   263 (right side of right eye)
  rotation_vector_target_angle_degrees: 0   (horizontal alignment)

Algorithm:
  1. Take the bounding box from the detection
  2. Compute rotation angle from keypoint 33 → keypoint 263 vector
  3. Set rotation so eye-line is horizontal (0 degrees)
  4. Output: NormalizedRect { x_center, y_center, width, height, rotation }
```

### Step 2: RectTransformationCalculator (Both paths)

```cpp
// Source: rect_transformation_calculator.cc
// Options applied:
//   scale_x: 1.5
//   scale_y: 1.5
//   square_long: true

void TransformNormalizedRect(NormalizedRect* rect, int image_width, int image_height) {
    float width  = rect->width();
    float height = rect->height();

    // --- Step 2a: Square the rect ---
    if (square_long) {
        // Convert to pixel, take max dimension, convert back to normalized
        float long_side = max(width * image_width, height * image_height);
        width  = long_side / image_width;
        height = long_side / image_height;
    }

    // --- Step 2b: Scale ---
    rect->set_width(width * scale_x);   // width  *= 1.5
    rect->set_height(height * scale_y); // height *= 1.5

    // --- Step 2c: Optional shift (not used in face landmark graphs) ---
    // shift_x = 0.0, shift_y = 0.0  (defaults)
}
```

### Step 3: Letterbox-Fit & Crop (ImagePreprocessingGraph)

Before the landmark model runs, the ROI is cropped from the image:

```
Algorithm:
  1. Crop the image to the expanded ROI rect (with rotation if any)
  2. Letterbox-fit the crop into 192×192 while preserving aspect ratio
     (adds black padding to the shorter side)
  3. Normalize pixel values to [0.0, 1.0]
  4. Feed to the landmark model
```

### Step 4: Post-processing (Coordinate Projection)

After landmark detection:

```
Algorithm:
  1. Remove letterbox padding adjustment from landmark coordinates
     (LandmarkLetterboxRemovalCalculator)
  2. Project landmarks from the 192×192 crop back to original image
     coordinates using the ROI rect
     (LandmarkProjectionCalculator)
```

---

## 4. Source Code References

| Component | File | Lines |
|---|---|---|
| Detection → ROI graph | [`face_detection_front_detection_to_roi.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt) | Full file (28 lines) |
| Landmarks → ROI graph | [`face_landmark_landmarks_to_roi.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt) | Full file (62 lines) |
| RectTransformation calculator | [`rect_transformation_calculator.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc) | Lines 59–111 (TransformRect/TransformNormalizedRect) |
| RectTransformation options proto | [`rect_transformation_calculator.proto`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.proto) | Full file |
| DetectionsToRects calculator | [`detections_to_rects_calculator.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/util/detections_to_rects_calculator.cc) | Lines 46–56 (NormRectFromKeyPoints), 175–199 (ComputeRotation) |
| LandmarksToDetection calculator | [`landmarks_to_detection_calculator.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/util/landmarks_to_detection_calculator.cc) | Lines 28–54 (ConvertLandmarksToDetection: min/max across all landmarks) |
| Tasks-vision single-face graph | [`face_landmarks_detector_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc) | Lines 140–148 and 132–138 (ConfigureFaceRectTransformationCalculator) |
| Tasks-vision full pipeline | [`face_landmarker_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc) | Lines 185–351 (BuildFaceLandmarkerGraph) |
| Landmark CPU subgraph | [`face_landmark_cpu.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt) | Full file (shows ImageToTensor 192×192, model I/O) |
| 478-landmark tensor decoding | [`tensors_to_face_landmarks_with_attention.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt) | Full file (shows 6-tensor split and refinement) |

---

## 5. Scale Factor Decision

### When does expansion happen?

| Condition | Path Used | Starting Rect | Expansion |
|---|---|---|---|
| **First frame** (no previous landmarks) | Path A (from detection) | Tight BlazeFace bbox around eyes/nose/mouth | 1.5× + square |
| **Tracking lost** (face moved too far, `min_similarity_threshold < 0.5`) | Path A (from detection) | Tight BlazeFace bbox | 1.5× + square |
| **Tracking maintained** (previous landmarks exist, IOU ≥ 0.5) | Path B (from landmarks) | Tight bbox around all 468/478 landmarks | 1.5× + square |
| **IMAGE mode** (not streaming) | Path A always | BlazeFace bbox | 1.5× + square |

### How is the scale factor determined?

The scale factor is **hardcoded** as `1.5` in the MediaPipe source:

```
options {
  [mediapipe.RectTransformationCalculatorOptions.ext] {
    scale_x: 1.5
    scale_y: 1.5
    square_long: true
  }
}
```

In the tasks-vision C++ API, it's set identically:

```cpp
void ConfigureFaceRectTransformationCalculator(
    mediapipe::RectTransformationCalculatorOptions* options) {
  options->set_scale_x(1.5f);
  options->set_scale_y(1.5f);
  options->set_square_long(true);
}
```

> **Why 1.5?** This is an empirically determined value. The BlazeFace tight bbox covers approximately the center 40–50% of the face area. Scaling by 1.5 on both axes (after squaring to the longer side) ensures the forehead, chin, and sides are included. Too large wastes computation on background; too small cuts off landmarks.

### The complete transformation formula:

```
Given a face detection bounding box: (cx, cy, w, h)  [normalized 0→1]

1. Square to long side:
   long = max(w * imgW, h * imgH)
   w' = long / imgW
   h' = long / imgH

2. Scale:
   w'' = w' * 1.5
   h'' = h' * 1.5

3. No shift applied (shift_x = 0, shift_y = 0)

Final ROI: (cx, cy, w'', h'')

Pixel dimensions of the crop fed to the landmark model:
   crop_px = max(w'' * imgW, h'' * imgH)  [this gets letterbox-fit to 192×192]
```

### For the tracking path (Path B), substitute the tight landmark bbox:

```
Given landmarks: [(x₁,y₁), (x₂,y₂), ..., (x₄₆₈,y₄₆₈)]  [normalized 0→1]

1. Compute tight bbox:
   xmin = min(x₁...x₄₆₈), xmax = max(x₁...x₄₆₈)
   ymin = min(y₁...y₄₆₈), ymax = max(y₁...y₄₆₈)
   cx = (xmin + xmax) / 2
   cy = (ymin + ymax) / 2
   w = xmax - xmin
   h = ymax - ymin

2. Square to long side:  [same as above]
3. Scale 1.5×:           [same as above]
```

---

## 6. Visual Summary

```
FACE DETECTION OUTPUT                  LANDMARK PIPELINE ROI
(tight box around eyes/nose/mouth)     (expanded 1.5× + square)

    ┌─────────────────┐                 ┌───────────────────────┐
    │                 │                 │                       │
    │    ┌─────────┐  │                 │     ┌─────────────┐   │
    │    │ ⚫  ⚫   │  │                 │     │  ⚫     ⚫   │   │
    │    │   nose  │  │     1.5×        │     │    nose     │   │
    │    │  mouth  │  │   ────────►     │     │   mouth     │   │
    │    └─────────┘  │                 │     │    chin     │   │
    │                 │                 │     └─────────────┘   │
    └─────────────────┘                 │                       │
                                        └───────────────────────┘

TRACKING PATH (from previous landmarks)
(tight bbox around ALL 468/478 landmarks → expands MORE)

    ┌───────────────────────┐           ┌───────────────────────────┐
    │  • • • • • • • • •   │           │                           │
    │  •           •       │           │    • • • • • • • • •     │
    │  •   ⚫  ⚫   •       │   1.5×    │    •               •     │
    │  •    nose   •       │ ────────►  │    •    ⚫   ⚫    •     │
    │  •   mouth   •       │           │    •     nose    •       │
    │  •    chin   •       │           │    •    mouth    •       │
    │  • • • • • • •       │           │    •     chin    •       │
    └───────────────────────┘           │    • • • • • • • •       │
                                        └───────────────────────────┘

    Tight landmark bbox (Path B)          Expanded landmark bbox
    is already larger than the                    1.5×
    detection bbox ↓                    is larger than
                                        expanded detection bbox (Path A) ↓
```

**Key takeaway:** The landmark output covers a larger area than the detection output because:
1. The tight bbox from all 468/478 landmarks is naturally larger than the BlazeFace 6-keypoint detection bbox
2. Both are expanded by the same 1.5× factor, so the absolute size difference is preserved
3. This is intentional — the mesh model needs forehead and chin data that BlazeFace doesn't include
