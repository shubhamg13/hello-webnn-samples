# YOLOv8-Face ONNX Model Specification

This document describes the input/output format, preprocessing, and postprocessing
pipeline for running inference with the YOLOv8-Face ONNX model.

---

## 1. Model Architecture Overview

The YOLOv8-Face model is based on the YOLOv8 architecture with a **Pose** detection head.
It performs face detection with 5-point facial landmark keypoints.

| Property | Value |
|----------|-------|
| Task | Pose (detection + keypoints) |
| Architecture | YOLOv8 P3–P5 (3 detection scales) |
| Input size | 640×640 (default) |
| Strides | [8, 16, 32] |
| Classes | 1 (`face`) |
| Keypoints | 5 (left-eye, right-eye, nose, left-mouth, right-mouth) × 3 (x, y, visibility) |
| DFL reg_max | 16 |

### Detection Head Output Channels

Each detection scale produces `no` channels per anchor point:

```
no = nc + reg_max * 4 + nk
   = 1  + 16 * 4     + 5 * 3
   = 1  + 64         + 15
   = 80
```

| Channel range | Content |
|---------------|---------|
| 0–63 | Bounding box regression (DFL-encoded distances: left, top, right, bottom) |
| 64 | Class confidence (face) |
| 65–79 | Keypoint offsets (5 keypoints × 3 dims: x, y, visibility) |

---

## 2. ONNX Model I/O

### Input

| Name | Shape | Dtype | Range | Layout |
|------|-------|-------|-------|--------|
| `images` | `(1, 3, 640, 640)` | float32 | [0.0, 1.0] | NCHW, **RGB** |

- **Static shape**: `(1, 3, 640, 640)` — fixed batch=1, fixed resolution.
- **Dynamic shape** (if `--dynamic` was used during export): `(batch, 3, height, width)`.
- The input must be normalized from `[0, 255]` uint8 to `[0.0, 1.0]` float32.
- Channel order is **RGB** (not BGR). If your image is BGR (OpenCV default), convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

### Outputs

The model produces **3 outputs**, one per detection scale:

| Output | Shape (static) | Stride | Grid size |
|--------|---------------|--------|-----------|
| `output0` | `(1, 80, 80, 80)` | 8 | 80×80 |
| `output1` | `(1, 80, 40, 40)` | 16 | 40×40 |
| `output2` | `(1, 80, 20, 20)` | 32 | 20×20 |

> **Note**: The exact output names depend on the PyTorch ONNX export. They may appear as
> `output0`, `output1`, `output2` or `output0`, `output0_1`, `output0_2`. Check with
> [Netron](https://netron.app/) after export.

Each output tensor encodes, for every grid cell:

```
channels 0–63:  bbox regression (4 sides × 16 distribution values = 64)
channel  64:    class score (face confidence, raw logit)
channels 65–79: keypoint offsets (5 keypoints × [dx, dy, visibility])
```

---

## 3. Preprocessing Pipeline

```text
Input Image (H_orig × W_orig × 3, BGR, uint8, [0, 255])
       │
       ▼
1. BGR → RGB
       │
       ▼
2. Letterbox resize to (640, 640)
   - Compute scale:  r = min(640 / H_orig, 640 / W_orig)
   - Resize:         new_w = round(W_orig * r), new_h = round(H_orig * r)
   - Padding (gray): pad_w = (640 - new_w) // 2, pad_h = (640 - new_h) // 2
       │
       ▼
3. Normalize: img = img / 255.0  →  [0.0, 1.0]
       │
       ▼
4. Reorder: HWC → CHW  (channels-first)
       │
       ▼
5. Add batch dim: (3, 640, 640) → (1, 3, 640, 640)
       │
       ▼
   Model Input (1, 3, 640, 640, float32)
```

### Preprocessing Code (Python / ONNX Runtime)

```python
import cv2
import numpy as np

def preprocess(image_path: str, imgsz: int = 640) -> tuple[np.ndarray, tuple, tuple, float]:
    """
    Preprocess an image for YOLOv8-Face ONNX inference.

    Args:
        image_path: Path to the input image.
        imgsz: Target square image size.

    Returns:
        blob:        Preprocessed tensor (1, 3, H, W), float32, [0,1].
        orig_shape:  Original image shape (H, W).
        new_shape:   Resized shape before padding (new_h, new_w).
        ratio:       Scale ratio applied to the original image.
    """
    img = cv2.imread(image_path)                        # BGR, uint8
    orig_shape = img.shape[:2]                          # (H, W)

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox resize + pad to (imgsz, imgsz)
    r = min(imgsz / orig_shape[0], imgsz / orig_shape[1])
    new_unpad = (int(round(orig_shape[1] * r)), int(round(orig_shape[0] * r)))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]
    dw, dh = dw // 2, dh // 2  # center padding

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Normalize & reorder
    img = img.astype(np.float32) / 255.0                # [0, 255] → [0, 1]
    img = img.transpose((2, 0, 1))                      # HWC → CHW
    img = np.expand_dims(img, axis=0)                   # (C, H, W) → (1, C, H, W)
    img = np.ascontiguousarray(img)

    return img, orig_shape, new_unpad, r
```

### Key Preprocessing Details

| Step | Detail |
|------|--------|
| Color space | **RGB** — the model was trained on RGB images |
| Resize interpolation | `cv2.INTER_LINEAR` (bilinear) |
| Padding value | `(114, 114, 114)` — ImageNet mean ≈ gray |
| Padding mode | `cv2.BORDER_CONSTANT` |
| Normalization | Divide by 255, **no** mean/std subtraction |
| Layout | NCHW (batch, channels, height, width) |

---

## 4. Postprocessing Pipeline

```text
Model Outputs: 3 tensors (1, 80, Hᵢ, Wᵢ) for i ∈ {0,1,2}
       │
       ▼
1. Reshape + Concat
   - Reshape each: (1, 80, Hᵢ, Wᵢ) → (1, 80, Hᵢ*Wᵢ)
   - Concat all:   (1, 80, 8400)    [80² + 40² + 20² = 8400]
       │
       ▼
2. Split channels
   - box_reg:  (1, 64, 8400)   ← channels 0–63
   - cls_raw:  (1,  1, 8400)   ← channel  64
   - kpt_raw:  (1, 15, 8400)   ← channels 65–79
       │
       ▼
3. Bounding Box Decoding
   a. DFL (Distribution Focal Loss) softmax + weighted sum:
        softmax over the 16 distribution values per side
        → (1, 4, 8400) distances: [left, top, right, bottom]

   b. dist2bbox (distance to xywh):
        anchors = grid cell centers (0.5 offset) for each scale
        cx = anchor_x - left_dist
        cy = anchor_y - top_dist
        w  = left_dist + right_dist
        h  = top_dist + bottom_dist
        → (1, 4, 8400) in xywh format, pixel coords

   c. Scale by stride:
        cx *= stride, cy *= stride, w *= stride, h *= stride
       (stride 8 for P3 grid, 16 for P4, 32 for P5)
       → xywh in original 640×640 space

   d. Convert xywh → xyxy:
        x1 = cx - w/2,  y1 = cy - h/2
        x2 = cx + w/2,  y2 = cy + h/2
       │
       ▼
4. Class score: sigmoid(cls_raw) → confidence
       │
       ▼
5. Keypoint Decoding
   - For each keypoint (x, y, vis):
        kpt_x = (raw_x * 2.0 + (anchor_x - 0.5)) * stride
        kpt_y = (raw_y * 2.0 + (anchor_y - 0.5)) * stride
        kpt_vis = sigmoid(raw_vis)
       │
       ▼
6. NMS (Non-Maximum Suppression)
   - Filter by confidence threshold (e.g., 0.25)
   - Class-agnostic or per-class NMS with IoU threshold (e.g., 0.45)
   - Keep top-K detections (e.g., max 300)
       │
       ▼
7. Scale boxes back to original image dimensions
   - Undo letterbox padding and resize
   - Clip to image boundaries
```

### Postprocessing Code (Python / NumPy)

```python
import numpy as np

# Constants
REG_MAX = 16
STRIDES = np.array([8, 16, 32], dtype=np.float32)
NC = 1       # number of classes (face = 1)
NK = 15       # number of keypoint values (5 * 3)

def make_anchors(feats, strides, offset=0.5):
    """Generate anchor grid points for each feature map."""
    anchors, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(w) + offset
        sy = np.arange(h) + offset
        sx, sy = np.meshgrid(sx, sy, indexing='ij')
        anchors.append(np.stack((sx, sy), axis=-1).reshape(-1, 2))
        stride_tensor.append(np.full(h * w, stride))
    return np.concatenate(anchors, axis=0), np.concatenate(stride_tensor, axis=0)


def dfl(x, reg_max=16):
    """Distribution Focal Loss decode: softmax + weighted sum."""
    # x shape: (4, reg_max * anchors) → reshape to (4, anchors, reg_max)
    anchors = x.shape[1] // reg_max
    x = x.reshape(4, anchors, reg_max)
    x = softmax(x, axis=-1)  # softmax over the 16 values
    weights = np.arange(reg_max, dtype=x.dtype)
    return (x * weights).sum(axis=-1)  # (4, anchors)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def postprocess(outputs, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Post-process YOLOv8-Face ONNX outputs.

    Args:
        outputs: List of 3 numpy arrays, each (1, 80, H, W).
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum number of detections to keep.

    Returns:
        boxes:     (N, 4) xyxy bounding boxes in 640×640 space.
        scores:    (N,) confidence scores.
        keypoints: (N, 5, 3) keypoints (x, y, visibility).
    """
    # 1. Generate anchors
    feats = [o[0] for o in outputs]  # remove batch dim for anchor gen
    anchors, strides = make_anchors([o[0] for o in outputs], STRIDES)  # (8400, 2), (8400,)

    # 2. Reshape & concat
    reshaped = []
    for o in outputs:
        _, c, h, w = o.shape
        reshaped.append(o.reshape(1, c, h * w))
    x = np.concatenate(reshaped, axis=2)  # (1, 80, 8400)

    # 3. Split
    box_reg = x[:, :REG_MAX * 4, :]    # (1, 64, 8400)
    cls_raw = x[:, REG_MAX * 4:REG_MAX * 4 + NC, :]  # (1, 1, 8400)
    kpt_raw = x[:, REG_MAX * 4 + NC:, :]  # (1, 15, 8400)

    # 4. Decode bounding boxes
    box_reg = box_reg[0]  # (64, 8400)
    dist = dfl(box_reg)   # (4, 8400) — left, top, right, bottom

    anchors_t = anchors.T  # (2, 8400)
    strides_t = strides.reshape(1, -1)  # (1, 8400)

    # dist2bbox: distance → xywh
    lt, rb = dist[:2], dist[2:]  # (left,top), (right,bottom)
    x1y1 = anchors_t - lt        # (2, 8400)
    x2y2 = anchors_t + rb        # (2, 8400)
    cx = (x1y1 + x2y2) / 2.0
    wh = x2y2 - x1y1
    xywh = np.concatenate([cx, wh], axis=0)  # (4, 8400) xywh

    # Scale by stride
    xywh *= strides_t  # pixel coordinates in 640×640

    # xywh → xyxy
    x1 = xywh[0] - xywh[2] / 2.0
    y1 = xywh[1] - xywh[3] / 2.0
    x2 = xywh[0] + xywh[2] / 2.0
    y2 = xywh[1] + xywh[3] / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)  # (8400, 4)

    # 5. Confidence scores
    scores = 1.0 / (1.0 + np.exp(-cls_raw[0, 0, :]))  # sigmoid

    # 6. Keypoint decoding
    kpt_raw = kpt_raw[0].reshape(NK // 3, 3, -1)  # (5, 3, 8400)
    kpt_xy = kpt_raw[:, :2, :]                     # (5, 2, 8400)
    kpt_vis = kpt_raw[:, 2:3, :]                   # (5, 1, 8400)

    # Decode: (x * 2.0 + (anchor - 0.5)) * stride
    kpt_x = (kpt_xy[:, 0, :] * 2.0 + (anchors_t[0:1, :] - 0.5)) * strides_t
    kpt_y = (kpt_xy[:, 1, :] * 2.0 + (anchors_t[1:2, :] - 0.5)) * strides_t
    kpt_vis = 1.0 / (1.0 + np.exp(-kpt_vis[:, 0, :]))  # sigmoid
    kpts = np.stack([kpt_x, kpt_y, kpt_vis], axis=2)  # (5, 8400, 3)
    kpts = kpts.transpose(1, 0, 2)                     # (8400, 5, 3)

    # 7. Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    kpts = kpts[mask]

    # 8. NMS (simple class-agnostic)
    keep = nms(boxes, scores, iou_thres)
    keep = keep[:max_det]

    return boxes[keep], scores[keep], kpts[keep]


def nms(boxes, scores, iou_thres):
    """Simple class-agnostic NMS."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])
        order = order[1:][ious[0] < iou_thres]
    return np.array(keep)


def box_iou(box1, box2):
    """Compute IoU between two sets of boxes (xyxy format)."""
    x1 = np.maximum(box1[:, 0:1], box2[:, 0])
    y1 = np.maximum(box1[:, 1:2], box2[:, 1])
    x2 = np.minimum(box1[:, 2:3], box2[:, 2])
    y2 = np.minimum(box1[:, 3:4], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-16)


def scale_boxes_back(boxes, orig_shape, ratio, dw, dh, imgsz=640):
    """
    Scale boxes from 640×640 model space back to original image space.

    Args:
        boxes:      (N, 4) boxes in xyxy format, model space.
        orig_shape: Original image (H, W).
        ratio:      Scale ratio used during letterboxing.
        dw, dh:     Padding added to each side.
        imgsz:      Model input size.

    Returns:
        Scaled boxes in original image coordinates.
    """
    # Undo padding
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    # Undo scale
    boxes /= ratio
    # Clip
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes
```

---

## 5. Full ONNX Inference Example

```python
import cv2
import numpy as np
import onnxruntime as ort

# ── Load ONNX model ──
session = ort.InferenceSession('yolov8n-face.onnx', providers=['CPUExecutionProvider'])

# ── Preprocess ──
img = cv2.imread('test.jpg')
orig_h, orig_w = img.shape[:2]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r = min(640 / orig_h, 640 / orig_w)
new_h, new_w = int(round(orig_h * r)), int(round(orig_w * r))
dw, dh = (640 - new_w) // 2, (640 - new_h) // 2

img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
img_padded = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
blob = np.ascontiguousarray(img_padded.transpose(2, 0, 1)[np.newaxis]).astype(np.float32) / 255.0

# ── Inference ──
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: blob})

# ── Postprocess ──
boxes, scores, kpts = postprocess(outputs, conf_thres=0.25, iou_thres=0.45)
boxes = scale_boxes_back(boxes, (orig_h, orig_w), r, dw, dh)

# ── Display results ──
for box, score, kpt in zip(boxes, scores, kpts):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{score:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for kx, ky, kv in kpt:
        if kv > 0.5:
            cv2.circle(img, (int(kx * orig_w / 640 / r + dw * ...)), ...)  # scale back
            # Scale keypoints back similarly

cv2.imwrite('result.jpg', img)
```

---

## 6. Key Constants Summary

| Symbol | Value | Description |
|--------|-------|-------------|
| `imgsz` | 640 | Input image size |
| `nc` | 1 | Number of classes (face) |
| `nk` | 15 | Keypoint output channels (5 kpts × 3) |
| `reg_max` | 16 | DFL distribution bins per bbox side |
| `no` | 80 | Total output channels per anchor |
| `strides` | [8, 16, 32] | Strides for P3, P4, P5 |
| `num_anchors` | 8400 | Total anchors (80² + 40² + 20²) |
| `grid_cell_offset` | 0.5 | Anchor grid offset |

---

## 7. References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Netron — ONNX model visualizer](https://netron.app/)
- This repo: [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face)
