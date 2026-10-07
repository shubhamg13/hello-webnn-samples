// ============================================================================
// YOLODetector — preprocessing, postprocessing, and drawing for the
// YOLOv8n-Face model (640×640 NCHW input, DFL + dist2bbox decode,
// 5 facial keypoints).
// ============================================================================

const YOLODetector = (() => {

  // ---- model constants (match ONNX_INFERENCE.md) ----
  const IMGSZ = 640;       // input image size (letterbox square)
  const REG_MAX = 16;      // DFL bins per bbox edge (before decode)
  const STRIDES = [8, 16, 32];
  const NC = 1;            // number of classes (face)
  const NK = 15;           // number of keypoint values (5 kpts × 3: x, y, visibility)

  // Raw YOLO output (before DFL): [1, 80, H, W] — 80 channels per anchor
  // DFL-decoded output:       [1, 20, H, W] — 20 channels per anchor
  //
  // DFL-decoded channel layout (0-indexed) — what postprocess() expects:
  //    0:        bbox-left   decoded distance (scalar, grid-cell units)
  //    1:        bbox-top    decoded distance
  //    2:        bbox-right  decoded distance
  //    3:        bbox-bottom decoded distance
  //    4:        class confidence (raw logit, before sigmoid)
  //    5..7:     keypoint 0  [x_raw, y_raw, vis_raw]  left eye
  //    8..10:    keypoint 1  [x_raw, y_raw, vis_raw]  right eye
  //   11..13:    keypoint 2  [x_raw, y_raw, vis_raw]  nose
  //   14..16:    keypoint 3  [x_raw, y_raw, vis_raw]  left mouth corner
  //   17..19:    keypoint 4  [x_raw, y_raw, vis_raw]  right mouth corner

  const NUM_ANCHORS = 8400;  // 80² + 40² + 20² = 6400 + 1600 + 400

  const CONF_THRES = 0.25;
  const IOU_THRES = 0.45;
  const MAX_DET = 300;

  // YOLOv8-face keypoints: left-eye, right-eye, nose, left-mouth, right-mouth
  const KEYPOINT_NAMES = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth'];
  const KEYPOINT_COLORS = ['#FF0000', '#FF0000', '#00AA00', '#FF8800', '#FF8800'];

  // ---- numeric helpers ----
  function sigmoid(x) {
    if (x < -80) x = -80;
    if (x > 80) x = 80;
    return 1.0 / (1.0 + Math.exp(-x));
  }

  // ==========================================================================
  // PREPROCESSING
  // ==========================================================================

  /**
   * Letterbox-resize an image or canvas to 640×640 with gray padding.
   * Returns raw RGBA uint8 pixels — the WebNN graph handles slice (RGBA→RGB),
   * cast (uint8→float32), normalization (÷255), and NHWC→NCHW transpose.
   *
   * @param {HTMLImageElement|HTMLCanvasElement} imageElement
   * @returns {{ pixels: Uint8ClampedArray, origShape: [number,number],
   *             ratio: number, dw: number, dh: number, letterboxCanvas: HTMLCanvasElement }}
   */
  // Cached letterbox canvas - reused across frames
  let _cachedLbCanvas = null;
  let _cachedLbCtx = null;

  function preprocess(imageElement) {
    const origW = imageElement.naturalWidth || imageElement.width;
    const origH = imageElement.naturalHeight || imageElement.height;

    // letterbox scale
    const r = Math.min(IMGSZ / origH, IMGSZ / origW);
    const newW = Math.round(origW * r);
    const newH = Math.round(origH * r);
    const dw = Math.floor((IMGSZ - newW) / 2);
    const dh = Math.floor((IMGSZ - newH) / 2);

    // Reuse cached 640x640 canvas
    if (!_cachedLbCanvas) {
      _cachedLbCanvas = document.createElement('canvas');
      _cachedLbCanvas.width = IMGSZ;
      _cachedLbCanvas.height = IMGSZ;
      _cachedLbCtx = _cachedLbCanvas.getContext('2d');
    }
    const canvas = _cachedLbCanvas;
    const ctx = _cachedLbCtx;

    // gray padding (114, 114, 114)
    ctx.fillStyle = 'rgb(114, 114, 114)';
    ctx.fillRect(0, 0, IMGSZ, IMGSZ);
    ctx.drawImage(imageElement, dw, dh, newW, newH);
    const imageData = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
    // Raw RGBA uint8 pixels
    const pixels = imageData.data;  // Uint8ClampedArray, RGBA [0,255], NHWC layout

    return {
      pixels,
      origShape: [origW, origH],
      ratio: r,
      dw,
      dh,
      letterboxCanvas: canvas
    };
  }

  // ---- debug helpers ----
  function showLetterboxPreview(canvas, dw, dh) {
    const debugCanvas = document.getElementById('letterboxCanvas');
    if (!debugCanvas) return;
    const debugCtx = debugCanvas.getContext('2d');
    debugCtx.imageSmoothingEnabled = false;
    // scale 640→256
    debugCtx.drawImage(canvas, 0, 0, 256, 256);
    document.getElementById('debugCard').style.display = '';
    const info = document.getElementById('paddingInfo');
    if (info) {
      info.innerHTML =
        '<strong>Letterbox 640×640 (YOLO):</strong><br>' +
        'dw (left/right)=' + dw + 'px<br>' +
        'dh (top/bottom)=' + dh + 'px<br>' +
        '<em style="font-size:0.75rem;color:#888;">Gray padding (114,114,114). Boxes shown in model space before scale-back.</em>';
    }
  }

  function drawRawBoxesOnLetterbox(refCanvas, boxes, scores, dw, dh) {
    const debugCanvas = document.getElementById('letterboxCanvas');
    if (!debugCanvas) return;
    const debugCtx = debugCanvas.getContext('2d');
    debugCtx.imageSmoothingEnabled = false;

    debugCtx.clearRect(0, 0, 256, 256);
    debugCtx.drawImage(refCanvas, 0, 0, 256, 256);

    // pad boundary (dashed red)
    const scale256 = 256 / IMGSZ;
    debugCtx.strokeStyle = 'rgba(255,0,0,0.5)';
    debugCtx.lineWidth = 1;
    debugCtx.setLineDash([4, 4]);
    debugCtx.strokeRect(dw * scale256, dh * scale256,
      (IMGSZ - 2 * dw) * scale256, (IMGSZ - 2 * dh) * scale256);
    debugCtx.setLineDash([]);

    for (let i = 0; i < boxes.length; i++) {
      const [x1, y1, x2, y2] = boxes[i];
      debugCtx.strokeStyle = '#00FF00';
      debugCtx.lineWidth = 2;
      debugCtx.strokeRect(x1 * scale256, y1 * scale256,
        (x2 - x1) * scale256, (y2 - y1) * scale256);
      debugCtx.fillStyle = '#00FF00';
      debugCtx.font = '10px monospace';
      debugCtx.fillText((scores[i] * 100).toFixed(0) + '%',
        x1 * scale256, y1 * scale256 - 2);
    }
  }

  // ==========================================================================
  // POSTPROCESSING
  // ==========================================================================

  /**
   * Generate anchor grid points for a set of feature-map dimensions.
   *
   * The three feature maps are grids at strides 8, 16, 32:
   *   - feats[0] = { h: 80, w: 80 }  stride=8   →  6400 anchors
   *   - feats[1] = { h: 40, w: 40 }  stride=16  →  1600 anchors
   *   - feats[2] = { h: 20, w: 20 }  stride=32  →   400 anchors
   *   total = 6400 + 1600 + 400 = 8400 anchors
   *
   * Each anchor is a grid cell center: (gx + 0.5, gy + 0.5). The 0.5 offset
   * centers the anchor within the cell; for the dist2bbox formula to work
   * correctly, the raw predicted edge distances must be relative to this center.
   *
   * @param {Array<{h:number,w:number}>} feats  feature-map dims per scale
   * @returns {{
   *   anchors: Float64Array,   // shape [total * 2], interleaved [ax0,ay0, ax1,ay1, ...]
   *   strides: Float64Array    // shape [total], stride per anchor [8,8,..., 16,16,..., 32,32,...]
   * }}
   */
  // Cached anchor grid - identical for every frame, compute once
  let _cachedAnchors = null;
  let _cachedFeatsKey = null;
  function makeAnchors(feats) {
    var key = feats.map(function(f) { return f.h + 'x' + f.w; }).join(',');
    if (_cachedAnchors && _cachedFeatsKey === key) return _cachedAnchors;

    const total = feats.reduce((s, f) => s + f.h * f.w, 0);  // total = 8400
    const anchors = new Float64Array(total * 2);              // length = 16800
    const strideArr = new Float64Array(total);                // length = 8400
    let off = 0;
    for (let i = 0; i < STRIDES.length; i++) {
      const { h, w } = feats[i];
      const stride = STRIDES[i];
      // C-order (row-major): gy varies slowest, gx fastest → matches NCHW layout
      for (let gy = 0; gy < h; gy++) {
        for (let gx = 0; gx < w; gx++) {
          anchors[off * 2]     = gx + 0.5;   // ax: grid X center (value ∈ [0.5, W-0.5])
          anchors[off * 2 + 1] = gy + 0.5;   // ay: grid Y center (value ∈ [0.5, H-0.5])
          strideArr[off] = stride;           // {8, 16, or 32}
          off++;
        }
      }
    }
    _cachedAnchors = { anchors, strides: strideArr };
    _cachedFeatsKey = key;
    return _cachedAnchors;
  }

  /**
   * [LEGACY] DFL (Distribution Focal Loss) decode: convert 16 raw distribution
   * values to a single scalar distance via softmax + weighted sum.
   *
   * NOTE: This function is no longer used by postprocess().  The DFL decode
   * is now performed in a WebNN graph (see buildDFLGraph in
   * yolov8n-face-graph.js) for GPU/NPU acceleration.  Kept here for
   * reference and standalone debugging.
   *
   * Shape trace for one call:
   *   Input:  data = Float32Array[80 * H * W]  ← e.g. 80 × 6400 = 512000 floats for scale 0
   *           chStart ∈ {0, 16, 32, 48}
   *           base = gy * W + gx  ∈ [0, H*W)
   *   Step 1: v = [16 raw logits] from channels [chStart, chStart+15] at spatial position base
   *           data indexing: (chStart + d) * plane + base
   *   Step 2: softmax(v) → expV[i] / sumExp  → 16 probabilities summing to 1
   *   Step 3: dist = Σᵢ pᵢ · i  for i ∈ [0, 15]
   *           This is a weighted average of bin indices → bounded in [0, 15]
   *   Output: a scalar distance (in grid-cell units before stride scaling)
   *
   * @param {Float32Array} data   one scale's output tensor, shape [80, H, W] flattened (NCHW, N=1 dropped)
   * @param {number} chStart      first channel for this bbox edge (left=0, top=16, right=32, bottom=48)
   * @param {number} H            feature map height (80/40/20)
   * @param {number} W            feature map width (80/40/20)
   * @param {number} base         spatial position = gy * W + gx
   * @returns {number} decoded distance in grid-cell units (range ~[0, 15])
   */
  function dflSide(data, chStart, H, W, base) {
    const plane = H * W;  // pixels per channel: 6400 / 1600 / 400
    // Step 1: gather 16 raw logits from channels chStart..chStart+15
    const v = new Array(REG_MAX);  // length 16
    for (let d = 0; d < REG_MAX; d++) {
      v[d] = data[(chStart + d) * plane + base];
    }
    // Step 2: numerically stable softmax → 16 probabilities
    let maxV = v[0];
    for (let d = 1; d < REG_MAX; d++) { if (v[d] > maxV) maxV = v[d]; }
    let sumExp = 0;
    const expV = new Array(REG_MAX);
    for (let d = 0; d < REG_MAX; d++) {
      const e = Math.exp(v[d] - maxV);
      expV[d] = e;
      sumExp += e;
    }
    // Step 3: weighted sum — bin index weighted by probability
    // dist = expected value of the distribution → sub-pixel edge distance
    let dist = 0;
    for (let d = 0; d < REG_MAX; d++) {
      dist += (expV[d] / sumExp) * d;
    }
    return dist;  // scalar ∈ roughly [0, 15], may extend beyond with unbounded inputs
  }

  /**
   * Full postprocessing pipeline for YOLOv8n-Face.
   *
   * The DFL (Distribution Focal Loss) decode is assumed to have already run
   * in a WebNN graph before this function is called.  This function expects
   * 20-channel DFL-decoded tensors, not the raw 80-channel backbone outputs.
   *
   * ============ OVERALL SHAPE FLOW ============
   *
   * Step 0 — Inputs (DFL-decoded):
   *   outputs[0]: Float32Array[1×20×80×80]  = 128000 values  (stride-8,  6400 anchors)
   *   outputs[1]: Float32Array[1×20×40×40]  =  32000 values  (stride-16, 1600 anchors)
   *   outputs[2]: Float32Array[1×20×20×20]  =   8000 values  (stride-32,  400 anchors)
   *                                                             total = 8400 anchors
   *   shapes:     [[1,20,80,80], [1,20,40,40], [1,20,20,20]]
   *
   * Step 0a — Per-scale feats:
   *   feats = [{h:80,w:80}, {h:40,w:40}, {h:20,w:20}]
   *
   * Step 0b — makeAnchors(feats):
   *   anchors:  Float64Array[16800]  interleaved [ax0,ay0, ax1,ay1, ...]
   *   strideArr: Float64Array[8400]  [8,8,..., 16,16,..., 32,32,...]
   *
   * Step 1 — Per-anchor loop (3 scales × H×W positions = 8400 iterations):
   *
   *   For each (scale, gy, gx) → anchorIdx ∈ [0, 8399]:
   *
   *   1a. Read pre-decoded bbox edge distances from channels 0..3:
   *         left   = data[0 * plane + base]  (scalar ∈ ~[0,15], grid units)
   *         top    = data[1 * plane + base]
   *         right  = data[2 * plane + base]
   *         bottom = data[3 * plane + base]
   *         (DFL softmax + weighted sum already done in WebNN)
   *
   *   1b. dist2bbox — grid → pixel:
   *         x1 = (ax - left)   * stride  → 640×640 image x
   *         y1 = (ay - top)    * stride  → 640×640 image y
   *         x2 = (ax + right)  * stride
   *         y2 = (ay + bottom) * stride
   *         where ax = gx + 0.5,  ay = gy + 0.5,  stride ∈ {8,16,32}
   *         Box format: [x1, y1, x2, y2]  top-left / bottom-right corners
   *
   *   1c. Class score (channel 4):
   *         Channel 4 = raw logit  →  sigmoid → score ∈ [0, 1]
   *
   *   1d. Keypoints (5 kpts × [kx, ky, kv], channels 5..19):
   *         For each k ∈ {0..4}, channels 5+k*3 .. 7+k*3:
   *           kx = (kxRaw * 2.0 + (ax - 0.5)) * stride  → 640×640 pixel x
   *           ky = (kyRaw * 2.0 + (ay - 0.5)) * stride  → 640×640 pixel y
   *           kv = sigmoid(kvRaw)                        → visibility ∈ [0, 1]
   *
   *   1e. Confidence filter (per anchor):
   *         if score > confThres AND x2 > x1 AND y2 > y1:
   *           push to allBoxes[], allScores[], allKpts[]
   *
   * Step 2 — NMS (class-agnostic, xyxy IoU):
   *   allBoxes:  Array<[x1,y1,x2,y2]>  len = preNmsCount (≤ 8400)
   *   allScores: number[]               len = preNmsCount
   *   allKpts:   Array<5×[kx,ky,kv]>   len = preNmsCount
   *       ↓
   *   nms() → keep: number[]  len ≤ preNmsCount, sorted by descending score
   *       ↓   keep.slice(0, MAX_DET=300)
   *   keep: number[]  len ≤ 300
   *
   * Step 3 — Gather kept detections:
   *   boxes:     Array<[x1,y1,x2,y2]>  len ≤ 300  (model 640×640 space)
   *   scores:    number[]               len ≤ 300
   *   keypoints: Array<5×[kx,ky,kv]>   len ≤ 300  (model 640×640 space)
   *
   * ======== CHANNEL LAYOUT (20 channels per anchor, DFL-decoded) ========
   *
   *   ch 0:       bbox-left   decoded distance (scalar, grid-cell units)
   *   ch 1:       bbox-top    decoded distance
   *   ch 2:       bbox-right  decoded distance
   *   ch 3:       bbox-bottom decoded distance
   *   ch 4:       class score (raw logit, before sigmoid)
   *   ch 5..7:    keypoint 0  [x_raw, y_raw, vis_raw]  left eye
   *   ch 8..10:   keypoint 1  [x_raw, y_raw, vis_raw]  right eye
   *   ch 11..13:  keypoint 2  [x_raw, y_raw, vis_raw]  nose
   *   ch 14..16:  keypoint 3  [x_raw, y_raw, vis_raw]  left mouth corner
   *   ch 17..19:  keypoint 4  [x_raw, y_raw, vis_raw]  right mouth corner
   *
   * @param {Array<Float32Array>} outputs  3 NCHW scale tensors (N=1 dropped)
   *   When useDFLGraph=true:  DFL-decoded, shape [1,20,H,W] (ch 0-3 bbox, ch 4 cls, ch 5-19 kpts)
   *   When useDFLGraph=false: raw backbone,  shape [1,80,H,W] (ch 0-63 DFL bins, ch 64 cls, ch 65-79 kpts)
   * @param {Array<Array<number>>} shapes  corresponding tensor shapes
   * @param {number} [confThres]  confidence threshold (default 0.25)
   * @param {number} [iouThres]   IoU threshold for NMS (default 0.45)
   * @param {boolean} [useDFLGraph=true]  true = inputs are DFL-decoded (20ch); false = raw (80ch, JS dflSide)
   * @returns {{
   *   boxes: Array<[number,number,number,number]>,  // xyxy in 640×640 model space
   *   scores: number[],
   *   keypoints: Array<Array<[number,number,number]>>,  // each kpt: [x, y, visibility]
   *   preNmsCount: number  // detections before NMS (for debugging)
   * }}
   */
  function postprocess(outputs, shapes, confThres, iouThres, useDFLGraph) {
    if (confThres === undefined) confThres = CONF_THRES;
    if (iouThres === undefined) iouThres = IOU_THRES;
    if (useDFLGraph === undefined) useDFLGraph = true;
    console.log('post process use DFLGraph:', useDFLGraph);
    // Step 0a: extract feature-map dimensions
    // feats = [{h:80,w:80}, {h:40,w:40}, {h:20,w:20}]
    const feats = shapes.map(s => ({ h: s[2], w: s[3] }));

    // Step 0b: generate anchor grid
    // anchors: Float64Array[16800], strides: Float64Array[8400]
    const { anchors, strides: strideArr } = makeAnchors(feats);

    const totalAnchors = anchors.length / 2;  // 8400

    // Pre-allocated accumulators — reused across frames, reset via detCount
    if (!_allBoxes) {
      _allBoxes = new Array(8400);
      _allScores = new Array(8400);
      _allKpts = new Array(8400);
    }
    let detCount = 0;

    let anchorIdx = 0;
    // Iterate over the 3 detection scales
    for (let scale = 0; scale < outputs.length; scale++) {
      const data = outputs[scale];   // Float32Array, length = 20 * H * W (DFL-decoded)
      const [_, C, H, W] = shapes[scale];
      // C = 20 (channels), H = {80,40,20}, W = {80,40,20}
      const plane = H * W;      // spatial elements per channel: {6400, 1600, 400}
      const nAnchors = H * W;   // {6400, 1600, 400}

      // Walk grid in row-major (C-order): gy outer, gx inner
      for (let gy = 0; gy < H; gy++) {
        for (let gx = 0; gx < W; gx++) {
          const base = gy * W + gx;  // spatial offset ∈ [0, plane)
          const ax = anchors[anchorIdx * 2];       // grid cell center X
          const ay = anchors[anchorIdx * 2 + 1];   // grid cell center Y
          const stride = strideArr[anchorIdx];     // {8, 16, or 32}
          anchorIdx++;

          // ---- Step 1a: decode bbox edge distances ----
          let left, top, right, bottom;
          if (useDFLGraph) {
            // DFL already done in WebNN — read decoded scalars directly
            left   = data[0 * plane + base];  // ch 0
            top    = data[1 * plane + base];  // ch 1
            right  = data[2 * plane + base];  // ch 2
            bottom = data[3 * plane + base];  // ch 3
          } else {
            // DFL in JS — softmax + weighted sum over 16 bins per edge
            left   = dflSide(data, 0,  H, W, base);  // ch  0..15
            top    = dflSide(data, 16, H, W, base);  // ch 16..31
            right  = dflSide(data, 32, H, W, base);  // ch 32..47
            bottom = dflSide(data, 48, H, W, base);  // ch 48..63
          }

          // ---- Step 1b: dist2bbox — convert edge distances to pixel bbox ----
          // dist2bbox formula (Ultralytics official):
          //   x1 = (anchor_x − left)   × stride
          //   y1 = (anchor_y − top)    × stride
          //   x2 = (anchor_x + right)  × stride
          //   y2 = (anchor_y + bottom) × stride
          const x1 = (ax - left) * stride;    // left edge in 640×640 pixels
          const y1 = (ay - top) * stride;     // top edge in 640×640 pixels
          const x2 = (ax + right) * stride;   // right edge in 640×640 pixels
          const y2 = (ay + bottom) * stride;  // bottom edge in 640×640 pixels

          // ---- Step 1c: class score ----
          const clsCh = useDFLGraph ? 4 : 64;
          const clsRaw = data[clsCh * plane + base];  // raw logit ∈ (-∞, ∞)
          const score = sigmoid(clsRaw);               // → [0, 1]

          // ---- Step 1d: keypoint decode (5 kpts × [x,y,vis]) ----
          const kpts = [];  // Array<[kx, ky, kv]>, length 5
          const kptBaseCh = useDFLGraph ? 5 : 65;
          for (let k = 0; k < 5; k++) {
            const kOff = kptBaseCh + k * 3;  // ch offset for kpt k
            const kxRaw = data[kOff * plane + base];
            const kyRaw = data[(kOff + 1) * plane + base];
            const kvRaw = data[(kOff + 2) * plane + base];

            // Keypoint decode formula (Ultralytics official):
            //   kx = (rawX * 2.0 + (ax - 0.5)) * stride
            //   ky = (rawY * 2.0 + (ay - 0.5)) * stride
            //   kv = sigmoid(rawVis)
            // The raw x,y are displacements from the grid cell corner (gx, gy)
            // (ax - 0.5) = gx gives the cell top-left ∈ [0, W-1]
            const kx = (kxRaw * 2.0 + (ax - 0.5)) * stride;  // 640×640 pixel x
            const ky = (kyRaw * 2.0 + (ay - 0.5)) * stride;  // 640×640 pixel y
            const kv = sigmoid(kvRaw);                        // visibility ∈ [0, 1]

            kpts.push([kx, ky, kv]);
          }

          // ---- Step 1e: confidence filter ----
          // Only keep detections above threshold with non-degenerate boxes
          if (score > confThres && x2 > x1 && y2 > y1) {
            _allBoxes[detCount] = [x1, y1, x2, y2];
            _allScores[detCount] = score;
            _allKpts[detCount] = kpts;
            detCount++;
          }
        }
      }
    }

    // ---- Step 2: NMS (class-agnostic) ----
    const preNmsCount = detCount;
    // keep = indices sorted by descending score, removing overlapping boxes
    const keep = nms(_allBoxes, _allScores, iouThres, detCount).slice(0, MAX_DET);
    // keep: number[], len ≤ MAX_DET (300), sorted by score descending

    // ---- Step 3: gather final detections ----
    const boxes = keep.map(i => _allBoxes[i]);
    const scores = keep.map(i => _allScores[i]);
    const keypoints = keep.map(i => _allKpts[i]);

    return { boxes, scores, keypoints, preNmsCount };
  }

  // ---- IoU & NMS (xyxy format) ----
  /**
   * Compute IoU between one box and many boxes.
   *
   * @param {[number,number,number,number]} box1  single box [x1,y1,x2,y2]
   * @param {Array<[number,number,number,number]>} boxes  N boxes
   * @returns {Float64Array} ious, length = N, each ∈ [0, 1]
   */
  function boxIou(box1, boxes) {
    // box1 = [x1,y1,x2,y2], boxes = [[x1,y1,x2,y2], ...]
    const n = boxes.length;
    const ious = new Float64Array(n);
    const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);  // scalar
    for (let i = 0; i < n; i++) {
      const b = boxes[i];
      const ix1 = Math.max(box1[0], b[0]);
      const iy1 = Math.max(box1[1], b[1]);
      const ix2 = Math.min(box1[2], b[2]);
      const iy2 = Math.min(box1[3], b[3]);
      const iw = ix2 - ix1;
      const ih = iy2 - iy1;
      if (iw <= 0 || ih <= 0) { ious[i] = 0; continue; }
      const inter = iw * ih;                                           // intersection area
      const area2 = (b[2] - b[0]) * (b[3] - b[1]);
      ious[i] = inter / (area1 + area2 - inter + 1e-16);              // 1e-16 avoids NaN
    }
    return ious;
  }

  /**
   * Class-agnostic Non-Maximum Suppression.
   *
   * Algorithm: sort by descending score, then greedily keep the highest-scoring
   * box and suppress all remaining boxes whose IoU with it exceeds the threshold.
   *
   * Shape trace:
   *   boxes:  Array<[x1,y1,x2,y2]>  len N (pre-NMS count, ≤ 8400)
   *   scores: number[]               len N
   *       ↓ sort by score descending
   *   order: number[]                len N, indices into boxes/scores
   *       ↓ greedy IoU suppression
   *   keep: number[]                 len ≤ N, indices sorted by score descending
   *
   * @param {Array<[number,number,number,number]>} boxes  N boxes in xyxy format
   * @param {number[]} scores  N confidence scores
   * @param {number} iouThres  boxes with IoU ≥ this are suppressed
   * @returns {number[]} indices to keep, sorted by descending score
   */
  // Pre-allocated postprocess accumulators (reused across frames)
  let _allBoxes = null;
  let _allScores = null;
  let _allKpts = null;

  // Pre-allocated NMS arrays (reused across frames)
  let _nmsOrder = null;
  let _nmsSuppressed = null;
  let _nmsKeep = null;

  function nms(boxes, scores, iouThres, detCount) {
    var n = detCount !== undefined ? detCount : boxes.length;
    if (n === 0) return [];

    // Pre-allocate or grow arrays as needed
    if (!_nmsOrder || _nmsOrder.length < n) {
      _nmsOrder = new Array(n);
    }
    if (!_nmsSuppressed || _nmsSuppressed.length < n) {
      _nmsSuppressed = new Uint8Array(n);
    }
    if (!_nmsKeep || _nmsKeep.length < n) {
      _nmsKeep = new Array(n);
    }

    // Build sorted order (reuse array)
    for (var i = 0; i < n; i++) _nmsOrder[i] = i;
    _nmsOrder.length = n;
    _nmsOrder.sort(function(a, b) { return scores[b] - scores[a]; });

    // Reset suppressed array
    _nmsSuppressed.fill(0);

    var keepCount = 0;
    for (var i = 0; i < n; i++) {
      var idx = _nmsOrder[i];
      if (_nmsSuppressed[idx]) continue;
      _nmsKeep[keepCount++] = idx;
      var box1 = boxes[idx];
      for (var j = i + 1; j < n; j++) {
        var idx2 = _nmsOrder[j];
        if (_nmsSuppressed[idx2]) continue;
        var b2 = boxes[idx2];
        var ix1 = Math.max(box1[0], b2[0]);
        var iy1 = Math.max(box1[1], b2[1]);
        var ix2 = Math.min(box1[2], b2[2]);
        var iy2 = Math.min(box1[3], b2[3]);
        var iw = ix2 - ix1;
        var ih = iy2 - iy1;
        if (iw <= 0 || ih <= 0) continue;
        var inter = iw * ih;
        var area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        var area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
        var iou = inter / (area1 + area2 - inter + 1e-16);
        if (iou >= iouThres) _nmsSuppressed[idx2] = 1;
      }
    }
    _nmsKeep.length = keepCount;
    return _nmsKeep;
  }

  // ==========================================================================
  // SCALE BACK
  // ==========================================================================

  /**
   * Scale boxes and keypoints from 640×640 model space back to original image.
   *
   * @param {Array<[number,number,number,number]>} boxes  xyxy in model space
   * @param {Array<Array<[number,number,number]>>} keypoints
   * @param {[number,number]} origShape  [origW, origH]
   * @param {number} ratio   scale ratio from preprocessing
   * @param {number} dw      horizontal padding (pixels)
   * @param {number} dh      vertical padding (pixels)
   * @returns {{ boxes, keypoints }} in original image coordinates
   */
  function scaleBack(boxes, keypoints, origShape, ratio, dw, dh) {
    const [origW, origH] = origShape;
    const scaledBoxes = boxes.map(b => {
      const x1 = Math.max(0, Math.min(origW, (b[0] - dw) / ratio));
      const y1 = Math.max(0, Math.min(origH, (b[1] - dh) / ratio));
      const x2 = Math.max(0, Math.min(origW, (b[2] - dw) / ratio));
      const y2 = Math.max(0, Math.min(origH, (b[3] - dh) / ratio));
      return [x1, y1, x2, y2];
    });
    const scaledKpts = keypoints.map(kps =>
      kps.map(([kx, ky, kv]) => {
        const sx = (kx - dw) / ratio;
        const sy = (ky - dh) / ratio;
        return [Math.max(0, Math.min(origW, sx)), Math.max(0, Math.min(origH, sy)), kv];
      })
    );
    return { boxes: scaledBoxes, keypoints: scaledKpts };
  }

  // ==========================================================================
  // CONVERT TO UNIFORM DETECTION FORMAT
  // ==========================================================================

  /**
   * Convert YOLO results to the common detection format used by the face
   * landmark pipeline and drawing functions.
   *
   * Detection format: { data: [[xmin,ymin], [xmax,ymax], [kp0], [kp1], ...], score }
   */
  function toDetections(boxes, scores, keypoints, origShape) {
    const [origW, origH] = origShape;
    return boxes.map((box, i) => {
      const data = [
        [box[0] / origW, box[1] / origH],   // top-left normalized
        [box[2] / origW, box[3] / origH]    // bottom-right normalized
      ];
      // add 5 keypoints (only x,y, normalized — skip visibility for drawing)
      for (const kp of keypoints[i]) {
        data.push([kp[0] / origW, kp[1] / origH]);
      }
      return { data, score: scores[i] };
    });
  }

  // ==========================================================================
  // DRAWING (YOLO-specific — 5 keypoints)
  // ==========================================================================

  function drawDetections(overlayEl, detections) {
    const svgNS = 'http://www.w3.org/2000/svg';
    for (const det of detections) {
      const [xmin, ymin] = det.data[0];
      const [xmax, ymax] = det.data[1];

      // bounding box
      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('class', 'det-rect');
      rect.setAttribute('x', (xmin * 100).toFixed(4));
      rect.setAttribute('y', (ymin * 100).toFixed(4));
      rect.setAttribute('width', ((xmax - xmin) * 100).toFixed(4));
      rect.setAttribute('height', ((ymax - ymin) * 100).toFixed(4));
      overlayEl.appendChild(rect);

      // score label
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('class', 'det-text');
      text.setAttribute('x', (xmin * 100).toFixed(4));
      text.setAttribute('y', (ymin * 100 - 0.6).toFixed(4));
      text.textContent = (det.score * 100).toFixed(1) + '%';
      overlayEl.appendChild(text);

      // 5 keypoints
      const numKeypoints = Math.min(det.data.length - 2, 5);
      for (let k = 0; k < numKeypoints; k++) {
        const kp = det.data[k + 2];
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('class', 'keypoint');
        circle.setAttribute('cx', (kp[0] * 100).toFixed(4));
        circle.setAttribute('cy', (kp[1] * 100).toFixed(4));
        circle.setAttribute('r', '0.5');
        circle.style.fill = KEYPOINT_COLORS[k];
        overlayEl.appendChild(circle);
      }
    }
  }

  // ---- public API ----
  return {
    INPUT_SIZE:  { width: IMGSZ, height: IMGSZ },
    INPUT_LAYOUT: 'nchw',
    IMGSZ,
    STRIDES,
    KEYPOINT_NAMES,
    KEYPOINT_COLORS,
    CONF_THRES,
    IOU_THRES,
    NUM_ANCHORS,
    numKeypoints: 5,
    preprocess,
    showLetterboxPreview,
    drawRawBoxesOnLetterbox,
    postprocess,
    scaleBack,
    toDetections,
    drawDetections,
    // expose for standalone use
    sigmoid,
    makeAnchors,
    dflSide
  };
})();
