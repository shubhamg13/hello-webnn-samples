// ============================================================================
// SSDDetector — preprocessing, postprocessing, and drawing for SSD-based
// face detection models (Detector v1 ONNX, Detector v2 TFLite).
// ============================================================================

const SSDDetector = (() => {

  // ---- constants ----
  const MIN_SCORE = 0.5;
  const MIN_SUPPRESSION_THRESHOLD = 0.3;
  const RAW_SCORE_LIMIT = 80;

  const SSD_OPTIONS_SHORT = {
    num_layers: 4,
    input_size_height: 128,
    input_size_width: 128,
    anchor_offset_x: 0.5,
    anchor_offset_y: 0.5,
    strides: [8, 16, 16, 16],
    interpolated_scale_aspect_ratio: 1.0
  };

  const FaceIndex = {
    LEFT_EYE: 0,
    RIGHT_EYE: 1,
    NOSE_TIP: 2,
    MOUTH: 3,
    LEFT_EYE_TRAGION: 4,
    RIGHT_EYE_TRAGION: 5
  };

  const KEYPOINT_NAMES = ['Left Eye', 'Right Eye', 'Nose Tip', 'Mouth', 'L-Ear Trag', 'R-Ear Trag'];

  // ---- image preprocessing ----
  function preprocessImage(imageElement) {
    const originalWidth = imageElement.naturalWidth;
    const originalHeight = imageElement.naturalHeight;
    const targetWidth = 128;
    const targetHeight = 128;

    const aspectRatio = originalWidth / originalHeight;
    const targetAspectRatio = targetWidth / targetHeight;

    let scale, padX = 0, padY = 0;
    let canvasWidth, canvasHeight;

    if (aspectRatio > targetAspectRatio) {
      scale = targetWidth / originalWidth;
      canvasHeight = Math.round(originalHeight * scale);
      canvasWidth = targetWidth;
      padY = (targetHeight - canvasHeight) / 2 / targetHeight;
    } else {
      scale = targetHeight / originalHeight;
      canvasWidth = Math.round(originalWidth * scale);
      canvasHeight = targetHeight;
      padX = (targetWidth - canvasWidth) / 2 / targetWidth;
    }

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, targetWidth, targetHeight);

    const drawWidth = Math.round(canvasWidth);
    const drawHeight = Math.round(canvasHeight);
    const drawX = Math.round((targetWidth - drawWidth) / 2);
    const drawY = Math.round((targetHeight - drawHeight) / 2);

    ctx.drawImage(imageElement, drawX, drawY, drawWidth, drawHeight);

    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const data = imageData.data;
    const tensor = new Float32Array(targetHeight * targetWidth * 3);

    for (let i = 0; i < targetHeight * targetWidth; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      tensor[i * 3] = r * 2 - 1;
      tensor[i * 3 + 1] = g * 2 - 1;
      tensor[i * 3 + 2] = b * 2 - 1;
    }

    return { tensor, padding: [padX, padY, padX, padY], originalSize: [originalWidth, originalHeight], letterboxCanvas: canvas };
  }

  // ---- debug visualisation ----
  function showLetterboxPreview(canvas, padding) {
    const debugCanvas = document.getElementById('letterboxCanvas');
    const debugCtx = debugCanvas.getContext('2d');
    debugCtx.imageSmoothingEnabled = false;
    debugCtx.drawImage(canvas, 0, 0, 256, 256);
    document.getElementById('debugCard').style.display = '';
    const [padL, padT] = padding;
    document.getElementById('paddingInfo').innerHTML =
      '<strong>Padding (normalized):</strong><br>' +
      'padLeft=' + padL.toFixed(4) + ' (' + (padL * 128).toFixed(1) + 'px)<br>' +
      'padTop=' + padT.toFixed(4) + ' (' + (padT * 128).toFixed(1) + 'px)<br>' +
      'padRight=' + padL.toFixed(4) + '<br>' +
      'padBottom=' + padT.toFixed(4) + '<br>' +
      '<em style="font-size:0.75rem;color:#888;">Bounding box coordinates after NMS but BEFORE letterbox removal are shown below.</em>';
  }

  function drawRawBoxesOnLetterbox(refCanvas, detections, padding) {
    const debugCanvas = document.getElementById('letterboxCanvas');
    const debugCtx = debugCanvas.getContext('2d');
    debugCtx.imageSmoothingEnabled = false;

    debugCtx.clearRect(0, 0, 256, 256);
    debugCtx.drawImage(refCanvas, 0, 0, 256, 256);

    const [padL, padT, padR, padB] = padding;
    debugCtx.strokeStyle = 'rgba(255,0,0,0.5)';
    debugCtx.lineWidth = 1;
    debugCtx.setLineDash([4, 4]);
    debugCtx.strokeRect(padL * 256, padT * 256, (1 - padL - padR) * 256, (1 - padT - padB) * 256);
    debugCtx.setLineDash([]);

    for (const det of detections) {
      const [xmin, ymin] = det.data[0];
      const [xmax, ymax] = det.data[1];
      debugCtx.strokeStyle = '#00FF00';
      debugCtx.lineWidth = 2;
      debugCtx.strokeRect(xmin * 256, ymin * 256, (xmax - xmin) * 256, (ymax - ymin) * 256);
      debugCtx.fillStyle = '#00FF00';
      debugCtx.font = '10px monospace';
      debugCtx.fillText((det.score * 100).toFixed(0) + '%', xmin * 256, ymin * 256 - 2);
    }
  }

  // ---- anchor generation ----
  function generateAnchors(opts) {
    const anchors = [];
    const numLayers = opts.num_layers;
    const strides = opts.strides;
    const inputHeight = opts.input_size_height;
    const inputWidth = opts.input_size_width;
    const anchorOffsetX = opts.anchor_offset_x;
    const anchorOffsetY = opts.anchor_offset_y;
    const interpolatedScaleAspectRatio = opts.interpolated_scale_aspect_ratio;

    let layerId = 0;
    while (layerId < numLayers) {
      let lastSameStrideLayer = layerId;
      let repeats = 0;
      while (lastSameStrideLayer < numLayers &&
             strides[lastSameStrideLayer] === strides[layerId]) {
        lastSameStrideLayer++;
        repeats += interpolatedScaleAspectRatio === 1.0 ? 2 : 1;
      }
      const stride = strides[layerId];
      const featureMapHeight = Math.floor(inputHeight / stride);
      const featureMapWidth = Math.floor(inputWidth / stride);

      for (let y = 0; y < featureMapHeight; y++) {
        const yCenter = (y + anchorOffsetY) / featureMapHeight;
        for (let x = 0; x < featureMapWidth; x++) {
          const xCenter = (x + anchorOffsetX) / featureMapWidth;
          for (let r = 0; r < repeats; r++) {
            anchors.push([xCenter, yCenter]);
          }
        }
      }
      layerId = lastSameStrideLayer;
    }
    return anchors;
  }

  // ---- box decoding ----
  function decodeBoxes(rawBoxes, anchors, inputSize) {
    const scale = inputSize.width;
    const numAnchors = anchors.length;
    const numPoints = rawBoxes.length / numAnchors / 2;
    const boxes = [];

    for (let i = 0; i < numAnchors; i++) {
      const rawBox = [];
      for (let j = 0; j < numPoints * 2; j++) {
        rawBox.push(rawBoxes[i * numPoints * 2 + j]);
      }

      const scaled = rawBox.map(v => v / scale);
      const xCenter = scaled[0] + anchors[i][0];
      const yCenter = scaled[1] + anchors[i][1];
      const w = scaled[2];
      const h = scaled[3];

      for (let k = 2; k < numPoints; k++) {
        scaled[k * 2] += anchors[i][0];
        scaled[k * 2 + 1] += anchors[i][1];
      }

      const xmin = xCenter - w / 2;
      const ymin = yCenter - h / 2;
      const xmax = xCenter + w / 2;
      const ymax = yCenter + h / 2;

      const box = [[xmin, ymin], [xmax, ymax]];
      for (let k = 2; k < numPoints; k++) {
        box.push([scaled[k * 2], scaled[k * 2 + 1]]);
      }
      boxes.push(box);
    }
    return boxes;
  }

  // ---- scoring ----
  function getSigmoidScores(rawScores) {
    const scores = [];
    for (let i = 0; i < rawScores.length; i++) {
      let score = rawScores[i];
      if (score < -RAW_SCORE_LIMIT) score = -RAW_SCORE_LIMIT;
      if (score > RAW_SCORE_LIMIT) score = RAW_SCORE_LIMIT;
      scores.push(1 / (1 + Math.exp(-score)));
    }
    return scores;
  }

  // ---- detection conversion ----
  function convertToDetections(boxes, scores, minScore) {
    const detections = [];
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > minScore) {
        const box = boxes[i];
        if (box[1][0] > box[0][0] && box[1][1] > box[0][1]) {
          detections.push({ data: box, score: scores[i] });
        }
      }
    }
    return detections;
  }

  // ---- IoU & NMS ----
  function computeIoU(axmin, aymin, axmax, aymax, bxmin, bymin, bxmax, bymax) {
    const interXmin = Math.max(axmin, bxmin);
    const interYmin = Math.max(aymin, bymin);
    const interXmax = Math.min(axmax, bxmax);
    const interYmax = Math.min(aymax, bymax);
    if (interXmax < interXmin || interYmax < interYmin) return 0;
    const interArea = (interXmax - interXmin) * (interYmax - interYmin);
    const box1Area = (axmax - axmin) * (aymax - aymin);
    const box2Area = (bxmax - bxmin) * (bymax - bymin);
    return interArea / (box1Area + box2Area - interArea);
  }

  function nonMaxSuppression(detections, minSuppressionThreshold) {
    if (detections.length === 0) return [];
    const sorted = [...detections].sort((a, b) => b.score - a.score);
    const keep = [];
    while (sorted.length > 0) {
      const current = sorted.shift();
      keep.push(current);
      const remaining = [];
      for (const d of sorted) {
        const iou = computeIoU(
          current.data[0][0], current.data[0][1], current.data[1][0], current.data[1][1],
          d.data[0][0], d.data[0][1], d.data[1][0], d.data[1][1]
        );
        if (iou < minSuppressionThreshold) remaining.push(d);
      }
      sorted.length = 0;
      sorted.push(...remaining);
    }
    return keep;
  }

  // ---- letterbox removal ----
  function removeLetterboxPadding(detections, padding) {
    const [padLeft, padTop, padRight, padBottom] = padding;
    const hScale = 1 - (padLeft + padRight);
    const vScale = 1 - (padTop + padBottom);
    return detections.map(d => ({
      data: d.data.map(point => [(point[0] - padLeft) / hScale, (point[1] - padTop) / vScale]),
      score: d.score
    }));
  }

  // ---- drawing ----
  const KEYPOINT_COLORS = ['#FF0000', '#FF0000', '#00AA00', '#0000FF', '#FF8800', '#FF8800'];

  function drawDetections(overlayEl, detections) {
    const svgNS = 'http://www.w3.org/2000/svg';
    for (const det of detections) {
      const [xmin, ymin] = det.data[0];
      const [xmax, ymax] = det.data[1];

      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('class', 'det-rect');
      rect.setAttribute('x', (xmin * 100).toFixed(4));
      rect.setAttribute('y', (ymin * 100).toFixed(4));
      rect.setAttribute('width', ((xmax - xmin) * 100).toFixed(4));
      rect.setAttribute('height', ((ymax - ymin) * 100).toFixed(4));
      overlayEl.appendChild(rect);

      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('class', 'det-text');
      text.setAttribute('x', (xmin * 100).toFixed(4));
      text.setAttribute('y', (ymin * 100 - 0.6).toFixed(4));
      text.textContent = (det.score * 100).toFixed(1) + '%';
      overlayEl.appendChild(text);

      const numKeypoints = det.data.length - 2;
      for (let k = 0; k < numKeypoints; k++) {
        const kp = det.data[k + 2];
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('class', 'keypoint');
        circle.setAttribute('cx', (kp[0] * 100).toFixed(4));
        circle.setAttribute('cy', (kp[1] * 100).toFixed(4));
        circle.setAttribute('r', '0.5');
        circle.style.fill = KEYPOINT_COLORS[k] || '#FF0000';
        overlayEl.appendChild(circle);
      }
    }
  }

  // ---- public API ----
  return {
    INPUT_SIZE: { width: 128, height: 128 },
    INPUT_LAYOUT: 'nhwc',
    SSD_OPTIONS_SHORT,
    FaceIndex,
    KEYPOINT_NAMES,
    KEYPOINT_COLORS,
    MIN_SCORE,
    MIN_SUPPRESSION_THRESHOLD,
    preprocess: preprocessImage,
    showLetterboxPreview,
    drawRawBoxesOnLetterbox,
    generateAnchors,
    decodeBoxes,
    getSigmoidScores,
    convertToDetections,
    computeIoU,
    nonMaxSuppression,
    removeLetterboxPadding,
    drawDetections,
    numKeypoints: 6
  };
})();
