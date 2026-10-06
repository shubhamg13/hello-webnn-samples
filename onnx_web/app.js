const MIN_SCORE = 0.5;
const MIN_SUPPRESSION_THRESHOLD = 0.3;
const RAW_SCORE_LIMIT = 80;
const NUM_LANDMARKS = 468;
const NUM_DIMS = 3;
const ROI_SCALE = 1.5;
const LANDMARK_THRESHOLD = 0.5;
const DETECTION_INPUT_SIZE = { width: 128, height: 128 };
const LANDMARK_INPUT_SIZE = { width: 192, height: 192 };

const FaceIndex = {
  LEFT_EYE: 0, RIGHT_EYE: 1, NOSE_TIP: 2,
  MOUTH: 3, LEFT_EYE_TRAGION: 4, RIGHT_EYE_TRAGION: 5
};

const FACE_LANDMARK_CONNECTIONS = [
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
  [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40],
  [40, 39], [39, 37], [37, 0], [0, 267], [267, 269],
  [269, 270], [270, 409], [409, 291], [78, 95], [95, 88], [88, 178],
  [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324],
  [324, 308], [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312],
  [312, 311], [311, 310], [310, 415], [415, 308],
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154],
  [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159],
  [159, 158], [158, 157], [157, 173], [173, 133],
  [46, 53], [53, 52], [52, 65], [65, 55], [70, 63], [63, 105], [105, 66],
  [66, 107],
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381],
  [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386],
  [386, 385], [385, 384], [384, 398], [398, 362],
  [276, 283], [283, 282], [282, 295], [295, 285], [300, 293], [293, 334],
  [334, 296], [296, 336],
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
  [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
  [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
  [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
  [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
  [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]
];

const SSD_OPTIONS = {
  num_layers: 4,
  input_size_height: 128,
  input_size_width: 128,
  anchor_offset_x: 0.5,
  anchor_offset_y: 0.5,
  strides: [8, 16, 16, 16],
  interpolated_scale_aspect_ratio: 1.0
};

// ─── Anchor generation ────────────────────────────────────────

function generateAnchors(opts) {
  const anchors = [];
  let layerId = 0;
  while (layerId < opts.num_layers) {
    let lastSameStrideLayer = layerId;
    let repeats = 0;
    while (lastSameStrideLayer < opts.num_layers &&
           opts.strides[lastSameStrideLayer] === opts.strides[layerId]) {
      lastSameStrideLayer++;
      repeats += opts.interpolated_scale_aspect_ratio === 1.0 ? 2 : 1;
    }
    const stride = opts.strides[layerId];
    const fh = Math.floor(opts.input_size_height / stride);
    const fw = Math.floor(opts.input_size_width / stride);
    for (let y = 0; y < fh; y++) {
      for (let x = 0; x < fw; x++) {
        for (let r = 0; r < repeats; r++) {
          anchors.push([
            (x + opts.anchor_offset_x) / fw,
            (y + opts.anchor_offset_y) / fh
          ]);
        }
      }
    }
    layerId = lastSameStrideLayer;
  }
  return anchors;
}

// ─── Box decoding ─────────────────────────────────────────────

function decodeBoxes(rawBoxes, anchors) {
  const scale = DETECTION_INPUT_SIZE.width;
  const numAnchors = anchors.length;
  const boxes = [];
  const totalValues = rawBoxes.length;

  for (let i = 0; i < numAnchors; i++) {
    const offset = i * 16;
    if (offset + 16 > totalValues) break;
    const box = [];
    for (let j = 0; j < 16; j++) {
      box.push(rawBoxes[offset + j]);
    }
    const scaled = box.map(v => v / scale);

    const xCenter = scaled[0] + anchors[i][0];
    const yCenter = scaled[1] + anchors[i][1];
    const w = scaled[2];
    const h = scaled[3];

    for (let k = 2; k < 8; k++) {
      scaled[k * 2] += anchors[i][0];
      scaled[k * 2 + 1] += anchors[i][1];
    }

    const decoded = [
      [xCenter - w / 2, yCenter - h / 2],
      [xCenter + w / 2, yCenter + h / 2]
    ];
    for (let k = 2; k < 8; k++) {
      decoded.push([scaled[k * 2], scaled[k * 2 + 1]]);
    }
    boxes.push(decoded);
  }
  return boxes;
}

// ─── Math helpers ─────────────────────────────────────────────

function sigmoid(x) {
  if (x < -RAW_SCORE_LIMIT) x = -RAW_SCORE_LIMIT;
  if (x > RAW_SCORE_LIMIT) x = RAW_SCORE_LIMIT;
  return 1 / (1 + Math.exp(-x));
}

function computeIoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) {
  const ix = Math.max(xmin1, xmin2);
  const iy = Math.max(ymin1, ymin2);
  const ixx = Math.min(xmax1, xmax2);
  const iyy = Math.min(ymax1, ymax2);
  if (ixx < ix || iyy < iy) return 0;
  const iArea = (ixx - ix) * (iyy - iy);
  const uArea = (xmax1 - xmin1) * (ymax1 - ymin1) +
                (xmax2 - xmin2) * (ymax2 - ymin2) - iArea;
  return iArea / uArea;
}

function nonMaxSuppression(detections, iouThreshold, minScore) {
  const filtered = detections.filter(d => d.score >= minScore);
  if (!filtered.length) return [];
  const sorted = [...filtered].sort((a, b) => b.score - a.score);
  const keep = [];
  while (sorted.length) {
    const curr = sorted.shift();
    keep.push(curr);
    const remaining = [];
    for (const det of sorted) {
      if (computeIoU(
        curr.data[0][0], curr.data[0][1], curr.data[1][0], curr.data[1][1],
        det.data[0][0], det.data[0][1], det.data[1][0], det.data[1][1]
      ) < iouThreshold) {
        remaining.push(det);
      }
    }
    sorted.length = 0;
    sorted.push(...remaining);
  }
  return keep;
}

// ─── Preprocessing ────────────────────────────────────────────

function preprocessDetection(imageEl) {
  const tw = DETECTION_INPUT_SIZE.width;
  const th = DETECTION_INPUT_SIZE.height;
  const ow = imageEl.naturalWidth;
  const oh = imageEl.naturalHeight;

  const ar = ow / oh;
  const tar = tw / th;
  let canvasW, canvasH, padX = 0, padY = 0;

  if (ar > tar) {
    const s = tw / ow;
    canvasW = tw;
    canvasH = Math.round(oh * s);
    padY = (th - canvasH) / 2 / th;
  } else {
    const s = th / oh;
    canvasH = th;
    canvasW = Math.round(ow * s);
    padX = (tw - canvasW) / 2 / tw;
  }

  const canvas = document.createElement('canvas');
  canvas.width = tw;
  canvas.height = th;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, tw, th);
  ctx.drawImage(imageEl,
    Math.round((tw - canvasW) / 2),
    Math.round((th - canvasH) / 2),
    Math.round(canvasW), Math.round(canvasH)
  );

  const data = ctx.getImageData(0, 0, tw, th).data;
  const tensor = new Float32Array(tw * th * 3);
  for (let i = 0; i < tw * th; i++) {
    tensor[i * 3] = data[i * 4] / 255 * 2 - 1;
    tensor[i * 3 + 1] = data[i * 4 + 1] / 255 * 2 - 1;
    tensor[i * 3 + 2] = data[i * 4 + 2] / 255 * 2 - 1;
  }
  return { tensor, padding: [padX, padY, padX, padY] };
}

function removePadding(detections, padding) {
  if (!detections || !detections.length) return [];
  const [pl, pt, pr, pb] = padding;
  const hScale = 1 - (pl + pr);
  const vScale = 1 - (pt + pb);
  return detections.map(d => ({
    data: d.data.map(p => [
      (p[0] - pl) / hScale,
      (p[1] - pt) / vScale
    ]),
    score: d.score
  }));
}

// ─── ROI computation ──────────────────────────────────────────

function computeROI(detection, imgWidth, imgHeight) {
  const [xmin, ymin] = detection.data[0];
  const [xmax, ymax] = detection.data[1];
  const cx = (xmin + xmax) / 2;
  const cy = (ymin + ymax) / 2;

  const leftEye = detection.data[FaceIndex.LEFT_EYE + 2];
  const rightEye = detection.data[FaceIndex.RIGHT_EYE + 2];
  const dx = rightEye[0] - leftEye[0];
  const dy = leftEye[1] - rightEye[1];
  const angle = -Math.atan2(dy, dx);
  const PI = Math.PI;
  const rotation = angle - 2 * PI * Math.floor((angle + PI) / (2 * PI));

  const absW = (xmax - xmin) * imgWidth;
  const absH = (ymax - ymin) * imgHeight;
  const longSide = Math.max(absW, absH);

  return {
    xCenter: cx,
    yCenter: cy,
    width: (longSide / imgWidth) * ROI_SCALE,
    height: (longSide / imgHeight) * ROI_SCALE,
    rotation
  };
}

function cropToROI(imageEl, roi, outputSize) {
  const canvas = document.createElement('canvas');
  canvas.width = outputSize.width;
  canvas.height = outputSize.height;
  const ctx = canvas.getContext('2d');

  const imgW = imageEl.naturalWidth;
  const imgH = imageEl.naturalHeight;
  const rw = roi.width * imgW;
  const rh = roi.height * imgH;
  const rx = (roi.xCenter - roi.width / 2) * imgW;
  const ry = (roi.yCenter - roi.height / 2) * imgH;

  ctx.translate(outputSize.width / 2, outputSize.height / 2);
  ctx.rotate(roi.rotation);
  ctx.drawImage(imageEl, rx, ry, rw, rh,
    -outputSize.width / 2, -outputSize.height / 2,
    outputSize.width, outputSize.height
  );
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const data = ctx.getImageData(0, 0, outputSize.width, outputSize.height).data;
  const tensor = new Float32Array(outputSize.width * outputSize.height * 3);
  for (let i = 0; i < outputSize.width * outputSize.height; i++) {
    tensor[i * 3] = data[i * 4] / 255;
    tensor[i * 3 + 1] = data[i * 4 + 1] / 255;
    tensor[i * 3 + 2] = data[i * 4 + 2] / 255;
  }
  return { tensor, canvas };
}

function projectLandmarks(landmarks, roi, imgWidth, imgHeight) {
  const cos = Math.cos(roi.rotation);
  const sin = Math.sin(roi.rotation);
  const iw = LANDMARK_INPUT_SIZE.width;
  const ih = LANDMARK_INPUT_SIZE.height;

  return landmarks.map(lm => {
    let x = lm.x / iw;
    let y = lm.y / ih;
    x -= 0.5;
    y -= 0.5;
    const rx = x * cos - y * sin;
    const ry = x * sin + y * cos;
    return {
      x: (rx * roi.width + roi.xCenter),
      y: (ry * roi.height + roi.yCenter),
      z: lm.z / iw
    };
  });
}

// ─── Drawing ──────────────────────────────────────────────────

function drawResults(ctx, imageEl, detections, allLandmarks, showLandmarks) {
  const iw = imageEl.naturalWidth;
  const ih = imageEl.naturalHeight;
  const sx = ctx.canvas.width / iw;
  const sy = ctx.canvas.height / ih;

  for (const det of detections) {
    const [xmin, ymin] = det.data[0];
    const [xmax, ymax] = det.data[1];
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin * iw * sx, ymin * ih * sy,
      (xmax - xmin) * iw * sx, (ymax - ymin) * ih * sy);
    ctx.fillStyle = '#00FF00';
    ctx.font = '16px Arial';
    ctx.fillText(`${(det.score * 100).toFixed(1)}%`,
      xmin * iw * sx, (ymin * ih - 5) * sy);
  }

  if (!showLandmarks) return;

  ctx.strokeStyle = '#FFFF00';
  ctx.lineWidth = 0.8;
  for (const landmarks of allLandmarks) {
    for (const [start, end] of FACE_LANDMARK_CONNECTIONS) {
      if (start < landmarks.length && end < landmarks.length) {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(p1.x * iw * sx, p1.y * ih * sy);
        ctx.lineTo(p2.x * iw * sx, p2.y * ih * sy);
        ctx.stroke();
      }
    }
    ctx.fillStyle = '#00FFFF';
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * iw * sx, lm.y * ih * sy, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ─── Main pipeline ────────────────────────────────────────────

async function main() {
  const statusEl = document.getElementById('status');
  const imageEl = document.getElementById('image');
  const overlayEl = document.getElementById('overlay');
  const roiCanvas = document.getElementById('roiCanvas');
  const runBtn = document.getElementById('runBtn');
  const backendSelect = document.getElementById('backendSelect');
  const landmarksCheck = document.getElementById('landmarksCheck');
  const debugRoiCheck = document.getElementById('debugRoiCheck');

  const anchors = generateAnchors(SSD_OPTIONS);

  async function run() {
    try {
      statusEl.textContent = 'Loading ONNX models...';
      runBtn.disabled = true;

      const [detectorBuffer, landmarkBuffer] = await Promise.all([
        fetch('../assets/face_detector.onnx').then(r => r.arrayBuffer()),
        fetch('../assets/face_landmark.onnx').then(r => r.arrayBuffer())
      ]);

      statusEl.textContent = 'Creating inference sessions...';
      const backend = backendSelect.value;
      const ep = backend === 'wasm' ? ['wasm'] : [{ name: backend }];
      const detector = await ort.InferenceSession.create(detectorBuffer, { executionProviders: ep });
      const landmarkSession = await ort.InferenceSession.create(landmarkBuffer, { executionProviders: ep });

      await new Promise(r => imageEl.complete ? r() : imageEl.onload = r);

      const iw = imageEl.naturalWidth;
      const ih = imageEl.naturalHeight;
      overlayEl.width = imageEl.width;
      overlayEl.height = imageEl.height;

      // ── Face Detection ──
      statusEl.textContent = 'Detecting faces...';
      const { tensor, padding } = preprocessDetection(imageEl);
      const inputName = detector.inputNames[0];
      const detInput = new ort.Tensor('float32', tensor,
        [1, DETECTION_INPUT_SIZE.height, DETECTION_INPUT_SIZE.width, 3]);
      const detFeeds = { [inputName]: detInput };
      const detOut = await detector.run(detFeeds);

      const detKeys = Object.keys(detOut);
      const regKey = detKeys.find(k => k.includes('regress')) || detKeys[0];
      const classKey = detKeys.find(k => k.includes('class')) || detKeys[1];
      const rawBoxes = detOut[regKey].data;
      const rawScores = detOut[classKey].data;

      const decoded = decodeBoxes(rawBoxes, anchors);
      const scores = new Float32Array(rawScores.length);
      for (let i = 0; i < rawScores.length; i++) scores[i] = sigmoid(rawScores[i]);

      let detections = [];
      for (let i = 0; i < scores.length; i++) {
        if (scores[i] > MIN_SCORE) {
          const box = decoded[i];
          if (box[1][0] > box[0][0] && box[1][1] > box[0][1]) {
            detections.push({ data: box, score: scores[i] });
          }
        }
      }
      detections = nonMaxSuppression(detections, MIN_SUPPRESSION_THRESHOLD, MIN_SCORE);
      detections = removePadding(detections, padding);

      statusEl.textContent = `Found ${detections.length} face(s)`;

      let allLandmarks = [];
      const useLandmarks = landmarksCheck.checked;

      // ── Face Landmark ──
      if (useLandmarks && detections.length > 0) {
        statusEl.textContent = 'Detecting landmarks...';
        for (const det of detections) {
          const roi = computeROI(det, iw, ih);
          const { tensor: lmTensor, canvas: roiCvs } = cropToROI(imageEl, roi, LANDMARK_INPUT_SIZE);

          const lmInputName = landmarkSession.inputNames[0];
          const lmInput = new ort.Tensor('float32', lmTensor,
            [1, LANDMARK_INPUT_SIZE.height, LANDMARK_INPUT_SIZE.width, 3]);
          const lmFeeds = { [lmInputName]: lmInput };
          const lmOut = await landmarkSession.run(lmFeeds);

          const lmKeys = Object.keys(lmOut);
          const rawLandmarks = lmOut[lmKeys[0]].data;
          const rawFlag = lmOut[lmKeys[1]].data[0];

          if (sigmoid(rawFlag) > LANDMARK_THRESHOLD) {
            const landmarks = [];
            for (let i = 0; i < NUM_LANDMARKS; i++) {
              landmarks.push({
                x: rawLandmarks[i * 3],
                y: rawLandmarks[i * 3 + 1],
                z: rawLandmarks[i * 3 + 2]
              });
            }
            allLandmarks.push(projectLandmarks(landmarks, roi, iw, ih));

            // Debug ROI canvas
            if (debugRoiCheck.checked) {
              roiCanvas.style.display = 'block';
              roiCanvas.width = LANDMARK_INPUT_SIZE.width;
              roiCanvas.height = LANDMARK_INPUT_SIZE.height;
              const rctx = roiCanvas.getContext('2d');
              rctx.drawImage(roiCvs, 0, 0);
              rctx.fillStyle = '#00FF00';
              for (let i = 0; i < NUM_LANDMARKS; i++) {
                rctx.beginPath();
                rctx.arc(rawLandmarks[i * 3], rawLandmarks[i * 3 + 1], 2, 0, Math.PI * 2);
                rctx.fill();
              }
            }
          }
        }
      }

      // ── Render ──
      const ctx = overlayEl.getContext('2d');
      ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);
      drawResults(ctx, imageEl, detections, allLandmarks, useLandmarks);

      statusEl.textContent =
        `Found ${detections.length} face(s) using ONNX (${backend.toUpperCase()})` +
        (useLandmarks && allLandmarks.length ? `, ${NUM_LANDMARKS * allLandmarks.length} landmarks` : '');

      if (!useLandmarks && debugRoiCheck.checked) {
        roiCanvas.style.display = 'none';
      }

    } catch (err) {
      console.error(err);
      console.error(err.stack);
      statusEl.textContent = 'Error: ' + err.message + ' (see console for details)';
    } finally {
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener('click', run);

  document.getElementById('clearBtn').addEventListener('click', () => {
    const ctx = overlayEl.getContext('2d');
    ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);
    roiCanvas.style.display = 'none';
    statusEl.textContent = 'Ready';
  });
}

window.addEventListener('load', main);
