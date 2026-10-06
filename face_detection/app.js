const MIN_SCORE = 0.5;
const MIN_SUPPRESSION_THRESHOLD = 0.3;
const RAW_SCORE_LIMIT = 80;

const NUM_LANDMARKS = 468;
const NUM_DIMS = 3;
const ROI_SCALE = 1.5;
const LANDMARK_DETECTION_THRESHOLD = 0.5;
const LANDMARK_INPUT_SIZE = { width: 192, height: 192 };

const FaceIndex = {
  LEFT_EYE: 0,
  RIGHT_EYE: 1,
  NOSE_TIP: 2,
  MOUTH: 3,
  LEFT_EYE_TRAGION: 4,
  RIGHT_EYE_TRAGION: 5
};

const FACE_LANDMARK_CONNECTIONS = [
  // Lips
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
  [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40],
  [40, 39], [39, 37], [37, 0], [0, 267], [267, 269],
  [269, 270], [270, 409], [409, 291], [78, 95], [95, 88], [88, 178],
  [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324],
  [324, 308], [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312],
  [312, 311], [311, 310], [310, 415], [415, 308],
  // Left eye
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154],
  [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159],
  [159, 158], [158, 157], [157, 173], [173, 133],
  // Left eyebrow
  [46, 53], [53, 52], [52, 65], [65, 55], [70, 63], [63, 105], [105, 66],
  [66, 107],
  // Right eye
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381],
  [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386],
  [386, 385], [385, 384], [384, 398], [398, 362],
  // Right eyebrow
  [276, 283], [283, 282], [282, 295], [295, 285], [300, 293], [293, 334],
  [334, 296], [296, 336],
  // Face oval
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
  [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
  [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
  [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
  [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
  [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]
];

const SSD_OPTIONS_SHORT = {
  num_layers: 4,
  input_size_height: 128,
  input_size_width: 128,
  anchor_offset_x: 0.5,
  anchor_offset_y: 0.5,
  strides: [8, 16, 16, 16],
  interpolated_scale_aspect_ratio: 1.0
};

class FaceDetector {
  constructor(modelType, backend) {
    this.modelType = modelType;
    this.backend = backend;
    this.model = null;
    this.anchors = generateAnchors(SSD_OPTIONS_SHORT);
    this.inputSize = { width: 128, height: 128 };
  }

  async load() {
    if (this.modelType === 'tfjs') {
      await tf.setBackend(this.backend);
      await tf.ready();
      console.log('TFJS backend:', tf.getBackend());
      this.model = await tf.loadGraphModel('../assets/face-detection-tfjs-short-v1/model.json');
    } else if (this.modelType === 'tflite') {
      this.model = await tflite.loadTFLiteModel('../assets/face_detector_ref.tflite');
      console.log('Using face_detector_ref.tflite');
    }
    console.log(this.modelType.toUpperCase() + ' model loaded successfully');
  }

  async detect(imageElement) {
    const { tensor, padding, originalSize } = this.preprocessImage(imageElement);
    
    const inputTensor = tf.tensor4d(tensor, [1, this.inputSize.height, this.inputSize.width, 3]);
    const outputs = this.model.predict(inputTensor);

    let rawBoxes, rawScores;
    
    if (this.modelType === 'tflite') {
      rawBoxes = await outputs.regressors.data();
      rawScores = await outputs.classificators.data();
    } else if (this.modelType === 'tfjs') {
      const rawData = await outputs.data();
      const numAnchors = this.anchors.length;
      const numFeatures = rawData.length / numAnchors;

      rawBoxes = [];
      rawScores = [];
      for (let i = 0; i < numAnchors; i++) {
        rawScores.push(rawData[i * numFeatures + 0]);
        for (let j = 1; j < 17; j++) {
          rawBoxes.push(rawData[i * numFeatures + j]);
        }
      }
    }
    
    inputTensor.dispose();
    if (Array.isArray(outputs)) {
      outputs.forEach(o => o.dispose());
    } else if (outputs && outputs.dispose) {
      outputs.dispose();
    }

    const boxes = this.decodeBoxes(rawBoxes);
    const scores = this.getSigmoidScores(rawScores);
    let detections = this.convertToDetections(boxes, scores);
    detections = nonMaxSuppression(detections, MIN_SUPPRESSION_THRESHOLD, MIN_SCORE);
    detections = this.removeLetterboxPadding(detections, padding);
    
    return { detections, originalSize };
  }

  preprocessImage(imageElement) {
    const originalWidth = imageElement.naturalWidth;
    const originalHeight = imageElement.naturalHeight;
    const originalSize = [originalWidth, originalHeight];

    const targetWidth = this.inputSize.width;
    const targetHeight = this.inputSize.height;

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

    const padding = [padX, padY, padX, padY];

    return { tensor, padding, originalSize };
  }

  decodeBoxes(rawBoxes) {
    const scale = this.inputSize.width;
    const numAnchors = this.anchors.length;
    const numPoints = rawBoxes.length / numAnchors / 2;
    const boxes = [];

    for (let i = 0; i < numAnchors; i++) {
      const rawBox = [];
      for (let j = 0; j < numPoints * 2; j++) {
        rawBox.push(rawBoxes[i * numPoints * 2 + j]);
      }

      const scaled = rawBox.map(v => v / scale);
      
      const xCenter = scaled[0] + this.anchors[i][0];
      const yCenter = scaled[1] + this.anchors[i][1];
      const w = scaled[2];
      const h = scaled[3];

      for (let k = 2; k < numPoints; k++) {
        scaled[k * 2] += this.anchors[i][0];
        scaled[k * 2 + 1] += this.anchors[i][1];
      }

      const xmin = xCenter - w / 2;
      const ymin = yCenter - h / 2;
      const xmax = xCenter + w / 2;
      const ymax = yCenter + h / 2;

      const box = [
        [xmin, ymin],
        [xmax, ymax]
      ];

      for (let k = 2; k < numPoints; k++) {
        box.push([scaled[k * 2], scaled[k * 2 + 1]]);
      }

      boxes.push(box);
    }

    return boxes;
  }

  getSigmoidScores(rawScores) {
    const scores = [];
    for (let i = 0; i < rawScores.length; i++) {
      let score = rawScores[i];
      if (score < -RAW_SCORE_LIMIT) score = -RAW_SCORE_LIMIT;
      if (score > RAW_SCORE_LIMIT) score = RAW_SCORE_LIMIT;
      scores.push(1 / (1 + Math.exp(-score)));
    }
    return scores;
  }

  convertToDetections(boxes, scores) {
    const detections = [];
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > MIN_SCORE) {
        const box = boxes[i];
        if (box[1][0] > box[0][0] && box[1][1] > box[0][1]) {
          detections.push({
            data: box,
            score: scores[i]
          });
        }
      }
    }
    return detections;
  }

  removeLetterboxPadding(detections, padding) {
    const [padLeft, padTop, padRight, padBottom] = padding;
    const hScale = 1 - (padLeft + padRight);
    const vScale = 1 - (padTop + padBottom);

    return detections.map(detection => {
      const adjusted = detection.data.map(point => {
        const x = (point[0] - padLeft) / hScale;
        const y = (point[1] - padTop) / vScale;
        return [x, y];
      });
      return { data: adjusted, score: detection.score };
    });
  }
}

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

function nonMaxSuppression(detections, minSuppressionThreshold, minScore) {
  if (detections.length === 0) return [];

  const sortedDetections = [...detections].sort((a, b) => b.score - a.score);
  const keep = [];

  while (sortedDetections.length > 0) {
    const current = sortedDetections.shift();
    keep.push(current);

    const remaining = [];
    for (const detection of sortedDetections) {
      const iou = computeIoU(current.data[0], current.data[1], detection.data[0], detection.data[1]);
      if (iou < minSuppressionThreshold) {
        remaining.push(detection);
      }
    }
    sortedDetections.length = 0;
    sortedDetections.push(...remaining);
  }

  return keep;
}

function computeIoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) {
  const interXmin = Math.max(xmin1, xmin2);
  const interYmin = Math.max(ymin1, ymin2);
  const interXmax = Math.min(xmax1, xmax2);
  const interYmax = Math.min(ymax1, ymax2);

  if (interXmax < interXmin || interYmax < interYmin) {
    return 0;
  }

  const interArea = (interXmax - interXmin) * (interYmax - interYmin);
  const box1Area = (xmax1 - xmin1) * (ymax1 - ymin1);
  const box2Area = (xmax2 - xmin2) * (ymax2 - ymin2);
  const unionArea = box1Area + box2Area - interArea;

  return interArea / unionArea;
}

class FaceLandmarkDetector {
  constructor(modelType, backend) {
    this.modelType = modelType;
    this.backend = backend;
    this.model = null;
    this.inputSize = LANDMARK_INPUT_SIZE;
  }

  async load() {
    if (this.modelType === 'tfjs') {
      await tf.setBackend(this.backend);
      await tf.ready();
      this.model = await tf.loadGraphModel('../assets/facemesh-tfjs-default-v1/model.json');
    } else if (this.modelType === 'tflite') {
      this.model = await tflite.loadTFLiteModel('../assets/face_landmark_ref.tflite');
    }
    console.log('FaceLandmark ' + this.modelType.toUpperCase() + ' model loaded');
  }

  async detect(imageElement, detections) {
    if (!detections || detections.length === 0) {
      return [];
    }

    const allLandmarks = [];

    for (const detection of detections) {
      const roi = this.computeROI(detection, [imageElement.naturalWidth, imageElement.naturalHeight]);
      const tensor = this.cropImageToROI(imageElement, roi, this.inputSize);
      
      const inputTensor = tf.tensor4d(tensor, [1, this.inputSize.height, this.inputSize.width, 3]);
      const outputs = this.model.predict(inputTensor);

      let rawLandmarks, rawFaceFlag;
      
      if (this.modelType === 'tflite') {
        if (typeof outputs === 'object' && !Array.isArray(outputs)) {
          const keys = Object.keys(outputs);
          rawLandmarks = await outputs[keys[0]].data();
          rawFaceFlag = await outputs[keys[1]].data();
        } else if (Array.isArray(outputs)) {
          rawLandmarks = await outputs[0].data();
          rawFaceFlag = await outputs[1].data();
        }
      } else if (this.modelType === 'tfjs') {
        // TFJS: outputs is an array of 3 tensors
        // outputs[0]: [1, 266] - auxiliary (ignore)
        // outputs[1]: [1, 1] - face flag
        // outputs[2]: [1, 1404] - landmarks
        rawLandmarks = await outputs[2].data();
        rawFaceFlag = (await outputs[1].data())[0];
      }

      inputTensor.dispose();
      if (Array.isArray(outputs)) {
        outputs.forEach(o => o.dispose());
      } else if (outputs && outputs.dispose) {
        outputs.dispose();
      }

      const faceFlag = this.sigmoid(rawFaceFlag);
      
      if (faceFlag > LANDMARK_DETECTION_THRESHOLD) {
        const landmarks = [];
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          landmarks.push({
            x: rawLandmarks[i * 3],
            y: rawLandmarks[i * 3 + 1],
            z: rawLandmarks[i * 3 + 2]
          });
        }
        
        const projected = this.projectLandmarks(landmarks, roi, [imageElement.naturalWidth, imageElement.naturalHeight]);
        
        allLandmarks.push(projected);
      }
    }

    return allLandmarks;
  }

  computeROI(detection, imageSize) {
    const [imgWidth, imgHeight] = imageSize;
    const [xmin, ymin] = detection.data[0];
    const [xmax, ymax] = detection.data[1];
    
    const width = xmax - xmin;
    const height = ymax - ymin;
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;
    
    const leftEye = detection.data[FaceIndex.LEFT_EYE + 2];
    const rightEye = detection.data[FaceIndex.RIGHT_EYE + 2];
    
    const dx = rightEye[0] - leftEye[0];
    const dy = leftEye[1] - rightEye[1];
    const angle = -Math.atan2(dy, dx);
    
    // Normalize to [0, 2*PI] like Python
    const PI = Math.PI;
    const TWO_PI = 2 * PI;
    const rotation = angle - TWO_PI * Math.floor((angle + PI) / TWO_PI);
    
    // Python: long_size = max(bbox_width_pixels, bbox_height_pixels)
    //          normalized_width = long_size / image_width
    //          normalized_height = long_size / image_height
    //          Then scales each by ROI_SCALE (1.5, 1.5) tuple
    const absWidth = width * imgWidth;
    const absHeight = height * imgHeight;
    const longSide = Math.max(absWidth, absHeight);
    
    const roiWidth = (longSide / imgWidth) * ROI_SCALE;
    const roiHeight = (longSide / imgHeight) * ROI_SCALE;
    
    return {
      xCenter: cx,
      yCenter: cy,
      width: roiWidth,
      height: roiHeight,
      rotation: rotation,
      normalized: true
    };
  }

  cropImageToROI(imageElement, roi, outputSize) {
    const canvas = document.createElement('canvas');
    canvas.width = outputSize.width;
    canvas.height = outputSize.height;
    const ctx = canvas.getContext('2d');
    
    const imgW = imageElement.naturalWidth;
    const imgH = imageElement.naturalHeight;
    
    const roiW = roi.width * imgW;
    const roiH = roi.height * imgH;
    const roiX = (roi.xCenter - roi.width / 2) * imgW;
    const roiY = (roi.yCenter - roi.height / 2) * imgH;
    
    ctx.translate(outputSize.width / 2, outputSize.height / 2);
    ctx.rotate(roi.rotation);
    ctx.drawImage(
      imageElement,
      roiX, roiY, roiW, roiH,
      -outputSize.width / 2, -outputSize.height / 2,
      outputSize.width, outputSize.height
    );
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    // Draw to debug canvas
    const debugCanvas = document.getElementById('roiCanvas');
    if (debugCanvas) {
      const debugCtx = debugCanvas.getContext('2d');
      debugCanvas.width = outputSize.width;
      debugCanvas.height = outputSize.height;
      debugCtx.drawImage(canvas, 0, 0);
    }
    
    const imageData = ctx.getImageData(0, 0, outputSize.width, outputSize.height);
    const data = imageData.data;
    const tensor = new Float32Array(outputSize.height * outputSize.width * 3);
    
    for (let i = 0; i < outputSize.height * outputSize.width; i++) {
      tensor[i * 3] = data[i * 4] / 255;
      tensor[i * 3 + 1] = data[i * 4 + 1] / 255;
      tensor[i * 3 + 2] = data[i * 4 + 2] / 255;
    }
    
    return tensor;
  }

  projectLandmarks(landmarks, roi, originalSize) {
    const cos = Math.cos(roi.rotation);
    const sin = Math.sin(roi.rotation);
    
    return landmarks.map(lm => {
      // Normalize from tensor space to 0-1
      let x = lm.x / this.inputSize.width;
      let y = lm.y / this.inputSize.height;
      let z = lm.z / this.inputSize.width;
      
      // Center (subtract 0.5)
      x -= 0.5;
      y -= 0.5;
      
      // Rotate around center (0,0 in centered space)
      const rx = x * cos - y * sin;
      const ry = x * sin + y * cos;
      x = rx;
      y = ry;
      
      // Scale by ROI dimensions (normalized coordinates!)
      x *= roi.width;
      y *= roi.height;
      
      // Translate to ROI center
      x += roi.xCenter;
      y += roi.yCenter;
      
      return { x, y, z };
    });
  }

  sigmoid(x) {
    if (x < -RAW_SCORE_LIMIT) x = -RAW_SCORE_LIMIT;
    if (x > RAW_SCORE_LIMIT) x = RAW_SCORE_LIMIT;
    return 1 / (1 + Math.exp(-x));
  }

  drawLandmarksOnROI(landmarks) {
    const debugCanvas = document.getElementById('roiCanvas');
    if (!debugCanvas) return;
    
    const ctx = debugCanvas.getContext('2d');
    
    // Landmarks are normalized [0,1] in ROI space, scale to ROI dimensions (192x192)
    const roiW = this.inputSize.width;
    const roiH = this.inputSize.height;
    
    // Draw landmarks (normalized [0,1] to pixel coords in ROI space)
    ctx.fillStyle = '#00FF00';
    for (const lm of landmarks) {
      const lmx = lm.x * roiW;
      const lmy = lm.y * roiH;
      ctx.beginPath();
      ctx.arc(lmx, lmy, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw connections
    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 1;
    for (const [start, end] of FACE_LANDMARK_CONNECTIONS) {
      if (start < landmarks.length && end < landmarks.length) {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(p1.x * roiW, p1.y * roiH);
        ctx.lineTo(p2.x * roiW, p2.y * roiH);
        ctx.stroke();
      }
    }
  }

  drawRawLandmarksOnROI(rawLandmarks) {
    const debugCanvas = document.getElementById('roiCanvas');
    if (!debugCanvas) return;
    
    const ctx = debugCanvas.getContext('2d');
    const w = debugCanvas.width;
    const h = debugCanvas.height;
    
    // Draw landmarks (raw pixel coordinates [0, 192])
    // Scale to match canvas size
    ctx.fillStyle = '#00FF00';
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const x = rawLandmarks[i * 3];
      const y = rawLandmarks[i * 3 + 1];
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw connections
    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 1;
    for (const [start, end] of FACE_LANDMARK_CONNECTIONS) {
      if (start < NUM_LANDMARKS && end < NUM_LANDMARKS) {
        const x1 = rawLandmarks[start * 3];
        const y1 = rawLandmarks[start * 3 + 1];
        const x2 = rawLandmarks[end * 3];
        const y2 = rawLandmarks[end * 3 + 1];
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  }
}

async function main() {
  const statusEl = document.getElementById('status');
  const imageEl = document.getElementById('image');
  const overlayEl = document.getElementById('overlay');
  const modelSelect = document.getElementById('modelSelect');
  const backendSelect = document.getElementById('backendSelect');
  const landmarksCheck = document.getElementById('landmarksCheck');
  const runBtn = document.getElementById('runBtn');

  let detector = null;
  let landmarkDetector = null;

  modelSelect.addEventListener('change', () => {
    backendSelect.disabled = modelSelect.value === 'tflite';
    clearOverlay();
  });
  backendSelect.addEventListener('change', clearOverlay);
  landmarksCheck.addEventListener('change', clearOverlay);
  backendSelect.disabled = true;

  function clearOverlay() {
    const ctx = overlayEl.getContext('2d');
    ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);
    statusEl.textContent = 'Select model and click Run';
  }

  async function runDetection() {
    const modelType = modelSelect.value;
    const backend = backendSelect.value;
    const useLandmarks = landmarksCheck.checked;
    
    try {
      statusEl.textContent = `Loading ${modelType.toUpperCase()} model...`;
      runBtn.disabled = true;

      detector = new FaceDetector(modelType, backend);
      await detector.load();

      if (useLandmarks) {
        statusEl.textContent = `Loading landmark model...`;
        landmarkDetector = new FaceLandmarkDetector(modelType, backend);
        await landmarkDetector.load();
      }

      await new Promise(resolve => {
        if (imageEl.complete) {
          resolve();
        } else {
          imageEl.onload = resolve;
        }
      });

      overlayEl.width = imageEl.naturalWidth;
      overlayEl.height = imageEl.naturalHeight;
      overlayEl.style.width = imageEl.width + 'px';
      overlayEl.style.height = imageEl.height + 'px';

      const ctx = overlayEl.getContext('2d');
      ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);

      statusEl.textContent = 'Detecting faces...';

      const { detections, originalSize } = await detector.detect(imageEl);

      const [imgWidth, imgHeight] = originalSize;
      const scaleX = imageEl.width / imgWidth;
      const scaleY = imageEl.height / imgHeight;

      for (const detection of detections) {
        const [xmin, ymin] = detection.data[0];
        const [xmax, ymax] = detection.data[1];

        const pixelXmin = xmin * imgWidth;
        const pixelYmin = ymin * imgHeight;
        const pixelXmax = xmax * imgWidth;
        const pixelYmax = ymax * imgHeight;

        const scaleX = imageEl.width / imgWidth;
        const scaleY = imageEl.height / imgHeight;

        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(
          pixelXmin * scaleX,
          pixelYmin * scaleY,
          (pixelXmax - pixelXmin) * scaleX,
          (pixelYmax - pixelYmin) * scaleY
        );

        ctx.fillStyle = '#00FF00';
        ctx.font = '16px Arial';
        ctx.fillText(
          `Face: ${(detection.score * 100).toFixed(1)}%`,
          pixelXmin * scaleX,
          (pixelYmin - 5) * scaleY
        );
      }

      let landmarksResults = [];
      if (useLandmarks && detections.length > 0) {
        statusEl.textContent = 'Detecting landmarks...';
        landmarksResults = await landmarkDetector.detect(imageEl, detections);
        
        // Draw ROI rectangle on the main image for debugging
        const roi = landmarkDetector.computeROI(detections[0], originalSize);
        const roiX = (roi.xCenter - roi.width / 2) * imgWidth;
        const roiY = (roi.yCenter - roi.height / 2) * imgHeight;
        const roiW = roi.width * imgWidth;
        const roiH = roi.height * imgHeight;
        
        for (const landmarks of landmarksResults) {
          ctx.fillStyle = '#FF0000';
          for (const lm of landmarks) {
            const lmx = lm.x * imgWidth * scaleX;
            const lmy = lm.y * imgHeight * scaleY;
            ctx.beginPath();
            ctx.arc(lmx, lmy, 1.5, 0, Math.PI * 2);
            ctx.fill();
          }
          
          ctx.strokeStyle = '#FFFF00';
          ctx.lineWidth = 0.5;
          for (const [start, end] of FACE_LANDMARK_CONNECTIONS) {
            if (start < landmarks.length && end < landmarks.length) {
              const p1 = landmarks[start];
              const p2 = landmarks[end];
              ctx.beginPath();
              ctx.moveTo(p1.x * imgWidth * scaleX, p1.y * imgHeight * scaleY);
              ctx.lineTo(p2.x * imgWidth * scaleX, p2.y * imgHeight * scaleY);
              ctx.stroke();
            }
          }
        }
      }

      const backendInfo = modelType === 'tfjs' ? ` (${backend.toUpperCase()})` : '';
      let resultText = `Found ${detections.length} face(s) using ${modelType.toUpperCase()}${backendInfo} model`;
      if (useLandmarks) {
        const totalLandmarks = landmarksResults.reduce((sum, arr) => sum + arr.length, 0);
        resultText += `, ${totalLandmarks} landmarks`;
      }
      statusEl.textContent = resultText;

    } catch (error) {
      console.error('Error:', error);
      statusEl.textContent = 'Error: ' + error.message;
    } finally {
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener('click', runDetection);
}

window.addEventListener('load', main);
