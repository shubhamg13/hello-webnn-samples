const MIN_SCORE = 0.5;
const MIN_SUPPRESSION_THRESHOLD = 0.3;
const RAW_SCORE_LIMIT = 80;

const NUM_LANDMARKS = 468;  // 1404 / 3 - from reference face_landmark.tflite
const NUM_DIMS = 3;
const ROI_SCALE = 1.5;
const LANDMARK_DETECTION_THRESHOLD = 0.5;
const LANDMARK_INPUT_SIZE = { width: 192, height: 192 };

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

function log(msg) {
  console.log(msg);
  const statusEl = document.getElementById('status');
  statusEl.textContent += msg + '\n';
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

function nonMaxSuppression(detections, minSuppressionThreshold) {
  if (detections.length === 0) return [];

  const sortedDetections = [...detections].sort((a, b) => b.score - a.score);
  const keep = [];

  while (sortedDetections.length > 0) {
    const current = sortedDetections.shift();
    keep.push(current);

    const remaining = [];
    for (const detection of sortedDetections) {
      const iou = computeIoU(
        current.data[0][0], current.data[0][1], current.data[1][0], current.data[1][1],
        detection.data[0][0], detection.data[0][1], detection.data[1][0], detection.data[1][1]
      );
      if (iou < minSuppressionThreshold) {
        remaining.push(detection);
      }
    }
    sortedDetections.length = 0;
    sortedDetections.push(...remaining);
  }

  return keep;
}

function sigmoid(x) {
  if (x < -RAW_SCORE_LIMIT) x = -RAW_SCORE_LIMIT;
  if (x > RAW_SCORE_LIMIT) x = RAW_SCORE_LIMIT;
  return 1 / (1 + Math.exp(-x));
}

function preprocessImage(imageElement, inputSize) {
  const originalWidth = imageElement.naturalWidth;
  const originalHeight = imageElement.naturalHeight;
  const originalSize = [originalWidth, originalHeight];

  const targetWidth = inputSize.width;
  const targetHeight = inputSize.height;

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

function getSigmoidScores(rawScores) {
  const scores = [];
  for (let i = 0; i < rawScores.length; i++) {
    scores.push(sigmoid(rawScores[i]));
  }
  return scores;
}

function convertToDetections(boxes, scores, minScore) {
  const detections = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > minScore) {
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

function removeLetterboxPadding(detections, padding) {
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

function computeROI(detection, imageSize) {
  const [imgWidth, imgHeight] = imageSize;
  const [xmin, ymin] = detection.data[0];
  const [xmax, ymax] = detection.data[1];
  
  const width = xmax - xmin;
  const height = ymax - ymin;
  const cx = (xmin + xmax) / 2;
  const cy = (ymin + ymax) / 2;
  
  const leftEye = detection.data[FaceIndex.LEFT_EYE + 2];
  const rightEye = detection.data[FaceIndex.RIGHT_EYE + 2];
  
  // Python uses: y0 - y1 (leftEye - rightEye)
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
  
  console.log('JS ROI:', { xCenter: cx, yCenter: cy, width: roiWidth, height: roiHeight, rotation: rotation });
  
  return {
    xCenter: cx,
    yCenter: cy,
    width: roiWidth,
    height: roiHeight,
    rotation: rotation,
    normalized: true
  };
}

function cropImageToROI(imageElement, roi, outputSize) {
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
  const debugCtx = debugCanvas.getContext('2d');
  debugCanvas.width = outputSize.width;
  debugCanvas.height = outputSize.height;
  debugCtx.drawImage(canvas, 0, 0);
  
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

function projectLandmarks(landmarks, roi, originalSize) {
  const cos = Math.cos(roi.rotation);
  const sin = Math.sin(roi.rotation);
  
  return landmarks.map(lm => {
    let x = lm.x / LANDMARK_INPUT_SIZE.width;
    let y = lm.y / LANDMARK_INPUT_SIZE.height;
    let z = lm.z / LANDMARK_INPUT_SIZE.width;
    
    x -= 0.5;
    y -= 0.5;
    
    const rx = x * cos - y * sin;
    const ry = x * sin + y * cos;
    x = rx;
    y = ry;
    
    x *= roi.width;
    y *= roi.height;
    
    x += roi.xCenter;
    y += roi.yCenter;
    
    return { x, y, z };
  });
}

async function main() {
  const statusEl = document.getElementById('status');
  const imageEl = document.getElementById('image');
  const overlayEl = document.getElementById('overlay');
  
  statusEl.textContent = '';
  
  try {
    log('Loading face detection model...');
    const faceDetectorModel = await tflite.loadTFLiteModel('../assets/face_detector_ref.tflite');
    log('Face detector loaded (ref model)');
    
    log('\nLoading face landmarks model...');
    const landmarkModel = await tflite.loadTFLiteModel('../assets/face_landmark_ref.tflite');
    log('Face landmarks model loaded');
    
    await new Promise(resolve => {
      if (imageEl.complete) resolve();
      else imageEl.onload = resolve;
    });
    
    overlayEl.width = imageEl.naturalWidth;
    overlayEl.height = imageEl.naturalHeight;
    overlayEl.style.width = imageEl.width + 'px';
    overlayEl.style.height = imageEl.height + 'px';
    
    log('Running face detection...');
    const anchors = generateAnchors(SSD_OPTIONS_SHORT);
    const faceInputSize = { width: 128, height: 128 };
    
    const { tensor, padding, originalSize } = preprocessImage(imageEl, faceInputSize);
    
    log(`JS padding: ${JSON.stringify(padding)}`);
    log(`JS tensor shape: ${tensor.length}`);
    // Check RGB values in -1 to 1 range
    const idx32 = (32 * 128 + 64) * 3;
    const idx64 = (64 * 128 + 64) * 3;
    const idx96 = (96 * 128 + 64) * 3;
    log(`JS tensor at (32,64) [RGB]: ${tensor[idx32]}, ${tensor[idx32+1]}, ${tensor[idx32+2]}`);
    log(`JS tensor at (64,64) [RGB]: ${tensor[idx64]}, ${tensor[idx64+1]}, ${tensor[idx64+2]}`);
    log(`JS tensor at (96,64) [RGB]: ${tensor[idx96]}, ${tensor[idx96+1]}, ${tensor[idx96+2]}`);
    const inputTensor = tf.tensor4d(tensor, [1, faceInputSize.height, faceInputSize.width, 3]);
    const faceOutputs = faceDetectorModel.predict(inputTensor);
    
    const rawBoxes = await faceOutputs.regressors.data();
    const rawScores = await faceOutputs.classificators.data();
    
    log(`=== Raw Model Output ===`);
    log(`rawBoxes (first 20): ${JSON.stringify(Array.from(rawBoxes.slice(0, 20)))}`);
    log(`rawScores (first 20): ${JSON.stringify(Array.from(rawScores.slice(0, 20)))}`);
    log(`Anchors (first 3): ${JSON.stringify(anchors.slice(0, 3))}`);
    log(`Num anchors: ${anchors.length}`);
    
    const boxes = decodeBoxes(rawBoxes, anchors, faceInputSize);
    const scores = getSigmoidScores(rawScores);
    let detections = convertToDetections(boxes, scores, MIN_SCORE);
    
    log(`=== Before NMS ===`);
    log(`Total detections: ${detections.length}`);
    log(`All detections: ${JSON.stringify(detections.map(d => ({score: d.score, bbox: [d.data[0], d.data[1]]})))}`);
    
    detections = nonMaxSuppression(detections, MIN_SUPPRESSION_THRESHOLD);
    
    log(`=== After NMS (before letterbox) ===`);
    log(`Total detections: ${detections.length}`);
    log(`All detections: ${JSON.stringify(detections.map(d => ({score: d.score, bbox: [d.data[0], d.data[1]]})))}`);
    
    detections = removeLetterboxPadding(detections, padding);
    
    log(`=== After letterbox removal ===`);
    log(`Total detections: ${detections.length}`);
    
    log(`Detected ${detections.length} face(s)`);
    if (detections.length > 0) {
      const d = detections[0];
      log(`Detection bbox: xmin=${d.data[0]}, ymin=${d.data[1]}`);
      log(`Detection data[2]: ${d.data[2]}`);
      log(`Detection data[3]: ${d.data[3]}`);
      log(`Detection FaceIndex.LEFT_EYE+2: ${d.data[FaceIndex.LEFT_EYE + 2]}`);
      log(`Detection FaceIndex.RIGHT_EYE+2: ${d.data[FaceIndex.RIGHT_EYE + 2]}`);
    }
    
    inputTensor.dispose();
    faceOutputs.regressors.dispose();
    faceOutputs.classificators.dispose();
    
    const ctx = overlayEl.getContext('2d');
    const [imgWidth, imgHeight] = originalSize;
    
    for (const detection of detections) {
      const [xmin, ymin] = detection.data[0];
      const [xmax, ymax] = detection.data[1];
      
      const pixelXmin = xmin * imgWidth;
      const pixelYmin = ymin * imgHeight;
      const pixelXmax = xmax * imgWidth;
      const pixelYmax = ymax * imgHeight;
      
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(pixelXmin, pixelYmin, pixelXmax - pixelXmin, pixelYmax - pixelYmin);
      
      const leftEye = detection.data[FaceIndex.LEFT_EYE + 2];
      const rightEye = detection.data[FaceIndex.RIGHT_EYE + 2];
      
      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.arc(leftEye[0] * imgWidth, leftEye[1] * imgHeight, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(rightEye[0] * imgWidth, rightEye[1] * imgHeight, 5, 0, Math.PI * 2);
      ctx.fill();
      
      const roi = computeROI(detection, originalSize);
      
      const landmarkInputSize = LANDMARK_INPUT_SIZE;
      const landmarkTensor = cropImageToROI(imageEl, roi, landmarkInputSize);
      
      const lmInputTensor = tf.tensor4d(landmarkTensor, [1, landmarkInputSize.height, landmarkInputSize.width, 3]);
      const lmOutputs = landmarkModel.predict(lmInputTensor);
      
      let rawLandmarks, rawFaceFlag;
      
      if (Array.isArray(lmOutputs)) {
        rawLandmarks = await lmOutputs[0].data();
        rawFaceFlag = await lmOutputs[1].data();
      } else {
        const outputKeys = Object.keys(lmOutputs);
        rawLandmarks = await lmOutputs[outputKeys[0]].data();
        rawFaceFlag = await lmOutputs[outputKeys[1]].data();
      }
      
      lmInputTensor.dispose();
      if (Array.isArray(lmOutputs)) {
        lmOutputs.forEach(o => o.dispose());
      } else if (typeof lmOutputs === 'object') {
        Object.values(lmOutputs).forEach(o => o.dispose());
      }
      
      const faceFlagRaw = rawFaceFlag.length ? rawFaceFlag[0] : rawFaceFlag;
      const faceFlag = sigmoid(faceFlagRaw);
      
      if (faceFlag > LANDMARK_DETECTION_THRESHOLD) {
        const landmarks = [];
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          const x = rawLandmarks[i * 3];
          const y = rawLandmarks[i * 3 + 1];
          const z = rawLandmarks[i * 3 + 2];
          landmarks.push({ x, y, z });
        }
        
        const projectedLandmarks = projectLandmarks(landmarks, roi, originalSize);
        
        log(`Landmarks detected: ${projectedLandmarks.length}`);
        
        ctx.fillStyle = '#FF0000';
        for (const lm of projectedLandmarks) {
          const lmx = lm.x * imgWidth;
          const lmy = lm.y * imgHeight;
          ctx.beginPath();
          ctx.arc(lmx, lmy, 1, 0, Math.PI * 2);
          ctx.fill();
        }
        
        log(`Face landmarks detection SUCCESS`);
      } else {
        log(`Face landmarks detection FAILED (face flag below threshold)`);
      }
    }
    
    log('\n=== Verification Complete ===');
    
  } catch (error) {
    console.error('Error:', error);
    log('Error: ' + error.message);
  }
}

window.addEventListener('load', main);
