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

    for (let k = 4; k < numPoints; k++) {
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

function removeLetterboxPadding(detections, padding, minScore) {
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

function computeDifference(arr1, arr2) {
  let maxDiff = 0;
  let sumDiff = 0;
  for (let i = 0; i < arr1.length; i++) {
    const diff = Math.abs(arr1[i] - arr2[i]);
    maxDiff = Math.max(maxDiff, diff);
    sumDiff += diff;
  }
  return { maxDiff, meanDiff: sumDiff / arr1.length };
}

async function main() {
  const statusEl = document.getElementById('status');
  
  const imageEl = new Image();
  imageEl.src = '../assets/portrait.jpg';
  await new Promise(resolve => {
    if (imageEl.complete) resolve();
    else imageEl.onload = resolve;
  });

  statusEl.textContent = 'Loading TFLite model...';

  const tfliteModel = await tflite.loadTFLiteModel('../assets/face_detector.tflite');
  console.log('TFLite model loaded');

  statusEl.textContent = 'Loading TFJS model...';
  
  const tfjsModel = await tf.loadGraphModel('../assets/face-detection-tfjs-short-v1/model.json');
  console.log('TFJS model loaded');

  const anchors = generateAnchors(SSD_OPTIONS_SHORT);
  const inputSize = { width: 128, height: 128 };

  const { tensor, padding, originalSize } = preprocessImage(imageEl, inputSize);

  statusEl.textContent = 'Running TFLite inference...';
  
  const inputTensor = tf.tensor4d(tensor, [1, inputSize.height, inputSize.width, 3]);
  const tfliteOutputs = tfliteModel.predict(inputTensor);

  const tfliteRawBoxes = await tfliteOutputs.regressors.data();
  const tfliteRawScores = await tfliteOutputs.classificators.data();

  console.log('TFLite regressors shape:', tfliteOutputs.regressors.shape);
  console.log('TFLite classificators shape:', tfliteOutputs.classificators.shape);

  console.log('TFLite raw boxes (first 5):', tfliteRawBoxes.slice(0, 80));
  console.log('TFLite raw scores (first 10):', tfliteRawScores.slice(0, 10));

  const tfliteBoxes = decodeBoxes(tfliteRawBoxes, anchors, inputSize);
  const tfliteScores = getSigmoidScores(tfliteRawScores);

  console.log('TFLite decoded boxes (first 3):', tfliteBoxes.slice(0, 3));
  console.log('TFLite sigmoid scores (first 10):', tfliteScores.slice(0, 10));

  const tfliteAllDetections = convertToDetections(tfliteBoxes, tfliteScores, 0);
  const tfliteSortedByScore = [...tfliteAllDetections].sort((a, b) => b.score - a.score).slice(0, 5);
  console.log('TFLite top 5 by sigmoid score:');
  tfliteSortedByScore.forEach((d, i) => {
    console.log(`  #${i+1}: score=${d.score.toFixed(6)}, bbox=[${d.data[0][0].toFixed(4)},${d.data[0][1].toFixed(4)},${d.data[1][0].toFixed(4)},${d.data[1][1].toFixed(4)}]`);
  });

  let tfliteDetections = convertToDetections(tfliteBoxes, tfliteScores, MIN_SCORE);
  tfliteDetections = nonMaxSuppression(tfliteDetections, MIN_SUPPRESSION_THRESHOLD);
  tfliteDetections = removeLetterboxPadding(tfliteDetections, padding, MIN_SCORE);

  console.log('TFLite detections:', tfliteDetections);

  inputTensor.dispose();
  tfliteOutputs.regressors.dispose();
  tfliteOutputs.classificators.dispose();

  statusEl.textContent = 'Running TFJS inference...';

  const inputTensor2 = tf.tensor4d(tensor, [1, inputSize.height, inputSize.width, 3]);
  const tfjsOutputs = tfjsModel.predict(inputTensor2);

  console.log('TFJS output shape:', tfjsOutputs.shape);

  const tfjsRawData = await tfjsOutputs.data();
  console.log('TFJS raw data (first 85):', tfjsRawData.slice(0, 85));

  const numAnchors = anchors.length;
  const numFeatures = tfjsRawData.length / numAnchors;

  const tfjsRawBoxes = [];
  const tfjsRawScores = [];

  // NOTE: For TFJS model, index 0 is the score (logit), indices 1-16 are the box data (regressors)
  // This is different from TFLite which has separate regressors and classificators outputs
  for (let i = 0; i < numAnchors; i++) {
    tfjsRawScores.push(tfjsRawData[i * numFeatures + 0]);
    for (let j = 1; j < 17; j++) {
      tfjsRawBoxes.push(tfjsRawData[i * numFeatures + j]);
    }
  }

  console.log('TFJS raw boxes (first 80):', tfjsRawBoxes.slice(0, 80));
  console.log('TFJS raw scores (first 10):', tfjsRawScores.slice(0, 10));
  console.log('TFJS raw data [0-17] (first anchor):', tfjsRawData.slice(0, 17));

  const tfjsBoxes = decodeBoxes(tfjsRawBoxes, anchors, inputSize);
  const tfjsScores = getSigmoidScores(tfjsRawScores);

  console.log('TFJS decoded boxes (first 3):', tfjsBoxes.slice(0, 3));
  console.log('TFJS sigmoid scores (first 10):', tfjsScores.slice(0, 10));

  const tfjsAllDetections = convertToDetections(tfjsBoxes, tfjsScores, 0);
  const tfjsSortedByScore = [...tfjsAllDetections].sort((a, b) => b.score - a.score).slice(0, 5);
  console.log('TFJS top 5 by sigmoid score:');
  tfjsSortedByScore.forEach((d, i) => {
    console.log(`  #${i+1}: score=${d.score.toFixed(6)}, bbox=[${d.data[0][0].toFixed(4)},${d.data[0][1].toFixed(4)},${d.data[1][0].toFixed(4)},${d.data[1][1].toFixed(4)}]`);
  });

  let tfjsDetections = convertToDetections(tfjsBoxes, tfjsScores, MIN_SCORE);
  tfjsDetections = nonMaxSuppression(tfjsDetections, MIN_SUPPRESSION_THRESHOLD);
  tfjsDetections = removeLetterboxPadding(tfjsDetections, padding, MIN_SCORE);

  console.log('TFJS detections:', tfjsDetections);

  inputTensor2.dispose();
  tfjsOutputs.dispose();

  statusEl.textContent = 'Comparing models...\n';

  const boxDiff = computeDifference(tfliteRawBoxes, tfjsRawBoxes);
  const scoreDiff = computeDifference(tfliteRawScores, tfjsRawScores);

  let report = '';
  report += `=== Model Output Shapes ===\n`;
  report += `NOTE: TFJS model uses index 0 for score, indices 1-16 for box data\n`;
  report += `TFJS: [${tfjsOutputs.shape}] (896 anchors x 17 features)\n`;
  report += `TFLite Regressors: [${tfliteOutputs.regressors.shape}] (896 anchors x 16 features)\n`;
  report += `TFLite Classificators: [${tfliteOutputs.classificators.shape}] (896 anchors x 1 score)\n\n`;

  report += `=== Raw Output Comparison ===\n`;
  report += `Regressors (boxes) - Max Diff: ${boxDiff.maxDiff.toFixed(10)}, Mean Diff: ${boxDiff.meanDiff.toFixed(10)}\n`;
  report += `Classificators (scores) - Max Diff: ${scoreDiff.maxDiff.toFixed(10)}, Mean Diff: ${scoreDiff.meanDiff.toFixed(10)}\n\n`;

  report += `=== Detection Results ===\n`;
  report += `TFLite: Found ${tfliteDetections.length} face(s)\n`;
  report += `TFJS: Found ${tfjsDetections.length} face(s)\n`;

  if (tfliteDetections.length > 0 && tfjsDetections.length > 0) {
    const tfliteBox = tfliteDetections[0].data;
    const tfjsBox = tfjsDetections[0].data;
    const bboxDiff = computeDifference(
      [tfliteBox[0][0], tfliteBox[0][1], tfliteBox[1][0], tfliteBox[1][1]],
      [tfjsBox[0][0], tfjsBox[0][1], tfjsBox[1][0], tfjsBox[1][1]]
    );
    const scoreDiff2 = Math.abs(tfliteDetections[0].score - tfjsDetections[0].score);
    
    report += `\n=== First Detection Comparison ===\n`;
    report += `BBox diff - Max: ${bboxDiff.maxDiff.toFixed(6)}, Mean: ${bboxDiff.meanDiff.toFixed(6)}\n`;
    report += `Score diff: ${scoreDiff2.toFixed(10)}\n`;
  }

  report += `\n=== Conclusion ===\n`;
  if (boxDiff.maxDiff < 1e-5 && scoreDiff.maxDiff < 1e-5) {
    report += 'Models are IDENTICAL (differences < 1e-5)';
  } else if (boxDiff.maxDiff < 1e-3 && scoreDiff.maxDiff < 1e-3) {
    report += 'Models are VERY SIMILAR (differences < 1e-3)';
  } else if (boxDiff.maxDiff < 0.1 && scoreDiff.maxDiff < 0.1) {
    report += 'Models are SIMILAR but have small differences';
  } else {
    report += 'Models may be DIFFERENT - check raw outputs above';
  }

  console.log(report);
  statusEl.textContent = report;
}

window.addEventListener('load', main);
