const log = (msg, type = '') => {
    const statusEl = document.getElementById('status');
    const line = document.createElement('div');
    line.textContent = msg;
    if (type) line.className = type;
    statusEl.appendChild(line);
    console.log(msg);
};

const logSection = (title) => {
    log(`\n=== ${title} ===`);
};

async function loadModel(url) {
    log(`Loading ${url}...`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
    const arrayBuffer = await response.arrayBuffer();
    log(`  Loaded ${arrayBuffer.byteLength} bytes`);
    return arrayBuffer;
}

const MIN_SCORE = 0.5;
const MIN_SUPPRESSION_THRESHOLD = 0.3;

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

function decodeBoxes(rawBoxes, anchors, options) {
    const numAnchors = anchors.length;
    const scale = options.input_size_width;
    const boxes = [];
    
    for (let i = 0; i < numAnchors; i++) {
        const baseOffset = i * 16;
        
        const xCenter = rawBoxes[baseOffset + 1] / scale + anchors[i][0];
        const yCenter = rawBoxes[baseOffset + 0] / scale + anchors[i][1];
        const h = Math.exp(rawBoxes[baseOffset + 2]) / scale;
        const w = Math.exp(rawBoxes[baseOffset + 3]) / scale;
        
        const xmin = xCenter - w / 2;
        const ymin = yCenter - h / 2;
        const xmax = xCenter + w / 2;
        const ymax = yCenter + h / 2;
        
        boxes.push([xmin, ymin, xmax, ymax]);
    }
    
    return boxes;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function calculateIou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) {
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

function nonMaxSuppression(boxes, scores, options) {
    const detections = boxes.map((box, i) => ({
        data: [[box[0], box[1]], [box[2], box[3]]],
        score: scores[i]
    })).filter(d => d.score >= options.minScore);

    if (detections.length === 0) return [];

    const sortedDetections = [...detections].sort((a, b) => b.score - a.score);
    const keep = [];

    while (sortedDetections.length > 0) {
        const current = sortedDetections.shift();
        keep.push(current);

        const remaining = [];
        for (const detection of sortedDetections) {
            const iou = calculateIou(
                current.data[0][0], current.data[0][1], current.data[1][0], current.data[1][1],
                detection.data[0][0], detection.data[0][1], detection.data[1][0], detection.data[1][1]
            );
            if (iou < options.iouThreshold) {
                remaining.push(detection);
            }
        }
        sortedDetections.length = 0;
        sortedDetections.push(...remaining);
    }

    return keep;
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

    return tensor;
}

async function runTest() {
    try {
        logSection('LOADING MODELS');

        const faceDetectorBuffer = await loadModel('../assets/face_detector.onnx');
        const faceLandmarkBuffer = await loadModel('../assets/face_landmark.onnx');

        logSection('CREATING SESSIONS');

        const sessionOptions = {
            graphOptimizationLevel: 'all'
        };

        const detectorSession = await ort.InferenceSession.create(faceDetectorBuffer, sessionOptions);
        log('Face detector session created');

        const landmarkSession = await ort.InferenceSession.create(faceLandmarkBuffer, sessionOptions);
        log('Face landmark session created');

        logSection('LOADING IMAGE');

        const imageEl = document.createElement('img');
        imageEl.src = '../assets/portrait.jpg';
        await new Promise((resolve, reject) => {
            imageEl.onload = resolve;
            imageEl.onerror = reject;
        });
        log(`Image loaded: ${imageEl.width}x${imageEl.height}`);

        logSection('FACE DETECTION');

        const inputSize = { width: 128, height: 128 };
        const inputTensorData = preprocessImage(imageEl, inputSize);

        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, inputSize.height, inputSize.width, 3]);
        const feeds = { [detectorSession.inputNames[0]]: inputTensor };

        const results = await detectorSession.run(feeds);
        const outputKeys = Object.keys(results);
        log(`Outputs: ${outputKeys.join(', ')}`);

        const regressorsKey = outputKeys.find(k => k.includes('regress')) || outputKeys[0];
        const classificatorsKey = outputKeys.find(k => k.includes('class')) || outputKeys[1];

        const rawBoxes = results[regressorsKey].data;
        const rawScores = results[classificatorsKey].data;
        log(`Boxes shape: ${results[regressorsKey].dims.join(', ')}`);
        log(`Scores shape: ${results[classificatorsKey].dims.join(', ')}`);

        const anchors = generateAnchors(SSD_OPTIONS_SHORT);
        log(`Generated ${anchors.length} anchors`);

        const decodedBoxes = decodeBoxes(rawBoxes, anchors, SSD_OPTIONS_SHORT);

        const scores = new Float32Array(rawScores.length);
        for (let i = 0; i < rawScores.length; i++) {
            scores[i] = sigmoid(rawScores[i]);
        }

        const nmsResults = nonMaxSuppression(decodedBoxes, scores, {
            minScore: MIN_SCORE,
            iouThreshold: MIN_SUPPRESSION_THRESHOLD
        });

        log(`Detections: ${nmsResults.length}`);
        if (nmsResults.length > 0) {
            const best = nmsResults[0];
            log(`Best score: ${best.score.toFixed(4)}`);
            log(`Box: ${best.box.map(v => v.toFixed(3)).join(', ')}`);
        }

        logSection('FACE LANDMARK');

        if (nmsResults.length > 0) {
            const landmarkInputSize = { width: 192, height: 192 };
            const lmTensor = preprocessImage(imageEl, landmarkInputSize);
            const lmInputTensor = new ort.Tensor('float32', lmTensor, [1, landmarkInputSize.height, landmarkInputSize.width, 3]);

            const lmFeeds = { [landmarkSession.inputNames[0]]: lmInputTensor };
            const lmResults = await landmarkSession.run(lmFeeds);

            const lmOutputKeys = Object.keys(lmResults);
            log(`Outputs: ${lmOutputKeys.join(', ')}`);

            const rawLandmarks = lmResults[lmOutputKeys[0]].data;
            log(`Landmarks shape: ${lmResults[lmOutputKeys[0]].dims.join(', ')}`);
            log(`First 10 values: ${Array.from(rawLandmarks.slice(0, 10)).join(', ')}`);

            log('PASS', 'pass');
        } else {
            log('No face detected, skipping landmark', 'fail');
        }

        logSection('COMPLETE');
        log('ONNX JavaScript test completed successfully!', 'pass');

    } catch (err) {
        log(`ERROR: ${err.message}`, 'fail');
        console.error(err);
    }
}

runTest();