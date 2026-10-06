const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const log = (msg) => console.log(msg);
const logSection = (title) => log(`\n=== ${title} ===`);

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
    const scale = options.input_size_width;
    const boxes = [];
    
    for (let i = 0; i < anchors.length; i++) {
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

function preprocessImageJpeg(buffer, width, height) {
    const Jimp = require('jimp');
    return Jimp.read(buffer).then(image => {
        image.resize(width, height);
        const data = new Float32Array(width * height * 3);
        let idx = 0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixel = image.getPixelColor(x, y);
                const r = ((pixel >> 24) & 0xFF) / 255;
                const g = ((pixel >> 16) & 0xFF) / 255;
                const b = ((pixel >> 8) & 0xFF) / 255;
                data[idx++] = r * 2 - 1;
                data[idx++] = g * 2 - 1;
                data[idx++] = b * 2 - 1;
            }
        }
        return data;
    });
}

async function loadImage(imagePath, width, height) {
    const buffer = fs.readFileSync(imagePath);
    const Jimp = require('jimp');
    const image = await Jimp.read(buffer);
    image.resize(width, height);
    const data = new Float32Array(width * height * 3);
    let idx = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const r = Jimp.intToRGBA(image.getPixelColor(x, y)).r / 255;
            const g = Jimp.intToRGBA(image.getPixelColor(x, y)).g / 255;
            const b = Jimp.intToRGBA(image.getPixelColor(x, y)).b / 255;
            data[idx++] = r * 2 - 1;
            data[idx++] = g * 2 - 1;
            data[idx++] = b * 2 - 1;
        }
    }
    return data;
}

async function runTest() {
    try {
        logSection('LOADING MODELS');

        const assetsDir = path.join(__dirname, 'assets');
        const faceDetectorPath = path.join(assetsDir, 'face_detector.onnx');
        const faceLandmarkPath = path.join(assetsDir, 'face_landmark.onnx');

        log(`Loading ${faceDetectorPath}...`);
        const faceDetectorBuffer = fs.readFileSync(faceDetectorPath);
        log(`  Loaded ${faceDetectorBuffer.length} bytes`);

        log(`Loading ${faceLandmarkPath}...`);
        const faceLandmarkBuffer = fs.readFileSync(faceLandmarkPath);
        log(`  Loaded ${faceLandmarkBuffer.length} bytes`);

        logSection('CREATING SESSIONS');

        const detectorSession = await ort.InferenceSession.create(faceDetectorBuffer);
        log('Face detector session created');

        const landmarkSession = await ort.InferenceSession.create(faceLandmarkBuffer);
        log('Face landmark session created');

        logSection('LOADING IMAGE');

        const imagePath = path.join(assetsDir, 'portrait.jpg');
        const imageBuffer = fs.readFileSync(imagePath);
        const Jimp = require('jimp');
        const image = await Jimp.read(imageBuffer);
        const imgWidth = image.getWidth();
        const imgHeight = image.getHeight();
        log(`Image: ${imgWidth}x${imgHeight}`);

        logSection('FACE DETECTION');

        const inputSize = { width: 128, height: 128 };
        
        const inputTensorData = new Float32Array(inputSize.width * inputSize.height * 3);
        let idx = 0;
        image.resize(inputSize.width, inputSize.height);
        for (let y = 0; y < inputSize.height; y++) {
            for (let x = 0; x < inputSize.width; x++) {
                const pixel = image.getPixelColor(x, y);
                const r = ((pixel >> 24) & 0xFF) / 255;
                const g = ((pixel >> 16) & 0xFF) / 255;
                const b = ((pixel >> 8) & 0xFF) / 255;
                inputTensorData[idx++] = r * 2 - 1;
                inputTensorData[idx++] = g * 2 - 1;
                inputTensorData[idx++] = b * 2 - 1;
            }
        }

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
        log(`First 5 boxes: ${Array.from(rawBoxes.slice(0, 5)).join(', ')}`);
        log(`First 5 scores: ${Array.from(rawScores.slice(0, 5)).join(', ')}`);

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
            log(`Box: ${best.data[0][0].toFixed(3)}, ${best.data[0][1].toFixed(3)}, ${best.data[1][0].toFixed(3)}, ${best.data[1][1].toFixed(3)}`);
        }

        logSection('FACE LANDMARK');

        if (nmsResults.length > 0) {
            const landmarkInputSize = { width: 192, height: 192 };
            
            const lmTensorData = new Float32Array(landmarkInputSize.width * landmarkInputSize.height * 3);
            let idx = 0;
            image.resize(landmarkInputSize.width, landmarkInputSize.height);
            for (let y = 0; y < landmarkInputSize.height; y++) {
                for (let x = 0; x < landmarkInputSize.width; x++) {
                    const pixel = image.getPixelColor(x, y);
                    const r = ((pixel >> 24) & 0xFF) / 255;
                    const g = ((pixel >> 16) & 0xFF) / 255;
                    const b = ((pixel >> 8) & 0xFF) / 255;
                    lmTensorData[idx++] = r * 2 - 1;
                    lmTensorData[idx++] = g * 2 - 1;
                    lmTensorData[idx++] = b * 2 - 1;
                }
            }

            const lmInputTensor = new ort.Tensor('float32', lmTensorData, [1, landmarkInputSize.height, landmarkInputSize.width, 3]);
            const lmFeeds = { [landmarkSession.inputNames[0]]: lmInputTensor };

            const lmResults = await landmarkSession.run(lmFeeds);
            const lmOutputKeys = Object.keys(lmResults);
            log(`Outputs: ${lmOutputKeys.join(', ')}`);

            const rawLandmarks = lmResults[lmOutputKeys[0]].data;
            log(`Landmarks shape: ${lmResults[lmOutputKeys[0]].dims.join(', ')}`);
            log(`First 10 values: ${Array.from(rawLandmarks.slice(0, 10)).join(', ')}`);

            logSection('SUCCESS');
            log('ONNX Node.js test passed!');
        } else {
            log('No face detected, skipping landmark');
        }

    } catch (err) {
        log(`ERROR: ${err.message}`);
        console.error(err);
    }
}

runTest();