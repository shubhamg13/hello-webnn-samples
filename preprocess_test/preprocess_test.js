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

function log(msg) {
  console.log(msg);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent += msg + '\n';
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

  log(`Preprocessing: aspectRatio=${aspectRatio}, targetAspectRatio=${targetAspectRatio}`);
  log(`Preprocessing: scale=${scale}, canvasWidth=${canvasWidth}, canvasHeight=${canvasHeight}`);
  log(`Preprocessing: padX=${padX}, padY=${padY}`);

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

  log(`Preprocessing: drawX=${drawX}, drawY=${drawY}, drawWidth=${drawWidth}, drawHeight=${drawHeight}`);

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

async function main() {
  const statusEl = document.getElementById('status');
  const imageEl = document.getElementById('image');
  
  statusEl.textContent = '';

  try {
    await new Promise(resolve => {
      if (imageEl.complete) resolve();
      else imageEl.onload = resolve;
    });

    log(`Image: ${imageEl.naturalWidth}x${imageEl.naturalHeight}`);

    const faceInputSize = { width: 128, height: 128 };
    const { tensor, padding, originalSize } = preprocessImage(imageEl, faceInputSize);

    log(`Padding: ${JSON.stringify(padding)}`);

    // Log tensor values at specific locations
    const getPixel = (row, col) => {
      const idx = (row * 128 + col) * 3;
      return [tensor[idx], tensor[idx+1], tensor[idx+2]];
    };

    // Check letterbox areas
    log(`Top-left (0,0): ${JSON.stringify(getPixel(0, 0))}`);
    log(`Letterbox area (10, 64): ${JSON.stringify(getPixel(10, 64))}`);
    log(`Letterbox area (20, 64): ${JSON.stringify(getPixel(20, 64))}`);
    
    // Check actual image content areas
    log(`Image area (50, 64): ${JSON.stringify(getPixel(50, 64))}`);
    log(`Image area (64, 64): ${JSON.stringify(getPixel(64, 64))}`);
    log(`Image area (90, 64): ${JSON.stringify(getPixel(90, 64))}`);
    
    // Test anchors
    const anchors = generateAnchors(SSD_OPTIONS_SHORT);
    log(`Number of anchors: ${anchors.length}`);
    log(`First 5 anchors: ${JSON.stringify(anchors.slice(0, 5))}`);

  } catch (error) {
    console.error('Error:', error);
    log('Error: ' + error.message);
  }
}

window.addEventListener('load', main);