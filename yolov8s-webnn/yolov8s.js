class WeightsFile {
  constructor(buffer, manifest) {
    this.buffer = buffer;
    this.manifest = manifest;
  }

  static async load(weightsPath, manifestPath) {
    const [wRes, mRes] = await Promise.all([fetch(weightsPath), fetch(manifestPath)]);
    if (!wRes.ok) throw new Error('Failed to load weights: ' + wRes.statusText);
    if (!mRes.ok) throw new Error('Failed to load manifest: ' + mRes.statusText);
    const buffer = await wRes.arrayBuffer();
    const manifest = await mRes.json();
    if (manifest.format !== 'wg-weights-manifest') throw new Error('Invalid manifest format');
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
    if (magic !== 'WGWT') throw new Error('Invalid weights file');
    return new WeightsFile(buffer, manifest);
  }

  getSlice(name) {
    const t = this.manifest.tensors[name];
    if (!t) throw new Error('Tensor not found: ' + name);
    return t;
  }

  getData(name) {
    const t = this.getSlice(name);
    return this.buffer.slice(t.byteOffset, t.byteOffset + t.byteLength);
  }
}

/**
 * Build WebNN graph for model: yolov8s
 * Source format: onnx
 */
async function buildGraph(context, weights) {
  const builder = new MLGraphBuilder(context);

  // Graph inputs
  const images = builder.input('images', { dataType: 'float32', shape: [1,3,640,640] });

  // Constants (loaded from weights file)
  const model_0_conv_weight = (() => {
    const sl = weights.getSlice('model.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [32,3,3,3] }, new Float32Array(buf));
  })();
  const model_0_conv_bias = (() => {
    const sl = weights.getSlice('model.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [32] }, new Float32Array(buf));
  })();
  const model_1_conv_weight = (() => {
    const sl = weights.getSlice('model.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,32,3,3] }, new Float32Array(buf));
  })();
  const model_1_conv_bias = (() => {
    const sl = weights.getSlice('model.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_2_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.2.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,1,1] }, new Float32Array(buf));
  })();
  const model_2_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.2.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const onnx__Split_137 = (() => {
    const sl = weights.getSlice('onnx::Split_137');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [2] }, new BigInt64Array(buf));
  })();
  const model_2_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.2.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [32,32,3,3] }, new Float32Array(buf));
  })();
  const model_2_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.2.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [32] }, new Float32Array(buf));
  })();
  const model_2_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.2.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [32,32,3,3] }, new Float32Array(buf));
  })();
  const model_2_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.2.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [32] }, new Float32Array(buf));
  })();
  const model_2_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.2.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,96,1,1] }, new Float32Array(buf));
  })();
  const model_2_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.2.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_3_conv_weight = (() => {
    const sl = weights.getSlice('model.3.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,64,3,3] }, new Float32Array(buf));
  })();
  const model_3_conv_bias = (() => {
    const sl = weights.getSlice('model.3.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_4_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.4.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,1,1] }, new Float32Array(buf));
  })();
  const model_4_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.4.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const onnx__Split_157 = (() => {
    const sl = weights.getSlice('onnx::Split_157');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [2] }, new BigInt64Array(buf));
  })();
  const model_4_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.4.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_4_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.4.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_4_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.4.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_4_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.4.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_4_m_1_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.4.m.1.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_4_m_1_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.4.m.1.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_4_m_1_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.4.m.1.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_4_m_1_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.4.m.1.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_4_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.4.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,256,1,1] }, new Float32Array(buf));
  })();
  const model_4_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.4.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_5_conv_weight = (() => {
    const sl = weights.getSlice('model.5.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,128,3,3] }, new Float32Array(buf));
  })();
  const model_5_conv_bias = (() => {
    const sl = weights.getSlice('model.5.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_6_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.6.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,256,1,1] }, new Float32Array(buf));
  })();
  const model_6_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.6.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const onnx__Split_184 = (() => {
    const sl = weights.getSlice('onnx::Split_184');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [2] }, new BigInt64Array(buf));
  })();
  const model_6_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.6.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_6_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.6.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_6_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.6.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_6_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.6.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_6_m_1_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.6.m.1.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_6_m_1_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.6.m.1.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_6_m_1_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.6.m.1.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_6_m_1_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.6.m.1.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_6_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.6.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,512,1,1] }, new Float32Array(buf));
  })();
  const model_6_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.6.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_7_conv_weight = (() => {
    const sl = weights.getSlice('model.7.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512,256,3,3] }, new Float32Array(buf));
  })();
  const model_7_conv_bias = (() => {
    const sl = weights.getSlice('model.7.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512] }, new Float32Array(buf));
  })();
  const model_8_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.8.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512,512,1,1] }, new Float32Array(buf));
  })();
  const model_8_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.8.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512] }, new Float32Array(buf));
  })();
  const onnx__Split_211 = (() => {
    const sl = weights.getSlice('onnx::Split_211');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [2] }, new BigInt64Array(buf));
  })();
  const model_8_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.8.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,256,3,3] }, new Float32Array(buf));
  })();
  const model_8_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.8.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_8_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.8.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,256,3,3] }, new Float32Array(buf));
  })();
  const model_8_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.8.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_8_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.8.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512,768,1,1] }, new Float32Array(buf));
  })();
  const model_8_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.8.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512] }, new Float32Array(buf));
  })();
  const model_9_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.9.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,512,1,1] }, new Float32Array(buf));
  })();
  const model_9_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.9.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_9_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.9.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512,1024,1,1] }, new Float32Array(buf));
  })();
  const model_9_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.9.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512] }, new Float32Array(buf));
  })();
  const _model_10_Constant_output_0 = (() => {
    const sl = weights.getSlice('/model.10/Constant_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [4] }, new Float32Array(buf));
  })();
  const model_12_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.12.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,768,1,1] }, new Float32Array(buf));
  })();
  const model_12_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.12.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_12_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.12.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_12_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.12.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_12_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.12.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_12_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.12.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_12_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.12.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,384,1,1] }, new Float32Array(buf));
  })();
  const model_12_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.12.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_15_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.15.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,384,1,1] }, new Float32Array(buf));
  })();
  const model_15_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.15.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_15_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.15.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_15_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.15.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_15_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.15.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_15_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.15.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_15_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.15.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,192,1,1] }, new Float32Array(buf));
  })();
  const model_15_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.15.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_16_conv_weight = (() => {
    const sl = weights.getSlice('model.16.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_16_conv_bias = (() => {
    const sl = weights.getSlice('model.16.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_22_cv2_0_0_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.0.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,128,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv2_0_0_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.0.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_0_0_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.0.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv3_0_0_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.0.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_22_cv2_0_1_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.0.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv2_0_1_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.0.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_0_1_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.0.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv3_0_1_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.0.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_18_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.18.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,384,1,1] }, new Float32Array(buf));
  })();
  const model_18_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.18.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_22_cv2_0_2_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.0.2.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,1,1] }, new Float32Array(buf));
  })();
  const model_22_cv2_0_2_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.0.2.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_0_2_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.0.2.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [80,128,1,1] }, new Float32Array(buf));
  })();
  const model_22_cv3_0_2_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.0.2.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [80] }, new Float32Array(buf));
  })();
  const _model_22_Constant_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [3] }, new BigInt64Array(buf));
  })();
  const _model_22_Constant_3_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_3_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [3] }, new BigInt64Array(buf));
  })();
  const model_18_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.18.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_18_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.18.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_18_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.18.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_18_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.18.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_18_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.18.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,384,1,1] }, new Float32Array(buf));
  })();
  const model_18_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.18.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_19_conv_weight = (() => {
    const sl = weights.getSlice('model.19.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,256,3,3] }, new Float32Array(buf));
  })();
  const model_19_conv_bias = (() => {
    const sl = weights.getSlice('model.19.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_22_cv2_1_0_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.1.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,256,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv2_1_0_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.1.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_1_0_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.1.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,256,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv3_1_0_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.1.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_22_cv2_1_1_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.1.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv2_1_1_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.1.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_1_1_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.1.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv3_1_1_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.1.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_21_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.21.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512,768,1,1] }, new Float32Array(buf));
  })();
  const model_21_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.21.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512] }, new Float32Array(buf));
  })();
  const model_22_cv2_1_2_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.1.2.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,1,1] }, new Float32Array(buf));
  })();
  const model_22_cv2_1_2_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.1.2.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_1_2_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.1.2.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [80,128,1,1] }, new Float32Array(buf));
  })();
  const model_22_cv3_1_2_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.1.2.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [80] }, new Float32Array(buf));
  })();
  const model_21_m_0_cv1_conv_weight = (() => {
    const sl = weights.getSlice('model.21.m.0.cv1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,256,3,3] }, new Float32Array(buf));
  })();
  const model_21_m_0_cv1_conv_bias = (() => {
    const sl = weights.getSlice('model.21.m.0.cv1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_21_m_0_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.21.m.0.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256,256,3,3] }, new Float32Array(buf));
  })();
  const model_21_m_0_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.21.m.0.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [256] }, new Float32Array(buf));
  })();
  const model_21_cv2_conv_weight = (() => {
    const sl = weights.getSlice('model.21.cv2.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512,768,1,1] }, new Float32Array(buf));
  })();
  const model_21_cv2_conv_bias = (() => {
    const sl = weights.getSlice('model.21.cv2.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [512] }, new Float32Array(buf));
  })();
  const model_22_cv2_2_0_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.2.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,512,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv2_2_0_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.2.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_2_0_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.2.0.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,512,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv3_2_0_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.2.0.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_22_cv2_2_1_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.2.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv2_2_1_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.2.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_2_1_conv_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.2.1.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128,128,3,3] }, new Float32Array(buf));
  })();
  const model_22_cv3_2_1_conv_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.2.1.conv.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [128] }, new Float32Array(buf));
  })();
  const model_22_cv2_2_2_weight = (() => {
    const sl = weights.getSlice('model.22.cv2.2.2.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64,64,1,1] }, new Float32Array(buf));
  })();
  const model_22_cv2_2_2_bias = (() => {
    const sl = weights.getSlice('model.22.cv2.2.2.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [64] }, new Float32Array(buf));
  })();
  const model_22_cv3_2_2_weight = (() => {
    const sl = weights.getSlice('model.22.cv3.2.2.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [80,128,1,1] }, new Float32Array(buf));
  })();
  const model_22_cv3_2_2_bias = (() => {
    const sl = weights.getSlice('model.22.cv3.2.2.bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [80] }, new Float32Array(buf));
  })();
  const _model_22_dfl_Constant_output_0 = (() => {
    const sl = weights.getSlice('/model.22/dfl/Constant_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [4] }, new BigInt64Array(buf));
  })();
  const model_22_dfl_conv_weight = (() => {
    const sl = weights.getSlice('model.22.dfl.conv.weight');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [1,16,1,1] }, new Float32Array(buf));
  })();
  const _model_22_dfl_Constant_1_output_0 = (() => {
    const sl = weights.getSlice('/model.22/dfl/Constant_1_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [3] }, new BigInt64Array(buf));
  })();
  const _model_22_Constant_7_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_7_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [1] }, new BigInt64Array(buf));
  })();
  const _model_22_Mul_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Mul_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [1] }, new BigInt64Array(buf));
  })();
  const _model_22_Constant_6_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_6_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [1] }, new BigInt64Array(buf));
  })();
  const _model_22_Mul_1_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Mul_1_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int64', shape: [1] }, new BigInt64Array(buf));
  })();
  const _model_22_Constant_12_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_12_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [1,2,8400] }, new Float32Array(buf));
  })();
  const _model_22_Constant_13_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_13_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [1,2,8400] }, new Float32Array(buf));
  })();
  const _model_22_Constant_14_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_14_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [1] }, new Float32Array(buf));
  })();
  const _model_22_Constant_15_output_0 = (() => {
    const sl = weights.getSlice('/model.22/Constant_15_output_0');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float32', shape: [1,8400] }, new Float32Array(buf));
  })();

  // Graph operations
  // Conv
  const _model_0_conv_Conv_output_0 = builder.conv2d(images, model_0_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_0_conv_bias });
  // Sigmoid
  const _model_0_act_Sigmoid_output_0 = builder.sigmoid(_model_0_conv_Conv_output_0);
  // Mul
  const _model_0_act_Mul_output_0 = builder.mul(_model_0_conv_Conv_output_0, _model_0_act_Sigmoid_output_0);
  // Conv
  const _model_1_conv_Conv_output_0 = builder.conv2d(_model_0_act_Mul_output_0, model_1_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_1_conv_bias });
  // Sigmoid
  const _model_1_act_Sigmoid_output_0 = builder.sigmoid(_model_1_conv_Conv_output_0);
  // Mul
  const _model_1_act_Mul_output_0 = builder.mul(_model_1_conv_Conv_output_0, _model_1_act_Sigmoid_output_0);
  // Conv
  const _model_2_cv1_conv_Conv_output_0 = builder.conv2d(_model_1_act_Mul_output_0, model_2_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_2_cv1_conv_bias });
  // Sigmoid
  const _model_2_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_2_cv1_conv_Conv_output_0);
  // Mul
  const _model_2_cv1_act_Mul_output_0 = builder.mul(_model_2_cv1_conv_Conv_output_0, _model_2_cv1_act_Sigmoid_output_0);
  // Split
  const _model_2_Split_output_0_splits = builder.split(_model_2_cv1_act_Mul_output_0, [32, 32], { axis: 1 });
  const _model_2_Split_output_0 = _model_2_Split_output_0_splits[0];
  const _model_2_Split_output_1 = _model_2_Split_output_0_splits[1];
  // Conv
  const _model_2_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_2_Split_output_1, model_2_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_2_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_2_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_2_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_2_m_0_cv1_act_Mul_output_0 = builder.mul(_model_2_m_0_cv1_conv_Conv_output_0, _model_2_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_2_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_2_m_0_cv1_act_Mul_output_0, model_2_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_2_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_2_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_2_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_2_m_0_cv2_act_Mul_output_0 = builder.mul(_model_2_m_0_cv2_conv_Conv_output_0, _model_2_m_0_cv2_act_Sigmoid_output_0);
  // Add
  const _model_2_m_0_Add_output_0 = builder.add(_model_2_Split_output_1, _model_2_m_0_cv2_act_Mul_output_0);
  // Concat
  const _model_2_Concat_output_0 = builder.concat([_model_2_Split_output_0, _model_2_Split_output_1, _model_2_m_0_Add_output_0], 1);
  // Conv
  const _model_2_cv2_conv_Conv_output_0 = builder.conv2d(_model_2_Concat_output_0, model_2_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_2_cv2_conv_bias });
  // Sigmoid
  const _model_2_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_2_cv2_conv_Conv_output_0);
  // Mul
  const _model_2_cv2_act_Mul_output_0 = builder.mul(_model_2_cv2_conv_Conv_output_0, _model_2_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_3_conv_Conv_output_0 = builder.conv2d(_model_2_cv2_act_Mul_output_0, model_3_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_3_conv_bias });
  // Sigmoid
  const _model_3_act_Sigmoid_output_0 = builder.sigmoid(_model_3_conv_Conv_output_0);
  // Mul
  const _model_3_act_Mul_output_0 = builder.mul(_model_3_conv_Conv_output_0, _model_3_act_Sigmoid_output_0);
  // Conv
  const _model_4_cv1_conv_Conv_output_0 = builder.conv2d(_model_3_act_Mul_output_0, model_4_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_4_cv1_conv_bias });
  // Sigmoid
  const _model_4_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_4_cv1_conv_Conv_output_0);
  // Mul
  const _model_4_cv1_act_Mul_output_0 = builder.mul(_model_4_cv1_conv_Conv_output_0, _model_4_cv1_act_Sigmoid_output_0);
  // Split
  const _model_4_Split_output_0_splits = builder.split(_model_4_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
  const _model_4_Split_output_0 = _model_4_Split_output_0_splits[0];
  const _model_4_Split_output_1 = _model_4_Split_output_0_splits[1];
  // Conv
  const _model_4_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_4_Split_output_1, model_4_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_4_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_4_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_4_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_4_m_0_cv1_act_Mul_output_0 = builder.mul(_model_4_m_0_cv1_conv_Conv_output_0, _model_4_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_4_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_4_m_0_cv1_act_Mul_output_0, model_4_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_4_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_4_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_4_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_4_m_0_cv2_act_Mul_output_0 = builder.mul(_model_4_m_0_cv2_conv_Conv_output_0, _model_4_m_0_cv2_act_Sigmoid_output_0);
  // Add
  const _model_4_m_0_Add_output_0 = builder.add(_model_4_Split_output_1, _model_4_m_0_cv2_act_Mul_output_0);
  // Conv
  const _model_4_m_1_cv1_conv_Conv_output_0 = builder.conv2d(_model_4_m_0_Add_output_0, model_4_m_1_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_4_m_1_cv1_conv_bias });
  // Sigmoid
  const _model_4_m_1_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_4_m_1_cv1_conv_Conv_output_0);
  // Mul
  const _model_4_m_1_cv1_act_Mul_output_0 = builder.mul(_model_4_m_1_cv1_conv_Conv_output_0, _model_4_m_1_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_4_m_1_cv2_conv_Conv_output_0 = builder.conv2d(_model_4_m_1_cv1_act_Mul_output_0, model_4_m_1_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_4_m_1_cv2_conv_bias });
  // Sigmoid
  const _model_4_m_1_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_4_m_1_cv2_conv_Conv_output_0);
  // Mul
  const _model_4_m_1_cv2_act_Mul_output_0 = builder.mul(_model_4_m_1_cv2_conv_Conv_output_0, _model_4_m_1_cv2_act_Sigmoid_output_0);
  // Add
  const _model_4_m_1_Add_output_0 = builder.add(_model_4_m_0_Add_output_0, _model_4_m_1_cv2_act_Mul_output_0);
  // Concat
  const _model_4_Concat_output_0 = builder.concat([_model_4_Split_output_0, _model_4_Split_output_1, _model_4_m_0_Add_output_0, _model_4_m_1_Add_output_0], 1);
  // Conv
  const _model_4_cv2_conv_Conv_output_0 = builder.conv2d(_model_4_Concat_output_0, model_4_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_4_cv2_conv_bias });
  // Sigmoid
  const _model_4_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_4_cv2_conv_Conv_output_0);
  // Mul
  const _model_4_cv2_act_Mul_output_0 = builder.mul(_model_4_cv2_conv_Conv_output_0, _model_4_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_5_conv_Conv_output_0 = builder.conv2d(_model_4_cv2_act_Mul_output_0, model_5_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_5_conv_bias });
  // Sigmoid
  const _model_5_act_Sigmoid_output_0 = builder.sigmoid(_model_5_conv_Conv_output_0);
  // Mul
  const _model_5_act_Mul_output_0 = builder.mul(_model_5_conv_Conv_output_0, _model_5_act_Sigmoid_output_0);
  // Conv
  const _model_6_cv1_conv_Conv_output_0 = builder.conv2d(_model_5_act_Mul_output_0, model_6_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_6_cv1_conv_bias });
  // Sigmoid
  const _model_6_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_6_cv1_conv_Conv_output_0);
  // Mul
  const _model_6_cv1_act_Mul_output_0 = builder.mul(_model_6_cv1_conv_Conv_output_0, _model_6_cv1_act_Sigmoid_output_0);
  // Split
  const _model_6_Split_output_0_splits = builder.split(_model_6_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
  const _model_6_Split_output_0 = _model_6_Split_output_0_splits[0];
  const _model_6_Split_output_1 = _model_6_Split_output_0_splits[1];
  // Conv
  const _model_6_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_6_Split_output_1, model_6_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_6_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_6_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_6_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_6_m_0_cv1_act_Mul_output_0 = builder.mul(_model_6_m_0_cv1_conv_Conv_output_0, _model_6_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_6_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_6_m_0_cv1_act_Mul_output_0, model_6_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_6_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_6_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_6_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_6_m_0_cv2_act_Mul_output_0 = builder.mul(_model_6_m_0_cv2_conv_Conv_output_0, _model_6_m_0_cv2_act_Sigmoid_output_0);
  // Add
  const _model_6_m_0_Add_output_0 = builder.add(_model_6_Split_output_1, _model_6_m_0_cv2_act_Mul_output_0);
  // Conv
  const _model_6_m_1_cv1_conv_Conv_output_0 = builder.conv2d(_model_6_m_0_Add_output_0, model_6_m_1_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_6_m_1_cv1_conv_bias });
  // Sigmoid
  const _model_6_m_1_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_6_m_1_cv1_conv_Conv_output_0);
  // Mul
  const _model_6_m_1_cv1_act_Mul_output_0 = builder.mul(_model_6_m_1_cv1_conv_Conv_output_0, _model_6_m_1_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_6_m_1_cv2_conv_Conv_output_0 = builder.conv2d(_model_6_m_1_cv1_act_Mul_output_0, model_6_m_1_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_6_m_1_cv2_conv_bias });
  // Sigmoid
  const _model_6_m_1_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_6_m_1_cv2_conv_Conv_output_0);
  // Mul
  const _model_6_m_1_cv2_act_Mul_output_0 = builder.mul(_model_6_m_1_cv2_conv_Conv_output_0, _model_6_m_1_cv2_act_Sigmoid_output_0);
  // Add
  const _model_6_m_1_Add_output_0 = builder.add(_model_6_m_0_Add_output_0, _model_6_m_1_cv2_act_Mul_output_0);
  // Concat
  const _model_6_Concat_output_0 = builder.concat([_model_6_Split_output_0, _model_6_Split_output_1, _model_6_m_0_Add_output_0, _model_6_m_1_Add_output_0], 1);
  // Conv
  const _model_6_cv2_conv_Conv_output_0 = builder.conv2d(_model_6_Concat_output_0, model_6_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_6_cv2_conv_bias });
  // Sigmoid
  const _model_6_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_6_cv2_conv_Conv_output_0);
  // Mul
  const _model_6_cv2_act_Mul_output_0 = builder.mul(_model_6_cv2_conv_Conv_output_0, _model_6_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_7_conv_Conv_output_0 = builder.conv2d(_model_6_cv2_act_Mul_output_0, model_7_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_7_conv_bias });
  // Sigmoid
  const _model_7_act_Sigmoid_output_0 = builder.sigmoid(_model_7_conv_Conv_output_0);
  // Mul
  const _model_7_act_Mul_output_0 = builder.mul(_model_7_conv_Conv_output_0, _model_7_act_Sigmoid_output_0);
  // Conv
  const _model_8_cv1_conv_Conv_output_0 = builder.conv2d(_model_7_act_Mul_output_0, model_8_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_8_cv1_conv_bias });
  // Sigmoid
  const _model_8_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_8_cv1_conv_Conv_output_0);
  // Mul
  const _model_8_cv1_act_Mul_output_0 = builder.mul(_model_8_cv1_conv_Conv_output_0, _model_8_cv1_act_Sigmoid_output_0);
  // Split
  const _model_8_Split_output_0_splits = builder.split(_model_8_cv1_act_Mul_output_0, [256, 256], { axis: 1 });
  const _model_8_Split_output_0 = _model_8_Split_output_0_splits[0];
  const _model_8_Split_output_1 = _model_8_Split_output_0_splits[1];
  // Conv
  const _model_8_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_8_Split_output_1, model_8_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_8_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_8_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_8_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_8_m_0_cv1_act_Mul_output_0 = builder.mul(_model_8_m_0_cv1_conv_Conv_output_0, _model_8_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_8_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_8_m_0_cv1_act_Mul_output_0, model_8_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_8_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_8_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_8_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_8_m_0_cv2_act_Mul_output_0 = builder.mul(_model_8_m_0_cv2_conv_Conv_output_0, _model_8_m_0_cv2_act_Sigmoid_output_0);
  // Add
  const _model_8_m_0_Add_output_0 = builder.add(_model_8_Split_output_1, _model_8_m_0_cv2_act_Mul_output_0);
  // Concat
  const _model_8_Concat_output_0 = builder.concat([_model_8_Split_output_0, _model_8_Split_output_1, _model_8_m_0_Add_output_0], 1);
  // Conv
  const _model_8_cv2_conv_Conv_output_0 = builder.conv2d(_model_8_Concat_output_0, model_8_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_8_cv2_conv_bias });
  // Sigmoid
  const _model_8_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_8_cv2_conv_Conv_output_0);
  // Mul
  const _model_8_cv2_act_Mul_output_0 = builder.mul(_model_8_cv2_conv_Conv_output_0, _model_8_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_9_cv1_conv_Conv_output_0 = builder.conv2d(_model_8_cv2_act_Mul_output_0, model_9_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_9_cv1_conv_bias });
  // Sigmoid
  const _model_9_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_9_cv1_conv_Conv_output_0);
  // Mul
  const _model_9_cv1_act_Mul_output_0 = builder.mul(_model_9_cv1_conv_Conv_output_0, _model_9_cv1_act_Sigmoid_output_0);
  // MaxPool
  const _model_9_m_MaxPool_output_0 = builder.maxPool2d(_model_9_cv1_act_Mul_output_0, { windowDimensions: [5, 5], padding: [2, 2, 2, 2] });
  // MaxPool
  const _model_9_m_1_MaxPool_output_0 = builder.maxPool2d(_model_9_m_MaxPool_output_0, { windowDimensions: [5, 5], padding: [2, 2, 2, 2] });
  // MaxPool
  const _model_9_m_2_MaxPool_output_0 = builder.maxPool2d(_model_9_m_1_MaxPool_output_0, { windowDimensions: [5, 5], padding: [2, 2, 2, 2] });
  // Concat
  const _model_9_Concat_output_0 = builder.concat([_model_9_cv1_act_Mul_output_0, _model_9_m_MaxPool_output_0, _model_9_m_1_MaxPool_output_0, _model_9_m_2_MaxPool_output_0], 1);
  // Conv
  const _model_9_cv2_conv_Conv_output_0 = builder.conv2d(_model_9_Concat_output_0, model_9_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_9_cv2_conv_bias });
  // Sigmoid
  const _model_9_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_9_cv2_conv_Conv_output_0);
  // Mul
  const _model_9_cv2_act_Mul_output_0 = builder.mul(_model_9_cv2_conv_Conv_output_0, _model_9_cv2_act_Sigmoid_output_0);
  // Resize
  const _model_10_Resize_output_0 = builder.resample2d(_model_9_cv2_act_Mul_output_0, { mode: 'nearest-neighbor', scales: [2, 2], axes: [2, 3] });
  // Concat
  const _model_11_Concat_output_0 = builder.concat([_model_10_Resize_output_0, _model_6_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_12_cv1_conv_Conv_output_0 = builder.conv2d(_model_11_Concat_output_0, model_12_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_12_cv1_conv_bias });
  // Sigmoid
  const _model_12_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_12_cv1_conv_Conv_output_0);
  // Mul
  const _model_12_cv1_act_Mul_output_0 = builder.mul(_model_12_cv1_conv_Conv_output_0, _model_12_cv1_act_Sigmoid_output_0);
  // Split
  const _model_12_Split_output_0_splits = builder.split(_model_12_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
  const _model_12_Split_output_0 = _model_12_Split_output_0_splits[0];
  const _model_12_Split_output_1 = _model_12_Split_output_0_splits[1];
  // Conv
  const _model_12_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_12_Split_output_1, model_12_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_12_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_12_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_12_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_12_m_0_cv1_act_Mul_output_0 = builder.mul(_model_12_m_0_cv1_conv_Conv_output_0, _model_12_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_12_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_12_m_0_cv1_act_Mul_output_0, model_12_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_12_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_12_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_12_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_12_m_0_cv2_act_Mul_output_0 = builder.mul(_model_12_m_0_cv2_conv_Conv_output_0, _model_12_m_0_cv2_act_Sigmoid_output_0);
  // Concat
  const _model_12_Concat_output_0 = builder.concat([_model_12_Split_output_0, _model_12_Split_output_1, _model_12_m_0_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_12_cv2_conv_Conv_output_0 = builder.conv2d(_model_12_Concat_output_0, model_12_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_12_cv2_conv_bias });
  // Sigmoid
  const _model_12_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_12_cv2_conv_Conv_output_0);
  // Mul
  const _model_12_cv2_act_Mul_output_0 = builder.mul(_model_12_cv2_conv_Conv_output_0, _model_12_cv2_act_Sigmoid_output_0);
  // Resize
  const _model_13_Resize_output_0 = builder.resample2d(_model_12_cv2_act_Mul_output_0, { mode: 'nearest-neighbor', scales: [2, 2], axes: [2, 3] });
  // Concat
  const _model_14_Concat_output_0 = builder.concat([_model_13_Resize_output_0, _model_4_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_15_cv1_conv_Conv_output_0 = builder.conv2d(_model_14_Concat_output_0, model_15_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_15_cv1_conv_bias });
  // Sigmoid
  const _model_15_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_15_cv1_conv_Conv_output_0);
  // Mul
  const _model_15_cv1_act_Mul_output_0 = builder.mul(_model_15_cv1_conv_Conv_output_0, _model_15_cv1_act_Sigmoid_output_0);
  // Split
  const _model_15_Split_output_0_splits = builder.split(_model_15_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
  const _model_15_Split_output_0 = _model_15_Split_output_0_splits[0];
  const _model_15_Split_output_1 = _model_15_Split_output_0_splits[1];
  // Conv
  const _model_15_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_15_Split_output_1, model_15_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_15_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_15_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_15_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_15_m_0_cv1_act_Mul_output_0 = builder.mul(_model_15_m_0_cv1_conv_Conv_output_0, _model_15_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_15_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_15_m_0_cv1_act_Mul_output_0, model_15_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_15_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_15_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_15_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_15_m_0_cv2_act_Mul_output_0 = builder.mul(_model_15_m_0_cv2_conv_Conv_output_0, _model_15_m_0_cv2_act_Sigmoid_output_0);
  // Concat
  const _model_15_Concat_output_0 = builder.concat([_model_15_Split_output_0, _model_15_Split_output_1, _model_15_m_0_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_15_cv2_conv_Conv_output_0 = builder.conv2d(_model_15_Concat_output_0, model_15_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_15_cv2_conv_bias });
  // Sigmoid
  const _model_15_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_15_cv2_conv_Conv_output_0);
  // Mul
  const _model_15_cv2_act_Mul_output_0 = builder.mul(_model_15_cv2_conv_Conv_output_0, _model_15_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_16_conv_Conv_output_0 = builder.conv2d(_model_15_cv2_act_Mul_output_0, model_16_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_16_conv_bias });
  // Conv
  const _model_22_cv2_0_cv2_0_0_conv_Conv_output_0 = builder.conv2d(_model_15_cv2_act_Mul_output_0, model_22_cv2_0_0_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv2_0_0_conv_bias });
  // Conv
  const _model_22_cv3_0_cv3_0_0_conv_Conv_output_0 = builder.conv2d(_model_15_cv2_act_Mul_output_0, model_22_cv3_0_0_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv3_0_0_conv_bias });
  // Sigmoid
  const _model_16_act_Sigmoid_output_0 = builder.sigmoid(_model_16_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv2_0_cv2_0_0_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv3_0_cv3_0_0_conv_Conv_output_0);
  // Mul
  const _model_16_act_Mul_output_0 = builder.mul(_model_16_conv_Conv_output_0, _model_16_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv2_0_cv2_0_0_act_Mul_output_0 = builder.mul(_model_22_cv2_0_cv2_0_0_conv_Conv_output_0, _model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv3_0_cv3_0_0_act_Mul_output_0 = builder.mul(_model_22_cv3_0_cv3_0_0_conv_Conv_output_0, _model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0);
  // Concat
  const _model_17_Concat_output_0 = builder.concat([_model_16_act_Mul_output_0, _model_12_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_22_cv2_0_cv2_0_1_conv_Conv_output_0 = builder.conv2d(_model_22_cv2_0_cv2_0_0_act_Mul_output_0, model_22_cv2_0_1_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv2_0_1_conv_bias });
  // Conv
  const _model_22_cv3_0_cv3_0_1_conv_Conv_output_0 = builder.conv2d(_model_22_cv3_0_cv3_0_0_act_Mul_output_0, model_22_cv3_0_1_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv3_0_1_conv_bias });
  // Conv
  const _model_18_cv1_conv_Conv_output_0 = builder.conv2d(_model_17_Concat_output_0, model_18_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_18_cv1_conv_bias });
  // Sigmoid
  const _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv2_0_cv2_0_1_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv3_0_cv3_0_1_conv_Conv_output_0);
  // Sigmoid
  const _model_18_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_18_cv1_conv_Conv_output_0);
  // Mul
  const _model_22_cv2_0_cv2_0_1_act_Mul_output_0 = builder.mul(_model_22_cv2_0_cv2_0_1_conv_Conv_output_0, _model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv3_0_cv3_0_1_act_Mul_output_0 = builder.mul(_model_22_cv3_0_cv3_0_1_conv_Conv_output_0, _model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0);
  // Mul
  const _model_18_cv1_act_Mul_output_0 = builder.mul(_model_18_cv1_conv_Conv_output_0, _model_18_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_22_cv2_0_cv2_0_2_Conv_output_0 = builder.conv2d(_model_22_cv2_0_cv2_0_1_act_Mul_output_0, model_22_cv2_0_2_weight, { padding: [0, 0, 0, 0], bias: model_22_cv2_0_2_bias });
  // Conv
  const _model_22_cv3_0_cv3_0_2_Conv_output_0 = builder.conv2d(_model_22_cv3_0_cv3_0_1_act_Mul_output_0, model_22_cv3_0_2_weight, { padding: [0, 0, 0, 0], bias: model_22_cv3_0_2_bias });
  // Split
  const _model_18_Split_output_0_splits = builder.split(_model_18_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
  const _model_18_Split_output_0 = _model_18_Split_output_0_splits[0];
  const _model_18_Split_output_1 = _model_18_Split_output_0_splits[1];
  // Reshape
  const _model_22_Reshape_output_0 = builder.reshape(_model_22_cv2_0_cv2_0_2_Conv_output_0, [1, 64, 6400]);
  // Reshape
  const _model_22_Reshape_3_output_0 = builder.reshape(_model_22_cv3_0_cv3_0_2_Conv_output_0, [1, 80, 6400]);
  // Conv
  const _model_18_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_18_Split_output_1, model_18_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_18_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_18_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_18_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_18_m_0_cv1_act_Mul_output_0 = builder.mul(_model_18_m_0_cv1_conv_Conv_output_0, _model_18_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_18_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_18_m_0_cv1_act_Mul_output_0, model_18_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_18_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_18_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_18_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_18_m_0_cv2_act_Mul_output_0 = builder.mul(_model_18_m_0_cv2_conv_Conv_output_0, _model_18_m_0_cv2_act_Sigmoid_output_0);
  // Concat
  const _model_18_Concat_output_0 = builder.concat([_model_18_Split_output_0, _model_18_Split_output_1, _model_18_m_0_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_18_cv2_conv_Conv_output_0 = builder.conv2d(_model_18_Concat_output_0, model_18_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_18_cv2_conv_bias });
  // Sigmoid
  const _model_18_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_18_cv2_conv_Conv_output_0);
  // Mul
  const _model_18_cv2_act_Mul_output_0 = builder.mul(_model_18_cv2_conv_Conv_output_0, _model_18_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_19_conv_Conv_output_0 = builder.conv2d(_model_18_cv2_act_Mul_output_0, model_19_conv_weight, { padding: [1, 1, 1, 1], strides: [2, 2], bias: model_19_conv_bias });
  // Conv
  const _model_22_cv2_1_cv2_1_0_conv_Conv_output_0 = builder.conv2d(_model_18_cv2_act_Mul_output_0, model_22_cv2_1_0_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv2_1_0_conv_bias });
  // Conv
  const _model_22_cv3_1_cv3_1_0_conv_Conv_output_0 = builder.conv2d(_model_18_cv2_act_Mul_output_0, model_22_cv3_1_0_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv3_1_0_conv_bias });
  // Sigmoid
  const _model_19_act_Sigmoid_output_0 = builder.sigmoid(_model_19_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv2_1_cv2_1_0_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv3_1_cv3_1_0_conv_Conv_output_0);
  // Mul
  const _model_19_act_Mul_output_0 = builder.mul(_model_19_conv_Conv_output_0, _model_19_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv2_1_cv2_1_0_act_Mul_output_0 = builder.mul(_model_22_cv2_1_cv2_1_0_conv_Conv_output_0, _model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv3_1_cv3_1_0_act_Mul_output_0 = builder.mul(_model_22_cv3_1_cv3_1_0_conv_Conv_output_0, _model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0);
  // Concat
  const _model_20_Concat_output_0 = builder.concat([_model_19_act_Mul_output_0, _model_9_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_22_cv2_1_cv2_1_1_conv_Conv_output_0 = builder.conv2d(_model_22_cv2_1_cv2_1_0_act_Mul_output_0, model_22_cv2_1_1_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv2_1_1_conv_bias });
  // Conv
  const _model_22_cv3_1_cv3_1_1_conv_Conv_output_0 = builder.conv2d(_model_22_cv3_1_cv3_1_0_act_Mul_output_0, model_22_cv3_1_1_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv3_1_1_conv_bias });
  // Conv
  const _model_21_cv1_conv_Conv_output_0 = builder.conv2d(_model_20_Concat_output_0, model_21_cv1_conv_weight, { padding: [0, 0, 0, 0], bias: model_21_cv1_conv_bias });
  // Sigmoid
  const _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv2_1_cv2_1_1_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv3_1_cv3_1_1_conv_Conv_output_0);
  // Sigmoid
  const _model_21_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_21_cv1_conv_Conv_output_0);
  // Mul
  const _model_22_cv2_1_cv2_1_1_act_Mul_output_0 = builder.mul(_model_22_cv2_1_cv2_1_1_conv_Conv_output_0, _model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv3_1_cv3_1_1_act_Mul_output_0 = builder.mul(_model_22_cv3_1_cv3_1_1_conv_Conv_output_0, _model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0);
  // Mul
  const _model_21_cv1_act_Mul_output_0 = builder.mul(_model_21_cv1_conv_Conv_output_0, _model_21_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_22_cv2_1_cv2_1_2_Conv_output_0 = builder.conv2d(_model_22_cv2_1_cv2_1_1_act_Mul_output_0, model_22_cv2_1_2_weight, { padding: [0, 0, 0, 0], bias: model_22_cv2_1_2_bias });
  // Conv
  const _model_22_cv3_1_cv3_1_2_Conv_output_0 = builder.conv2d(_model_22_cv3_1_cv3_1_1_act_Mul_output_0, model_22_cv3_1_2_weight, { padding: [0, 0, 0, 0], bias: model_22_cv3_1_2_bias });
  // Split
  const _model_21_Split_output_0_splits = builder.split(_model_21_cv1_act_Mul_output_0, [256, 256], { axis: 1 });
  const _model_21_Split_output_0 = _model_21_Split_output_0_splits[0];
  const _model_21_Split_output_1 = _model_21_Split_output_0_splits[1];
  // Reshape
  const _model_22_Reshape_1_output_0 = builder.reshape(_model_22_cv2_1_cv2_1_2_Conv_output_0, [1, 64, 1600]);
  // Reshape
  const _model_22_Reshape_4_output_0 = builder.reshape(_model_22_cv3_1_cv3_1_2_Conv_output_0, [1, 80, 1600]);
  // Conv
  const _model_21_m_0_cv1_conv_Conv_output_0 = builder.conv2d(_model_21_Split_output_1, model_21_m_0_cv1_conv_weight, { padding: [1, 1, 1, 1], bias: model_21_m_0_cv1_conv_bias });
  // Sigmoid
  const _model_21_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(_model_21_m_0_cv1_conv_Conv_output_0);
  // Mul
  const _model_21_m_0_cv1_act_Mul_output_0 = builder.mul(_model_21_m_0_cv1_conv_Conv_output_0, _model_21_m_0_cv1_act_Sigmoid_output_0);
  // Conv
  const _model_21_m_0_cv2_conv_Conv_output_0 = builder.conv2d(_model_21_m_0_cv1_act_Mul_output_0, model_21_m_0_cv2_conv_weight, { padding: [1, 1, 1, 1], bias: model_21_m_0_cv2_conv_bias });
  // Sigmoid
  const _model_21_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_21_m_0_cv2_conv_Conv_output_0);
  // Mul
  const _model_21_m_0_cv2_act_Mul_output_0 = builder.mul(_model_21_m_0_cv2_conv_Conv_output_0, _model_21_m_0_cv2_act_Sigmoid_output_0);
  // Concat
  const _model_21_Concat_output_0 = builder.concat([_model_21_Split_output_0, _model_21_Split_output_1, _model_21_m_0_cv2_act_Mul_output_0], 1);
  // Conv
  const _model_21_cv2_conv_Conv_output_0 = builder.conv2d(_model_21_Concat_output_0, model_21_cv2_conv_weight, { padding: [0, 0, 0, 0], bias: model_21_cv2_conv_bias });
  // Sigmoid
  const _model_21_cv2_act_Sigmoid_output_0 = builder.sigmoid(_model_21_cv2_conv_Conv_output_0);
  // Mul
  const _model_21_cv2_act_Mul_output_0 = builder.mul(_model_21_cv2_conv_Conv_output_0, _model_21_cv2_act_Sigmoid_output_0);
  // Conv
  const _model_22_cv2_2_cv2_2_0_conv_Conv_output_0 = builder.conv2d(_model_21_cv2_act_Mul_output_0, model_22_cv2_2_0_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv2_2_0_conv_bias });
  // Conv
  const _model_22_cv3_2_cv3_2_0_conv_Conv_output_0 = builder.conv2d(_model_21_cv2_act_Mul_output_0, model_22_cv3_2_0_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv3_2_0_conv_bias });
  // Sigmoid
  const _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv2_2_cv2_2_0_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv3_2_cv3_2_0_conv_Conv_output_0);
  // Mul
  const _model_22_cv2_2_cv2_2_0_act_Mul_output_0 = builder.mul(_model_22_cv2_2_cv2_2_0_conv_Conv_output_0, _model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv3_2_cv3_2_0_act_Mul_output_0 = builder.mul(_model_22_cv3_2_cv3_2_0_conv_Conv_output_0, _model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0);
  // Conv
  const _model_22_cv2_2_cv2_2_1_conv_Conv_output_0 = builder.conv2d(_model_22_cv2_2_cv2_2_0_act_Mul_output_0, model_22_cv2_2_1_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv2_2_1_conv_bias });
  // Conv
  const _model_22_cv3_2_cv3_2_1_conv_Conv_output_0 = builder.conv2d(_model_22_cv3_2_cv3_2_0_act_Mul_output_0, model_22_cv3_2_1_conv_weight, { padding: [1, 1, 1, 1], bias: model_22_cv3_2_1_conv_bias });
  // Sigmoid
  const _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv2_2_cv2_2_1_conv_Conv_output_0);
  // Sigmoid
  const _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0 = builder.sigmoid(_model_22_cv3_2_cv3_2_1_conv_Conv_output_0);
  // Mul
  const _model_22_cv2_2_cv2_2_1_act_Mul_output_0 = builder.mul(_model_22_cv2_2_cv2_2_1_conv_Conv_output_0, _model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0);
  // Mul
  const _model_22_cv3_2_cv3_2_1_act_Mul_output_0 = builder.mul(_model_22_cv3_2_cv3_2_1_conv_Conv_output_0, _model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0);
  // Conv
  const _model_22_cv2_2_cv2_2_2_Conv_output_0 = builder.conv2d(_model_22_cv2_2_cv2_2_1_act_Mul_output_0, model_22_cv2_2_2_weight, { padding: [0, 0, 0, 0], bias: model_22_cv2_2_2_bias });
  // Conv
  const _model_22_cv3_2_cv3_2_2_Conv_output_0 = builder.conv2d(_model_22_cv3_2_cv3_2_1_act_Mul_output_0, model_22_cv3_2_2_weight, { padding: [0, 0, 0, 0], bias: model_22_cv3_2_2_bias });
  // Reshape
  const _model_22_Reshape_2_output_0 = builder.reshape(_model_22_cv2_2_cv2_2_2_Conv_output_0, [1, 64, 400]);
  // Reshape
  const _model_22_Reshape_5_output_0 = builder.reshape(_model_22_cv3_2_cv3_2_2_Conv_output_0, [1, 80, 400]);
  // Concat
  const _model_22_Concat_output_0 = builder.concat([_model_22_Reshape_output_0, _model_22_Reshape_1_output_0, _model_22_Reshape_2_output_0], 2);
  // Concat
  const _model_22_Concat_1_output_0 = builder.concat([_model_22_Reshape_3_output_0, _model_22_Reshape_4_output_0, _model_22_Reshape_5_output_0], 2);
  // Reshape
  const _model_22_dfl_Reshape_output_0 = builder.reshape(_model_22_Concat_output_0, [1, 4, 16, 8400]);
  // Sigmoid
  const _model_22_Sigmoid_output_0 = builder.sigmoid(_model_22_Concat_1_output_0);
  // Transpose
  const _model_22_dfl_Transpose_output_0 = builder.transpose(_model_22_dfl_Reshape_output_0, { permutation: [0, 2, 1, 3] });
  // Softmax
  const _model_22_dfl_Softmax_output_0 = builder.softmax(_model_22_dfl_Transpose_output_0, 1);
  // Conv
  const _model_22_dfl_conv_Conv_output_0 = builder.conv2d(_model_22_dfl_Softmax_output_0, model_22_dfl_conv_weight, { padding: [0, 0, 0, 0] });
  // Reshape
  const _model_22_dfl_Reshape_1_output_0 = builder.reshape(_model_22_dfl_conv_Conv_output_0, [1, 4, 8400]);
  // Slice
  const _model_22_Slice_output_0 = builder.slice(_model_22_dfl_Reshape_1_output_0, [0, 0, 0], [1, 2, 8400]);
  // Slice
  const _model_22_Slice_1_output_0 = builder.slice(_model_22_dfl_Reshape_1_output_0, [0, 2, 0], [1, 2, 8400]);
  // Sub
  const _model_22_Sub_output_0 = builder.sub(_model_22_Constant_12_output_0, _model_22_Slice_output_0);
  // Add
  const _model_22_Add_1_output_0 = builder.add(_model_22_Constant_13_output_0, _model_22_Slice_1_output_0);
  // Add
  const _model_22_Add_2_output_0 = builder.add(_model_22_Sub_output_0, _model_22_Add_1_output_0);
  // Sub
  const _model_22_Sub_1_output_0 = builder.sub(_model_22_Add_1_output_0, _model_22_Sub_output_0);
  // Div
  const _model_22_Div_1_output_0 = builder.div(_model_22_Add_2_output_0, _model_22_Constant_14_output_0);
  // Concat
  const _model_22_Concat_2_output_0 = builder.concat([_model_22_Div_1_output_0, _model_22_Sub_1_output_0], 1);
  // Mul
  const _model_22_Mul_2_output_0 = builder.mul(_model_22_Concat_2_output_0, _model_22_Constant_15_output_0);
  // Concat
  const output0 = builder.concat([_model_22_Mul_2_output_0, _model_22_Sigmoid_output_0], 1);

  // Build graph
  const namedOutputs = {};
  namedOutputs['output0'] = output0;
  // Capture actual output operand shapes (may differ from metadata for dynamic models)
  const outputShapes = {};
  for (const [name, operand] of Object.entries(namedOutputs)) {
    outputShapes[name] = Array.from(operand.shape);
  }
  return { graph: await builder.build(namedOutputs), outputShapes };
}

async function main(deviceType = 'cpu') {
  if (!navigator.ml) throw new Error('WebNN is not supported in this browser.');
  const context = await navigator.ml.createContext({ deviceType });
  const weights = await WeightsFile.load('yolov8s.weights', 'yolov8s.manifest.json');
  const buildStart = performance.now();
  const graph = await buildGraph(context, weights);
  console.log(`Graph build: ${(performance.now() - buildStart).toFixed(2)}ms on ${deviceType.toUpperCase()}`)

  // Create input tensors
  const inputData_images = new Float32Array(1228800); // [1,3,640,640]
  const inputTensor_images = await context.createTensor({ dataType: 'float32', shape: [1,3,640,640], writable: true });
  context.writeTensor(inputTensor_images, inputData_images);

  // Create output tensors — use actual shapes from built graph
  const outputTensor_output0 = await context.createTensor({ dataType: 'float32', shape: graph.outputShapes['output0'], readable: true });

  const inputs = {
    'images': inputTensor_images,
  };
  const outputs = {
    'output0': outputTensor_output0,
  };

  const start = performance.now();
  context.dispatch(graph.graph, inputs, outputs);

  // Read results
  const result_output0 = new Float32Array(await context.readTensor(outputTensor_output0));
  console.log(`Inference: ${(performance.now() - start).toFixed(2)}ms (1 run) on ${deviceType.toUpperCase()}`)

  // Benchmark: 50 runs
  const NUM_RUNS = 50;
  const runTimes = [];
  for (let i = 0; i < NUM_RUNS; i++) {
    const t0 = performance.now();
    context.dispatch(graph.graph, inputs, outputs);
    new Float32Array(await context.readTensor(outputTensor_output0));
    runTimes.push(performance.now() - t0);
  }
  const avgTime = (runTimes.reduce((a, b) => a + b, 0) / NUM_RUNS).toFixed(2);
  const sorted = [...runTimes].sort((a, b) => a - b);
  const medianTime = (NUM_RUNS % 2 ? sorted[NUM_RUNS >> 1] : (sorted[NUM_RUNS / 2 - 1] + sorted[NUM_RUNS / 2]) / 2).toFixed(2);
  console.log(`Inference: ${avgTime}ms (average \u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)
  console.log(`Inference: ${medianTime}ms (median \u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)

  return { 'output0': result_output0 };
}
