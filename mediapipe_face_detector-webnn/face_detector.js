/**
 * Build WebNN graph for model: face_detector
 * Source format: tflite
 */
async function buildGraph(context, weights) {
  const builder = new MLGraphBuilder(context);

  // Graph inputs
  const input = builder.input('input', { dataType: 'float32', shape: [1,128,128,3] });

  // Constants (loaded from weights file)
  const conv2d_Kernel = (() => {
    const sl = weights.getSlice('conv2d/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [24,5,5,3] }, new Float16Array(buf));
  })();
  const conv2d_Bias = (() => {
    const sl = weights.getSlice('conv2d/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [24] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,24] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [24] }, new Float16Array(buf));
  })();
  const conv2d_1_Kernel = (() => {
    const sl = weights.getSlice('conv2d_1/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [24,1,1,24] }, new Float16Array(buf));
  })();
  const conv2d_1_Bias = (() => {
    const sl = weights.getSlice('conv2d_1/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [24] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_1_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_1/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,24] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_1_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_1/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [24] }, new Float16Array(buf));
  })();
  const conv2d_2_Kernel = (() => {
    const sl = weights.getSlice('conv2d_2/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [28,1,1,24] }, new Float16Array(buf));
  })();
  const conv2d_2_Bias = (() => {
    const sl = weights.getSlice('conv2d_2/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [28] }, new Float16Array(buf));
  })();
  const channel_padding_Paddings = (() => {
    const sl = weights.getSlice('channel_padding/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_2_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_2/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,28] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_2_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_2/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [28] }, new Float16Array(buf));
  })();
  const conv2d_3_Kernel = (() => {
    const sl = weights.getSlice('conv2d_3/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [32,1,1,28] }, new Float16Array(buf));
  })();
  const conv2d_3_Bias = (() => {
    const sl = weights.getSlice('conv2d_3/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [32] }, new Float16Array(buf));
  })();
  const channel_padding_1_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_1/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_3_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_3/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,32] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_3_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_3/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [32] }, new Float16Array(buf));
  })();
  const conv2d_4_Kernel = (() => {
    const sl = weights.getSlice('conv2d_4/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [36,1,1,32] }, new Float16Array(buf));
  })();
  const conv2d_4_Bias = (() => {
    const sl = weights.getSlice('conv2d_4/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [36] }, new Float16Array(buf));
  })();
  const channel_padding_2_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_2/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_4_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_4/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,36] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_4_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_4/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [36] }, new Float16Array(buf));
  })();
  const conv2d_5_Kernel = (() => {
    const sl = weights.getSlice('conv2d_5/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [42,1,1,36] }, new Float16Array(buf));
  })();
  const conv2d_5_Bias = (() => {
    const sl = weights.getSlice('conv2d_5/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [42] }, new Float16Array(buf));
  })();
  const channel_padding_3_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_3/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_5_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_5/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,42] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_5_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_5/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [42] }, new Float16Array(buf));
  })();
  const conv2d_6_Kernel = (() => {
    const sl = weights.getSlice('conv2d_6/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [48,1,1,42] }, new Float16Array(buf));
  })();
  const conv2d_6_Bias = (() => {
    const sl = weights.getSlice('conv2d_6/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [48] }, new Float16Array(buf));
  })();
  const channel_padding_4_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_4/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_6_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_6/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,48] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_6_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_6/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [48] }, new Float16Array(buf));
  })();
  const conv2d_7_Kernel = (() => {
    const sl = weights.getSlice('conv2d_7/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [56,1,1,48] }, new Float16Array(buf));
  })();
  const conv2d_7_Bias = (() => {
    const sl = weights.getSlice('conv2d_7/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [56] }, new Float16Array(buf));
  })();
  const channel_padding_5_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_5/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_7_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_7/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,56] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_7_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_7/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [56] }, new Float16Array(buf));
  })();
  const conv2d_8_Kernel = (() => {
    const sl = weights.getSlice('conv2d_8/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [64,1,1,56] }, new Float16Array(buf));
  })();
  const conv2d_8_Bias = (() => {
    const sl = weights.getSlice('conv2d_8/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [64] }, new Float16Array(buf));
  })();
  const channel_padding_6_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_6/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_8_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_8/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,64] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_8_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_8/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [64] }, new Float16Array(buf));
  })();
  const conv2d_9_Kernel = (() => {
    const sl = weights.getSlice('conv2d_9/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [72,1,1,64] }, new Float16Array(buf));
  })();
  const conv2d_9_Bias = (() => {
    const sl = weights.getSlice('conv2d_9/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [72] }, new Float16Array(buf));
  })();
  const channel_padding_7_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_7/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_9_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_9/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,72] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_9_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_9/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [72] }, new Float16Array(buf));
  })();
  const conv2d_10_Kernel = (() => {
    const sl = weights.getSlice('conv2d_10/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [80,1,1,72] }, new Float16Array(buf));
  })();
  const conv2d_10_Bias = (() => {
    const sl = weights.getSlice('conv2d_10/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [80] }, new Float16Array(buf));
  })();
  const channel_padding_8_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_8/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_10_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_10/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,80] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_10_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_10/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [80] }, new Float16Array(buf));
  })();
  const conv2d_11_Kernel = (() => {
    const sl = weights.getSlice('conv2d_11/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [88,1,1,80] }, new Float16Array(buf));
  })();
  const conv2d_11_Bias = (() => {
    const sl = weights.getSlice('conv2d_11/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [88] }, new Float16Array(buf));
  })();
  const channel_padding_9_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_9/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_11_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_11/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,88] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_11_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_11/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [88] }, new Float16Array(buf));
  })();
  const conv2d_12_Kernel = (() => {
    const sl = weights.getSlice('conv2d_12/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96,1,1,88] }, new Float16Array(buf));
  })();
  const conv2d_12_Bias = (() => {
    const sl = weights.getSlice('conv2d_12/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const channel_padding_10_Paddings = (() => {
    const sl = weights.getSlice('channel_padding_10/Paddings');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'int32', shape: [4,2] }, new Int32Array(buf));
  })();
  const depthwise_conv2d_12_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_12/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_12_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_12/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const conv2d_13_Kernel = (() => {
    const sl = weights.getSlice('conv2d_13/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96,1,1,96] }, new Float16Array(buf));
  })();
  const conv2d_13_Bias = (() => {
    const sl = weights.getSlice('conv2d_13/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_13_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_13/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_13_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_13/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const conv2d_14_Kernel = (() => {
    const sl = weights.getSlice('conv2d_14/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96,1,1,96] }, new Float16Array(buf));
  })();
  const conv2d_14_Bias = (() => {
    const sl = weights.getSlice('conv2d_14/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_14_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_14/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_14_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_14/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const conv2d_15_Kernel = (() => {
    const sl = weights.getSlice('conv2d_15/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96,1,1,96] }, new Float16Array(buf));
  })();
  const conv2d_15_Bias = (() => {
    const sl = weights.getSlice('conv2d_15/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_15_Kernel = (() => {
    const sl = weights.getSlice('depthwise_conv2d_15/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [1,3,3,96] }, new Float16Array(buf));
  })();
  const depthwise_conv2d_15_Bias = (() => {
    const sl = weights.getSlice('depthwise_conv2d_15/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const conv2d_16_Kernel = (() => {
    const sl = weights.getSlice('conv2d_16/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96,1,1,96] }, new Float16Array(buf));
  })();
  const conv2d_16_Bias = (() => {
    const sl = weights.getSlice('conv2d_16/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();
  const classificator_8_Kernel = (() => {
    const sl = weights.getSlice('classificator_8/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [2,1,1,88] }, new Float16Array(buf));
  })();
  const classificator_8_Bias = (() => {
    const sl = weights.getSlice('classificator_8/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [2] }, new Float16Array(buf));
  })();
  const classificator_16_Kernel = (() => {
    const sl = weights.getSlice('classificator_16/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [6,1,1,96] }, new Float16Array(buf));
  })();
  const classificator_16_Bias = (() => {
    const sl = weights.getSlice('classificator_16/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [6] }, new Float16Array(buf));
  })();
  const regressor_8_Kernel = (() => {
    const sl = weights.getSlice('regressor_8/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [32,1,1,88] }, new Float16Array(buf));
  })();
  const regressor_8_Bias = (() => {
    const sl = weights.getSlice('regressor_8/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [32] }, new Float16Array(buf));
  })();
  const regressor_16_Kernel = (() => {
    const sl = weights.getSlice('regressor_16/Kernel');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96,1,1,96] }, new Float16Array(buf));
  })();
  const regressor_16_Bias = (() => {
    const sl = weights.getSlice('regressor_16/Bias');
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    return builder.constant({ dataType: 'float16', shape: [96] }, new Float16Array(buf));
  })();

  // Graph operations
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_Bias_dequantize = builder.cast(conv2d_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_Kernel_dequantize = builder.cast(conv2d_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: SAME
  const conv2d_raw = builder.conv2d(input, conv2d_Kernel_dequantize, { strides: [2,2], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_Bias_dequantize });
  const conv2d = conv2d_raw;
  // RELU
  const activation = builder.relu(conv2d);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_Bias_dequantize = builder.cast(depthwise_conv2d_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_Kernel_dequantize = builder.cast(depthwise_conv2d_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_raw = builder.conv2d(activation, depthwise_conv2d_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_Kernel_dequantize.shape[3], bias: depthwise_conv2d_Bias_dequantize });
  const depthwise_conv2d = depthwise_conv2d_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_1_Bias_dequantize = builder.cast(conv2d_1_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_1_Kernel_dequantize = builder.cast(conv2d_1_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_1_raw = builder.conv2d(depthwise_conv2d, conv2d_1_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_1_Bias_dequantize });
  const conv2d_1 = conv2d_1_raw;
  // ADD
  // ADD
  const add__xeno_compat__1_raw = builder.add(activation, conv2d_1);
  const add__xeno_compat__1 = add__xeno_compat__1_raw;
  // RELU
  const activation_1 = builder.relu(add__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_1_Kernel_dequantize = builder.cast(depthwise_conv2d_1_Kernel, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_1_Bias_dequantize = builder.cast(depthwise_conv2d_1_Bias, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_1_raw = builder.conv2d(activation_1, depthwise_conv2d_1_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_1_Kernel_dequantize.shape[3], bias: depthwise_conv2d_1_Bias_dequantize });
  const depthwise_conv2d_1 = depthwise_conv2d_1_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_2_Bias_dequantize = builder.cast(conv2d_2_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_2_Kernel_dequantize = builder.cast(conv2d_2_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_2_raw = builder.conv2d(depthwise_conv2d_1, conv2d_2_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_2_Bias_dequantize });
  const conv2d_2 = conv2d_2_raw;
  // PAD
  const channel_padding = builder.pad(activation_1, [0,0,0,0], [0,0,0,4]);
  // ADD
  // ADD
  const add_1__xeno_compat__1_raw = builder.add(channel_padding, conv2d_2);
  const add_1__xeno_compat__1 = add_1__xeno_compat__1_raw;
  // RELU
  const activation_2 = builder.relu(add_1__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_2_Bias_dequantize = builder.cast(depthwise_conv2d_2_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_2_Kernel_dequantize = builder.cast(depthwise_conv2d_2_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_2_raw = builder.conv2d(activation_2, depthwise_conv2d_2_Kernel_dequantize, { strides: [2,2], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_2_Kernel_dequantize.shape[3], bias: depthwise_conv2d_2_Bias_dequantize });
  const depthwise_conv2d_2 = depthwise_conv2d_2_raw;
  // MAX_POOL_2D
  // MAX_POOL_2D
  const max_pooling2d_raw = builder.maxPool2d(activation_2, { windowDimensions: [2,2], strides: [2,2], padding: [0, 0, 0, 0], layout: 'nhwc' });
  const max_pooling2d = max_pooling2d_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_3_Bias_dequantize = builder.cast(conv2d_3_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_3_Kernel_dequantize = builder.cast(conv2d_3_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_3_raw = builder.conv2d(depthwise_conv2d_2, conv2d_3_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_3_Bias_dequantize });
  const conv2d_3 = conv2d_3_raw;
  // PAD
  const channel_padding_1 = builder.pad(max_pooling2d, [0,0,0,0], [0,0,0,4]);
  // ADD
  // ADD
  const add_2__xeno_compat__1_raw = builder.add(channel_padding_1, conv2d_3);
  const add_2__xeno_compat__1 = add_2__xeno_compat__1_raw;
  // RELU
  const activation_3 = builder.relu(add_2__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_3_Bias_dequantize = builder.cast(depthwise_conv2d_3_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_3_Kernel_dequantize = builder.cast(depthwise_conv2d_3_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_3_raw = builder.conv2d(activation_3, depthwise_conv2d_3_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_3_Kernel_dequantize.shape[3], bias: depthwise_conv2d_3_Bias_dequantize });
  const depthwise_conv2d_3 = depthwise_conv2d_3_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_4_Bias_dequantize = builder.cast(conv2d_4_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_4_Kernel_dequantize = builder.cast(conv2d_4_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_4_raw = builder.conv2d(depthwise_conv2d_3, conv2d_4_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_4_Bias_dequantize });
  const conv2d_4 = conv2d_4_raw;
  // PAD
  const channel_padding_2 = builder.pad(activation_3, [0,0,0,0], [0,0,0,4]);
  // ADD
  // ADD
  const add_3__xeno_compat__1_raw = builder.add(channel_padding_2, conv2d_4);
  const add_3__xeno_compat__1 = add_3__xeno_compat__1_raw;
  // RELU
  const activation_4 = builder.relu(add_3__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_4_Bias_dequantize = builder.cast(depthwise_conv2d_4_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_4_Kernel_dequantize = builder.cast(depthwise_conv2d_4_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_4_raw = builder.conv2d(activation_4, depthwise_conv2d_4_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_4_Kernel_dequantize.shape[3], bias: depthwise_conv2d_4_Bias_dequantize });
  const depthwise_conv2d_4 = depthwise_conv2d_4_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_5_Bias_dequantize = builder.cast(conv2d_5_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_5_Kernel_dequantize = builder.cast(conv2d_5_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_5_raw = builder.conv2d(depthwise_conv2d_4, conv2d_5_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_5_Bias_dequantize });
  const conv2d_5 = conv2d_5_raw;
  // PAD
  const channel_padding_3 = builder.pad(activation_4, [0,0,0,0], [0,0,0,6]);
  // ADD
  // ADD
  const add_4__xeno_compat__1_raw = builder.add(channel_padding_3, conv2d_5);
  const add_4__xeno_compat__1 = add_4__xeno_compat__1_raw;
  // RELU
  const activation_5 = builder.relu(add_4__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_5_Bias_dequantize = builder.cast(depthwise_conv2d_5_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_5_Kernel_dequantize = builder.cast(depthwise_conv2d_5_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_5_raw = builder.conv2d(activation_5, depthwise_conv2d_5_Kernel_dequantize, { strides: [2,2], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_5_Kernel_dequantize.shape[3], bias: depthwise_conv2d_5_Bias_dequantize });
  const depthwise_conv2d_5 = depthwise_conv2d_5_raw;
  // MAX_POOL_2D
  // MAX_POOL_2D
  const max_pooling2d_1_raw = builder.maxPool2d(activation_5, { windowDimensions: [2,2], strides: [2,2], padding: [0, 0, 0, 0], layout: 'nhwc' });
  const max_pooling2d_1 = max_pooling2d_1_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_6_Bias_dequantize = builder.cast(conv2d_6_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_6_Kernel_dequantize = builder.cast(conv2d_6_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_6_raw = builder.conv2d(depthwise_conv2d_5, conv2d_6_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_6_Bias_dequantize });
  const conv2d_6 = conv2d_6_raw;
  // PAD
  const channel_padding_4 = builder.pad(max_pooling2d_1, [0,0,0,0], [0,0,0,6]);
  // ADD
  // ADD
  const add_5__xeno_compat__1_raw = builder.add(channel_padding_4, conv2d_6);
  const add_5__xeno_compat__1 = add_5__xeno_compat__1_raw;
  // RELU
  const activation_6 = builder.relu(add_5__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_6_Bias_dequantize = builder.cast(depthwise_conv2d_6_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_6_Kernel_dequantize = builder.cast(depthwise_conv2d_6_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_6_raw = builder.conv2d(activation_6, depthwise_conv2d_6_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_6_Kernel_dequantize.shape[3], bias: depthwise_conv2d_6_Bias_dequantize });
  const depthwise_conv2d_6 = depthwise_conv2d_6_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_7_Bias_dequantize = builder.cast(conv2d_7_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_7_Kernel_dequantize = builder.cast(conv2d_7_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_7_raw = builder.conv2d(depthwise_conv2d_6, conv2d_7_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_7_Bias_dequantize });
  const conv2d_7 = conv2d_7_raw;
  // PAD
  const channel_padding_5 = builder.pad(activation_6, [0,0,0,0], [0,0,0,8]);
  // ADD
  // ADD
  const add_6__xeno_compat__1_raw = builder.add(channel_padding_5, conv2d_7);
  const add_6__xeno_compat__1 = add_6__xeno_compat__1_raw;
  // RELU
  const activation_7 = builder.relu(add_6__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_7_Bias_dequantize = builder.cast(depthwise_conv2d_7_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_7_Kernel_dequantize = builder.cast(depthwise_conv2d_7_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_7_raw = builder.conv2d(activation_7, depthwise_conv2d_7_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_7_Kernel_dequantize.shape[3], bias: depthwise_conv2d_7_Bias_dequantize });
  const depthwise_conv2d_7 = depthwise_conv2d_7_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_8_Kernel_dequantize = builder.cast(conv2d_8_Kernel, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_8_Bias_dequantize = builder.cast(conv2d_8_Bias, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_8_raw = builder.conv2d(depthwise_conv2d_7, conv2d_8_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_8_Bias_dequantize });
  const conv2d_8 = conv2d_8_raw;
  // PAD
  const channel_padding_6 = builder.pad(activation_7, [0,0,0,0], [0,0,0,8]);
  // ADD
  // ADD
  const add_7__xeno_compat__1_raw = builder.add(channel_padding_6, conv2d_8);
  const add_7__xeno_compat__1 = add_7__xeno_compat__1_raw;
  // RELU
  const activation_8 = builder.relu(add_7__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_8_Kernel_dequantize = builder.cast(depthwise_conv2d_8_Kernel, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_8_Bias_dequantize = builder.cast(depthwise_conv2d_8_Bias, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_8_raw = builder.conv2d(activation_8, depthwise_conv2d_8_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_8_Kernel_dequantize.shape[3], bias: depthwise_conv2d_8_Bias_dequantize });
  const depthwise_conv2d_8 = depthwise_conv2d_8_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_9_Bias_dequantize = builder.cast(conv2d_9_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_9_Kernel_dequantize = builder.cast(conv2d_9_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_9_raw = builder.conv2d(depthwise_conv2d_8, conv2d_9_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_9_Bias_dequantize });
  const conv2d_9 = conv2d_9_raw;
  // PAD
  const channel_padding_7 = builder.pad(activation_8, [0,0,0,0], [0,0,0,8]);
  // ADD
  // ADD
  const add_8__xeno_compat__1_raw = builder.add(channel_padding_7, conv2d_9);
  const add_8__xeno_compat__1 = add_8__xeno_compat__1_raw;
  // RELU
  const activation_9 = builder.relu(add_8__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_9_Bias_dequantize = builder.cast(depthwise_conv2d_9_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_9_Kernel_dequantize = builder.cast(depthwise_conv2d_9_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_9_raw = builder.conv2d(activation_9, depthwise_conv2d_9_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_9_Kernel_dequantize.shape[3], bias: depthwise_conv2d_9_Bias_dequantize });
  const depthwise_conv2d_9 = depthwise_conv2d_9_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_10_Bias_dequantize = builder.cast(conv2d_10_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_10_Kernel_dequantize = builder.cast(conv2d_10_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_10_raw = builder.conv2d(depthwise_conv2d_9, conv2d_10_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_10_Bias_dequantize });
  const conv2d_10 = conv2d_10_raw;
  // PAD
  const channel_padding_8 = builder.pad(activation_9, [0,0,0,0], [0,0,0,8]);
  // ADD
  // ADD
  const add_9__xeno_compat__1_raw = builder.add(channel_padding_8, conv2d_10);
  const add_9__xeno_compat__1 = add_9__xeno_compat__1_raw;
  // RELU
  const activation_10 = builder.relu(add_9__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_10_Bias_dequantize = builder.cast(depthwise_conv2d_10_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_10_Kernel_dequantize = builder.cast(depthwise_conv2d_10_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_10_raw = builder.conv2d(activation_10, depthwise_conv2d_10_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_10_Kernel_dequantize.shape[3], bias: depthwise_conv2d_10_Bias_dequantize });
  const depthwise_conv2d_10 = depthwise_conv2d_10_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_11_Bias_dequantize = builder.cast(conv2d_11_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_11_Kernel_dequantize = builder.cast(conv2d_11_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_11_raw = builder.conv2d(depthwise_conv2d_10, conv2d_11_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_11_Bias_dequantize });
  const conv2d_11 = conv2d_11_raw;
  // PAD
  const channel_padding_9 = builder.pad(activation_10, [0,0,0,0], [0,0,0,8]);
  // ADD
  // ADD
  const add_10__xeno_compat__1_raw = builder.add(channel_padding_9, conv2d_11);
  const add_10__xeno_compat__1 = add_10__xeno_compat__1_raw;
  // RELU
  const activation_11 = builder.relu(add_10__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_11_Bias_dequantize = builder.cast(depthwise_conv2d_11_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_11_Kernel_dequantize = builder.cast(depthwise_conv2d_11_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_11_raw = builder.conv2d(activation_11, depthwise_conv2d_11_Kernel_dequantize, { strides: [2,2], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_11_Kernel_dequantize.shape[3], bias: depthwise_conv2d_11_Bias_dequantize });
  const depthwise_conv2d_11 = depthwise_conv2d_11_raw;
  // MAX_POOL_2D
  // MAX_POOL_2D
  const max_pooling2d_2_raw = builder.maxPool2d(activation_11, { windowDimensions: [2,2], strides: [2,2], padding: [0, 0, 0, 0], layout: 'nhwc' });
  const max_pooling2d_2 = max_pooling2d_2_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_12_Bias_dequantize = builder.cast(conv2d_12_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_12_Kernel_dequantize = builder.cast(conv2d_12_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_12_raw = builder.conv2d(depthwise_conv2d_11, conv2d_12_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_12_Bias_dequantize });
  const conv2d_12 = conv2d_12_raw;
  // PAD
  const channel_padding_10 = builder.pad(max_pooling2d_2, [0,0,0,0], [0,0,0,8]);
  // ADD
  // ADD
  const add_11__xeno_compat__1_raw = builder.add(channel_padding_10, conv2d_12);
  const add_11__xeno_compat__1 = add_11__xeno_compat__1_raw;
  // RELU
  const activation_12 = builder.relu(add_11__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_12_Bias_dequantize = builder.cast(depthwise_conv2d_12_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_12_Kernel_dequantize = builder.cast(depthwise_conv2d_12_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_12_raw = builder.conv2d(activation_12, depthwise_conv2d_12_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_12_Kernel_dequantize.shape[3], bias: depthwise_conv2d_12_Bias_dequantize });
  const depthwise_conv2d_12 = depthwise_conv2d_12_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_13_Bias_dequantize = builder.cast(conv2d_13_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_13_Kernel_dequantize = builder.cast(conv2d_13_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_13_raw = builder.conv2d(depthwise_conv2d_12, conv2d_13_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_13_Bias_dequantize });
  const conv2d_13 = conv2d_13_raw;
  // ADD
  // ADD
  const add_12__xeno_compat__1_raw = builder.add(activation_12, conv2d_13);
  const add_12__xeno_compat__1 = add_12__xeno_compat__1_raw;
  // RELU
  const activation_13 = builder.relu(add_12__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_13_Bias_dequantize = builder.cast(depthwise_conv2d_13_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_13_Kernel_dequantize = builder.cast(depthwise_conv2d_13_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_13_raw = builder.conv2d(activation_13, depthwise_conv2d_13_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_13_Kernel_dequantize.shape[3], bias: depthwise_conv2d_13_Bias_dequantize });
  const depthwise_conv2d_13 = depthwise_conv2d_13_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_14_Bias_dequantize = builder.cast(conv2d_14_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_14_Kernel_dequantize = builder.cast(conv2d_14_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_14_raw = builder.conv2d(depthwise_conv2d_13, conv2d_14_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_14_Bias_dequantize });
  const conv2d_14 = conv2d_14_raw;
  // ADD
  // ADD
  const add_13__xeno_compat__1_raw = builder.add(activation_13, conv2d_14);
  const add_13__xeno_compat__1 = add_13__xeno_compat__1_raw;
  // RELU
  const activation_14 = builder.relu(add_13__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_14_Bias_dequantize = builder.cast(depthwise_conv2d_14_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_14_Kernel_dequantize = builder.cast(depthwise_conv2d_14_Kernel, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_14_raw = builder.conv2d(activation_14, depthwise_conv2d_14_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_14_Kernel_dequantize.shape[3], bias: depthwise_conv2d_14_Bias_dequantize });
  const depthwise_conv2d_14 = depthwise_conv2d_14_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_15_Bias_dequantize = builder.cast(conv2d_15_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_15_Kernel_dequantize = builder.cast(conv2d_15_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_15_raw = builder.conv2d(depthwise_conv2d_14, conv2d_15_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_15_Bias_dequantize });
  const conv2d_15 = conv2d_15_raw;
  // ADD
  // ADD
  const add_14__xeno_compat__1_raw = builder.add(activation_14, conv2d_15);
  const add_14__xeno_compat__1 = add_14__xeno_compat__1_raw;
  // RELU
  const activation_15 = builder.relu(add_14__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_15_Kernel_dequantize = builder.cast(depthwise_conv2d_15_Kernel, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const depthwise_conv2d_15_Bias_dequantize = builder.cast(depthwise_conv2d_15_Bias, 'float32');
  // DEPTHWISE_CONV_2D
  // DepthwiseConv2D — padding: SAME
  const depthwise_conv2d_15_raw = builder.conv2d(activation_15, depthwise_conv2d_15_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ihwo', groups: depthwise_conv2d_15_Kernel_dequantize.shape[3], bias: depthwise_conv2d_15_Bias_dequantize });
  const depthwise_conv2d_15 = depthwise_conv2d_15_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_16_Bias_dequantize = builder.cast(conv2d_16_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const conv2d_16_Kernel_dequantize = builder.cast(conv2d_16_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: VALID
  const conv2d_16_raw = builder.conv2d(depthwise_conv2d_15, conv2d_16_Kernel_dequantize, { strides: [1,1], dilations: [1,1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: conv2d_16_Bias_dequantize });
  const conv2d_16 = conv2d_16_raw;
  // ADD
  // ADD
  const add_15__xeno_compat__1_raw = builder.add(activation_15, conv2d_16);
  const add_15__xeno_compat__1 = add_15__xeno_compat__1_raw;
  // RELU
  const activation_16 = builder.relu(add_15__xeno_compat__1);
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const classificator_8_Bias_dequantize = builder.cast(classificator_8_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const classificator_8_Kernel_dequantize = builder.cast(classificator_8_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: SAME
  const classificator_8_raw = builder.conv2d(activation_11, classificator_8_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: classificator_8_Bias_dequantize });
  const classificator_8 = classificator_8_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const classificator_16_Kernel_dequantize = builder.cast(classificator_16_Kernel, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const classificator_16_Bias_dequantize = builder.cast(classificator_16_Bias, 'float32');
  // CONV_2D
  // Conv2D — padding: SAME
  const classificator_16_raw = builder.conv2d(activation_16, classificator_16_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: classificator_16_Bias_dequantize });
  const classificator_16 = classificator_16_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const regressor_8_Kernel_dequantize = builder.cast(regressor_8_Kernel, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const regressor_8_Bias_dequantize = builder.cast(regressor_8_Bias, 'float32');
  // CONV_2D
  // Conv2D — padding: SAME
  const regressor_8_raw = builder.conv2d(activation_11, regressor_8_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: regressor_8_Bias_dequantize });
  const regressor_8 = regressor_8_raw;
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const regressor_16_Bias_dequantize = builder.cast(regressor_16_Bias, 'float32');
  // DEQUANTIZE
  // DEQUANTIZE
  // No quantization params found — using cast as fallback
  const regressor_16_Kernel_dequantize = builder.cast(regressor_16_Kernel, 'float32');
  // CONV_2D
  // Conv2D — padding: SAME
  const regressor_16_raw = builder.conv2d(activation_16, regressor_16_Kernel_dequantize, { strides: [1,1], dilations: [1,1], padding: [1, 1, 1, 1], inputLayout: 'nhwc', filterLayout: 'ohwi', bias: regressor_16_Bias_dequantize });
  const regressor_16 = regressor_16_raw;
  // RESHAPE
  const reshapeShape = [1, -1, 1];
  reshapeShape[1] = classificator_8.shape.reduce((a, b) => a * b, 1) / 1;
  const reshape = builder.reshape(classificator_8, reshapeShape);
  // RESHAPE
  const reshape_2Shape = [1, -1, 1];
  reshape_2Shape[1] = classificator_16.shape.reduce((a, b) => a * b, 1) / 1;
  const reshape_2 = builder.reshape(classificator_16, reshape_2Shape);
  // RESHAPE
  const reshape_1Shape = [1, -1, 16];
  reshape_1Shape[1] = regressor_8.shape.reduce((a, b) => a * b, 1) / 16;
  const reshape_1 = builder.reshape(regressor_8, reshape_1Shape);
  // RESHAPE
  const reshape_3Shape = [1, -1, 16];
  reshape_3Shape[1] = regressor_16.shape.reduce((a, b) => a * b, 1) / 16;
  const reshape_3 = builder.reshape(regressor_16, reshape_3Shape);
  // CONCATENATION
  const classificators = builder.concat([reshape, reshape_2], 1);
  // CONCATENATION
  const regressors = builder.concat([reshape_1, reshape_3], 1);

  // Build graph
  const namedOutputs = {};
  namedOutputs['regressors'] = regressors;
  namedOutputs['classificators'] = classificators;
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
  const weights = await WeightsFile.load('face_detector.weights', 'face_detector.manifest.json');
  const buildStart = performance.now();
  const graph = await buildGraph(context, weights);
  console.log(`Graph build: ${(performance.now() - buildStart).toFixed(2)}ms on ${deviceType.toUpperCase()}`)

  // Create input tensors
  const inputData_input = new Float32Array(49152); // [1,128,128,3]
  const inputTensor_input = await context.createTensor({ dataType: 'float32', shape: [1,128,128,3], writable: true });
  context.writeTensor(inputTensor_input, inputData_input);

  // Create output tensors — use actual shapes from built graph
  const outputTensor_regressors = await context.createTensor({ dataType: 'float32', shape: graph.outputShapes['regressors'], readable: true });
  const outputTensor_classificators = await context.createTensor({ dataType: 'float32', shape: graph.outputShapes['classificators'], readable: true });

  const inputs = {
    'input': inputTensor_input,
  };
  const outputs = {
    'regressors': outputTensor_regressors,
    'classificators': outputTensor_classificators,
  };

  const start = performance.now();
  context.dispatch(graph.graph, inputs, outputs);

  // Read results
  const result_regressors = new Float32Array(await context.readTensor(outputTensor_regressors));
  const result_classificators = new Float32Array(await context.readTensor(outputTensor_classificators));
  console.log(`Inference: ${(performance.now() - start).toFixed(2)}ms (1 run) on ${deviceType.toUpperCase()}`)

  // Benchmark: 50 runs
  const NUM_RUNS = 50;
  const runTimes = [];
  for (let i = 0; i < NUM_RUNS; i++) {
    const t0 = performance.now();
    context.dispatch(graph.graph, inputs, outputs);
    new Float32Array(await context.readTensor(outputTensor_regressors));
    new Float32Array(await context.readTensor(outputTensor_classificators));
    runTimes.push(performance.now() - t0);
  }
  const avgTime = (runTimes.reduce((a, b) => a + b, 0) / NUM_RUNS).toFixed(2);
  const sorted = [...runTimes].sort((a, b) => a - b);
  const medianTime = (NUM_RUNS % 2 ? sorted[NUM_RUNS >> 1] : (sorted[NUM_RUNS / 2 - 1] + sorted[NUM_RUNS / 2]) / 2).toFixed(2);
  console.log(`Inference: ${avgTime}ms (average \u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)
  console.log(`Inference: ${medianTime}ms (median \u00b7 ${NUM_RUNS} runs) on ${deviceType.toUpperCase()}`)

  return { 'regressors': result_regressors, 'classificators': result_classificators };
}
