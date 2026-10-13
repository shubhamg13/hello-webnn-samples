class Yolov8sSegWebnn {

  constructor() {

    this.graph_ = null;

    this.context_ = null;

    this.inputTensors_ = {};

    this.outputTensors_ = {};

  }



  async build(contextOptions) {

    // Load weights ArrayBuffer from yolov8s-seg-webnn.bin
        async function loadWeightsArrayBuffer() {
            const response = await fetch('yolov8s-seg-webnn.bin');
            if (!response.ok) {
                throw new Error('Failed to fetch weights: ' + response.statusText);
            }
            return await response.arrayBuffer();
        }

        const weights_array_buffer = await loadWeightsArrayBuffer();



    this.context_ = await navigator.ml.createContext(contextOptions);
    const builder = new MLGraphBuilder(this.context_);


    // Create graph input operands and tensors.

    const images = builder.input(
        'images',
        {dataType: 'float32', shape: [1, 3, 640, 640]}
    );

    this.inputTensors_['images'] = await this.context_.createTensor(
        {dataType: 'float32', shape: [1, 3, 640, 640], writable: true}
    );

    // Create graph operators.

    let var_model_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 0, 3456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 3, 3, 3]},
        var_model_0_conv_weight_buffer
    );

    let var_model_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 3456, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_0_conv_bias_buffer
    );

    const var__model_0_conv_Conv_output_0 = builder.conv2d(
        images, var_model_0_conv_weight,
        {
            bias: var_model_0_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_0_conv_Conv_output_0);

    const var__model_0_act_Mul_output_0 = builder.mul(var__model_0_conv_Conv_output_0, var__model_0_act_Sigmoid_output_0);

    let var_model_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 3584, 73728 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 32, 3, 3]},
        var_model_1_conv_weight_buffer
    );

    let var_model_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 77312, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_1_conv_bias_buffer
    );

    const var__model_1_conv_Conv_output_0 = builder.conv2d(
        var__model_0_act_Mul_output_0, var_model_1_conv_weight,
        {
            bias: var_model_1_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_1_conv_Conv_output_0);

    const var__model_1_act_Mul_output_0 = builder.mul(var__model_1_conv_Conv_output_0, var__model_1_act_Sigmoid_output_0);

    let var_model_2_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 77568, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_2_cv1_conv_weight_buffer
    );

    let var_model_2_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 93952, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_2_cv1_conv_bias_buffer
    );

    const var__model_2_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_1_act_Mul_output_0, var_model_2_cv1_conv_weight,
        {
            bias: var_model_2_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_2_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_2_cv1_conv_Conv_output_0);

    const var__model_2_cv1_act_Mul_output_0 = builder.mul(var__model_2_cv1_conv_Conv_output_0, var__model_2_cv1_act_Sigmoid_output_0);

    const var__model_2_Split_output_0_splits = builder.split(var__model_2_cv1_act_Mul_output_0, [32, 32], { axis: 1 });
const var__model_2_Split_output_0 = var__model_2_Split_output_0_splits[0];
const var__model_2_Split_output_1 = var__model_2_Split_output_0_splits[1];

    let var_model_2_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 119040, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_2_m_0_cv1_conv_weight_buffer
    );

    let var_model_2_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 155904, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_2_m_0_cv1_conv_bias_buffer
    );

    const var__model_2_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_2_Split_output_1, var_model_2_m_0_cv1_conv_weight,
        {
            bias: var_model_2_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_2_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_2_m_0_cv1_conv_Conv_output_0);

    const var__model_2_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_2_m_0_cv1_conv_Conv_output_0, var__model_2_m_0_cv1_act_Sigmoid_output_0);

    let var_model_2_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 156032, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_2_m_0_cv2_conv_weight_buffer
    );

    let var_model_2_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 192896, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_2_m_0_cv2_conv_bias_buffer
    );

    const var__model_2_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_2_m_0_cv1_act_Mul_output_0, var_model_2_m_0_cv2_conv_weight,
        {
            bias: var_model_2_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_2_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_2_m_0_cv2_conv_Conv_output_0);

    const var__model_2_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_2_m_0_cv2_conv_Conv_output_0, var__model_2_m_0_cv2_act_Sigmoid_output_0);

    const var__model_2_m_0_Add_output_0 = builder.add(var__model_2_Split_output_1, var__model_2_m_0_cv2_act_Mul_output_0);

    const var__model_2_Concat_output_0 = builder.concat(
        [var__model_2_Split_output_0, var__model_2_Split_output_1, var__model_2_m_0_Add_output_0],
        1,
    );

    let var_model_2_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 94208, 24576 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 96, 1, 1]},
        var_model_2_cv2_conv_weight_buffer
    );

    let var_model_2_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 118784, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_2_cv2_conv_bias_buffer
    );

    const var__model_2_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_2_Concat_output_0, var_model_2_cv2_conv_weight,
        {
            bias: var_model_2_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_2_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_2_cv2_conv_Conv_output_0);

    const var__model_2_cv2_act_Mul_output_0 = builder.mul(var__model_2_cv2_conv_Conv_output_0, var__model_2_cv2_act_Sigmoid_output_0);

    let var_model_3_conv_weight_buffer = new Float32Array(weights_array_buffer, 193024, 294912 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_3_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 64, 3, 3]},
        var_model_3_conv_weight_buffer
    );

    let var_model_3_conv_bias_buffer = new Float32Array(weights_array_buffer, 487936, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_3_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_3_conv_bias_buffer
    );

    const var__model_3_conv_Conv_output_0 = builder.conv2d(
        var__model_2_cv2_act_Mul_output_0, var_model_3_conv_weight,
        {
            bias: var_model_3_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_3_act_Sigmoid_output_0 = builder.sigmoid(var__model_3_conv_Conv_output_0);

    const var__model_3_act_Mul_output_0 = builder.mul(var__model_3_conv_Conv_output_0, var__model_3_act_Sigmoid_output_0);

    let var_model_4_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 488448, 65536 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 1, 1]},
        var_model_4_cv1_conv_weight_buffer
    );

    let var_model_4_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 553984, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_4_cv1_conv_bias_buffer
    );

    const var__model_4_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_3_act_Mul_output_0, var_model_4_cv1_conv_weight,
        {
            bias: var_model_4_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_4_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_4_cv1_conv_Conv_output_0);

    const var__model_4_cv1_act_Mul_output_0 = builder.mul(var__model_4_cv1_conv_Conv_output_0, var__model_4_cv1_act_Sigmoid_output_0);

    const var__model_4_Split_output_0_splits = builder.split(var__model_4_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
const var__model_4_Split_output_0 = var__model_4_Split_output_0_splits[0];
const var__model_4_Split_output_1 = var__model_4_Split_output_0_splits[1];

    let var_model_4_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 686080, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_4_m_0_cv1_conv_weight_buffer
    );

    let var_model_4_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 833536, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_4_m_0_cv1_conv_bias_buffer
    );

    const var__model_4_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_4_Split_output_1, var_model_4_m_0_cv1_conv_weight,
        {
            bias: var_model_4_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_4_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_4_m_0_cv1_conv_Conv_output_0);

    const var__model_4_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_4_m_0_cv1_conv_Conv_output_0, var__model_4_m_0_cv1_act_Sigmoid_output_0);

    let var_model_4_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 833792, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_4_m_0_cv2_conv_weight_buffer
    );

    let var_model_4_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 981248, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_4_m_0_cv2_conv_bias_buffer
    );

    const var__model_4_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_4_m_0_cv1_act_Mul_output_0, var_model_4_m_0_cv2_conv_weight,
        {
            bias: var_model_4_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_4_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_4_m_0_cv2_conv_Conv_output_0);

    const var__model_4_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_4_m_0_cv2_conv_Conv_output_0, var__model_4_m_0_cv2_act_Sigmoid_output_0);

    const var__model_4_m_0_Add_output_0 = builder.add(var__model_4_Split_output_1, var__model_4_m_0_cv2_act_Mul_output_0);

    let var_model_4_m_1_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 981504, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_4_m_1_cv1_conv_weight_buffer
    );

    let var_model_4_m_1_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 1128960, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_4_m_1_cv1_conv_bias_buffer
    );

    const var__model_4_m_1_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_4_m_0_Add_output_0, var_model_4_m_1_cv1_conv_weight,
        {
            bias: var_model_4_m_1_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_4_m_1_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_4_m_1_cv1_conv_Conv_output_0);

    const var__model_4_m_1_cv1_act_Mul_output_0 = builder.mul(var__model_4_m_1_cv1_conv_Conv_output_0, var__model_4_m_1_cv1_act_Sigmoid_output_0);

    let var_model_4_m_1_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 1129216, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_4_m_1_cv2_conv_weight_buffer
    );

    let var_model_4_m_1_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 1276672, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_4_m_1_cv2_conv_bias_buffer
    );

    const var__model_4_m_1_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_4_m_1_cv1_act_Mul_output_0, var_model_4_m_1_cv2_conv_weight,
        {
            bias: var_model_4_m_1_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_4_m_1_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_4_m_1_cv2_conv_Conv_output_0);

    const var__model_4_m_1_cv2_act_Mul_output_0 = builder.mul(var__model_4_m_1_cv2_conv_Conv_output_0, var__model_4_m_1_cv2_act_Sigmoid_output_0);

    const var__model_4_m_1_Add_output_0 = builder.add(var__model_4_m_0_Add_output_0, var__model_4_m_1_cv2_act_Mul_output_0);

    const var__model_4_Concat_output_0 = builder.concat(
        [var__model_4_Split_output_0, var__model_4_Split_output_1, var__model_4_m_0_Add_output_0, var__model_4_m_1_Add_output_0],
        1,
    );

    let var_model_4_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 554496, 131072 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 256, 1, 1]},
        var_model_4_cv2_conv_weight_buffer
    );

    let var_model_4_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 685568, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_4_cv2_conv_bias_buffer
    );

    const var__model_4_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_4_Concat_output_0, var_model_4_cv2_conv_weight,
        {
            bias: var_model_4_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_4_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_4_cv2_conv_Conv_output_0);

    const var__model_4_cv2_act_Mul_output_0 = builder.mul(var__model_4_cv2_conv_Conv_output_0, var__model_4_cv2_act_Sigmoid_output_0);

    let var_model_5_conv_weight_buffer = new Float32Array(weights_array_buffer, 1276928, 1179648 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_5_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 128, 3, 3]},
        var_model_5_conv_weight_buffer
    );

    let var_model_5_conv_bias_buffer = new Float32Array(weights_array_buffer, 2456576, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_5_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_5_conv_bias_buffer
    );

    const var__model_5_conv_Conv_output_0 = builder.conv2d(
        var__model_4_cv2_act_Mul_output_0, var_model_5_conv_weight,
        {
            bias: var_model_5_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_5_act_Sigmoid_output_0 = builder.sigmoid(var__model_5_conv_Conv_output_0);

    const var__model_5_act_Mul_output_0 = builder.mul(var__model_5_conv_Conv_output_0, var__model_5_act_Sigmoid_output_0);

    let var_model_6_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 2457600, 262144 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 1, 1]},
        var_model_6_cv1_conv_weight_buffer
    );

    let var_model_6_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 2719744, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_6_cv1_conv_bias_buffer
    );

    const var__model_6_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_5_act_Mul_output_0, var_model_6_cv1_conv_weight,
        {
            bias: var_model_6_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_6_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_6_cv1_conv_Conv_output_0);

    const var__model_6_cv1_act_Mul_output_0 = builder.mul(var__model_6_cv1_conv_Conv_output_0, var__model_6_cv1_act_Sigmoid_output_0);

    const var__model_6_Split_output_0_splits = builder.split(var__model_6_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
const var__model_6_Split_output_0 = var__model_6_Split_output_0_splits[0];
const var__model_6_Split_output_1 = var__model_6_Split_output_0_splits[1];

    let var_model_6_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 3246080, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_6_m_0_cv1_conv_weight_buffer
    );

    let var_model_6_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 3835904, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_6_m_0_cv1_conv_bias_buffer
    );

    const var__model_6_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_6_Split_output_1, var_model_6_m_0_cv1_conv_weight,
        {
            bias: var_model_6_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_6_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_6_m_0_cv1_conv_Conv_output_0);

    const var__model_6_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_6_m_0_cv1_conv_Conv_output_0, var__model_6_m_0_cv1_act_Sigmoid_output_0);

    let var_model_6_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 3836416, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_6_m_0_cv2_conv_weight_buffer
    );

    let var_model_6_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 4426240, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_6_m_0_cv2_conv_bias_buffer
    );

    const var__model_6_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_6_m_0_cv1_act_Mul_output_0, var_model_6_m_0_cv2_conv_weight,
        {
            bias: var_model_6_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_6_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_6_m_0_cv2_conv_Conv_output_0);

    const var__model_6_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_6_m_0_cv2_conv_Conv_output_0, var__model_6_m_0_cv2_act_Sigmoid_output_0);

    const var__model_6_m_0_Add_output_0 = builder.add(var__model_6_Split_output_1, var__model_6_m_0_cv2_act_Mul_output_0);

    let var_model_6_m_1_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 4426752, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_6_m_1_cv1_conv_weight_buffer
    );

    let var_model_6_m_1_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 5016576, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_6_m_1_cv1_conv_bias_buffer
    );

    const var__model_6_m_1_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_6_m_0_Add_output_0, var_model_6_m_1_cv1_conv_weight,
        {
            bias: var_model_6_m_1_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_6_m_1_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_6_m_1_cv1_conv_Conv_output_0);

    const var__model_6_m_1_cv1_act_Mul_output_0 = builder.mul(var__model_6_m_1_cv1_conv_Conv_output_0, var__model_6_m_1_cv1_act_Sigmoid_output_0);

    let var_model_6_m_1_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 5017088, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_6_m_1_cv2_conv_weight_buffer
    );

    let var_model_6_m_1_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 5606912, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_6_m_1_cv2_conv_bias_buffer
    );

    const var__model_6_m_1_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_6_m_1_cv1_act_Mul_output_0, var_model_6_m_1_cv2_conv_weight,
        {
            bias: var_model_6_m_1_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_6_m_1_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_6_m_1_cv2_conv_Conv_output_0);

    const var__model_6_m_1_cv2_act_Mul_output_0 = builder.mul(var__model_6_m_1_cv2_conv_Conv_output_0, var__model_6_m_1_cv2_act_Sigmoid_output_0);

    const var__model_6_m_1_Add_output_0 = builder.add(var__model_6_m_0_Add_output_0, var__model_6_m_1_cv2_act_Mul_output_0);

    const var__model_6_Concat_output_0 = builder.concat(
        [var__model_6_Split_output_0, var__model_6_Split_output_1, var__model_6_m_0_Add_output_0, var__model_6_m_1_Add_output_0],
        1,
    );

    let var_model_6_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 2720768, 524288 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 512, 1, 1]},
        var_model_6_cv2_conv_weight_buffer
    );

    let var_model_6_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 3245056, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_6_cv2_conv_bias_buffer
    );

    const var__model_6_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_6_Concat_output_0, var_model_6_cv2_conv_weight,
        {
            bias: var_model_6_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_6_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_6_cv2_conv_Conv_output_0);

    const var__model_6_cv2_act_Mul_output_0 = builder.mul(var__model_6_cv2_conv_Conv_output_0, var__model_6_cv2_act_Sigmoid_output_0);

    let var_model_7_conv_weight_buffer = new Float32Array(weights_array_buffer, 5607424, 4718592 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_7_conv_weight = builder.constant(
        {dataType: 'float32', shape: [512, 256, 3, 3]},
        var_model_7_conv_weight_buffer
    );

    let var_model_7_conv_bias_buffer = new Float32Array(weights_array_buffer, 10326016, 2048 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_7_conv_bias = builder.constant(
        {dataType: 'float32', shape: [512]},
        var_model_7_conv_bias_buffer
    );

    const var__model_7_conv_Conv_output_0 = builder.conv2d(
        var__model_6_cv2_act_Mul_output_0, var_model_7_conv_weight,
        {
            bias: var_model_7_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_7_act_Sigmoid_output_0 = builder.sigmoid(var__model_7_conv_Conv_output_0);

    const var__model_7_act_Mul_output_0 = builder.mul(var__model_7_conv_Conv_output_0, var__model_7_act_Sigmoid_output_0);

    let var_model_8_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 10328064, 1048576 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [512, 512, 1, 1]},
        var_model_8_cv1_conv_weight_buffer
    );

    let var_model_8_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 11376640, 2048 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [512]},
        var_model_8_cv1_conv_bias_buffer
    );

    const var__model_8_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_7_act_Mul_output_0, var_model_8_cv1_conv_weight,
        {
            bias: var_model_8_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_8_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_8_cv1_conv_Conv_output_0);

    const var__model_8_cv1_act_Mul_output_0 = builder.mul(var__model_8_cv1_conv_Conv_output_0, var__model_8_cv1_act_Sigmoid_output_0);

    const var__model_8_Split_output_0_splits = builder.split(var__model_8_cv1_act_Mul_output_0, [256, 256], { axis: 1 });
const var__model_8_Split_output_0 = var__model_8_Split_output_0_splits[0];
const var__model_8_Split_output_1 = var__model_8_Split_output_0_splits[1];

    let var_model_8_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 12953600, 2359296 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 3, 3]},
        var_model_8_m_0_cv1_conv_weight_buffer
    );

    let var_model_8_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 15312896, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_8_m_0_cv1_conv_bias_buffer
    );

    const var__model_8_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_8_Split_output_1, var_model_8_m_0_cv1_conv_weight,
        {
            bias: var_model_8_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_8_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_8_m_0_cv1_conv_Conv_output_0);

    const var__model_8_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_8_m_0_cv1_conv_Conv_output_0, var__model_8_m_0_cv1_act_Sigmoid_output_0);

    let var_model_8_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 15313920, 2359296 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 3, 3]},
        var_model_8_m_0_cv2_conv_weight_buffer
    );

    let var_model_8_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 17673216, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_8_m_0_cv2_conv_bias_buffer
    );

    const var__model_8_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_8_m_0_cv1_act_Mul_output_0, var_model_8_m_0_cv2_conv_weight,
        {
            bias: var_model_8_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_8_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_8_m_0_cv2_conv_Conv_output_0);

    const var__model_8_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_8_m_0_cv2_conv_Conv_output_0, var__model_8_m_0_cv2_act_Sigmoid_output_0);

    const var__model_8_m_0_Add_output_0 = builder.add(var__model_8_Split_output_1, var__model_8_m_0_cv2_act_Mul_output_0);

    const var__model_8_Concat_output_0 = builder.concat(
        [var__model_8_Split_output_0, var__model_8_Split_output_1, var__model_8_m_0_Add_output_0],
        1,
    );

    let var_model_8_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 11378688, 1572864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [512, 768, 1, 1]},
        var_model_8_cv2_conv_weight_buffer
    );

    let var_model_8_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 12951552, 2048 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [512]},
        var_model_8_cv2_conv_bias_buffer
    );

    const var__model_8_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_8_Concat_output_0, var_model_8_cv2_conv_weight,
        {
            bias: var_model_8_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_8_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_8_cv2_conv_Conv_output_0);

    const var__model_8_cv2_act_Mul_output_0 = builder.mul(var__model_8_cv2_conv_Conv_output_0, var__model_8_cv2_act_Sigmoid_output_0);

    let var_model_9_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 17674240, 524288 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 512, 1, 1]},
        var_model_9_cv1_conv_weight_buffer
    );

    let var_model_9_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 18198528, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_9_cv1_conv_bias_buffer
    );

    const var__model_9_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_8_cv2_act_Mul_output_0, var_model_9_cv1_conv_weight,
        {
            bias: var_model_9_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_9_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_9_cv1_conv_Conv_output_0);

    const var__model_9_cv1_act_Mul_output_0 = builder.mul(var__model_9_cv1_conv_Conv_output_0, var__model_9_cv1_act_Sigmoid_output_0);

    const var__model_9_m_MaxPool_output_0 = builder.maxPool2d(
        var__model_9_cv1_act_Mul_output_0,
        {
            strides: [1, 1], padding: [2, 2, 2, 2], windowDimensions: [5, 5], dilations: [1, 1], roundingType: 'floor'
        }
    );

    const var__model_9_m_1_MaxPool_output_0 = builder.maxPool2d(
        var__model_9_m_MaxPool_output_0,
        {
            strides: [1, 1], padding: [2, 2, 2, 2], windowDimensions: [5, 5], dilations: [1, 1], roundingType: 'floor'
        }
    );

    const var__model_9_m_2_MaxPool_output_0 = builder.maxPool2d(
        var__model_9_m_1_MaxPool_output_0,
        {
            strides: [1, 1], padding: [2, 2, 2, 2], windowDimensions: [5, 5], dilations: [1, 1], roundingType: 'floor'
        }
    );

    const var__model_9_Concat_output_0 = builder.concat(
        [var__model_9_cv1_act_Mul_output_0, var__model_9_m_MaxPool_output_0, var__model_9_m_1_MaxPool_output_0, var__model_9_m_2_MaxPool_output_0],
        1,
    );

    let var_model_9_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 18199552, 2097152 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [512, 1024, 1, 1]},
        var_model_9_cv2_conv_weight_buffer
    );

    let var_model_9_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 20296704, 2048 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [512]},
        var_model_9_cv2_conv_bias_buffer
    );

    const var__model_9_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_9_Concat_output_0, var_model_9_cv2_conv_weight,
        {
            bias: var_model_9_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_9_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_9_cv2_conv_Conv_output_0);

    const var__model_9_cv2_act_Mul_output_0 = builder.mul(var__model_9_cv2_conv_Conv_output_0, var__model_9_cv2_act_Sigmoid_output_0);

    const var__model_10_Resize_output_0 = builder.resample2d(
        var__model_9_cv2_act_Mul_output_0,
        {
            mode: 'nearest-neighbor', scales: [2.0, 2.0], sizes: undefined
        }
    );

    const var__model_11_Concat_output_0 = builder.concat(
        [var__model_10_Resize_output_0, var__model_6_cv2_act_Mul_output_0],
        1,
    );

    let var_model_12_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 20298752, 786432 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 768, 1, 1]},
        var_model_12_cv1_conv_weight_buffer
    );

    let var_model_12_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 21085184, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_12_cv1_conv_bias_buffer
    );

    const var__model_12_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_11_Concat_output_0, var_model_12_cv1_conv_weight,
        {
            bias: var_model_12_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_12_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_12_cv1_conv_Conv_output_0);

    const var__model_12_cv1_act_Mul_output_0 = builder.mul(var__model_12_cv1_conv_Conv_output_0, var__model_12_cv1_act_Sigmoid_output_0);

    const var__model_12_Split_output_0_splits = builder.split(var__model_12_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
const var__model_12_Split_output_0 = var__model_12_Split_output_0_splits[0];
const var__model_12_Split_output_1 = var__model_12_Split_output_0_splits[1];

    let var_model_12_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 21480448, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_12_m_0_cv1_conv_weight_buffer
    );

    let var_model_12_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 22070272, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_12_m_0_cv1_conv_bias_buffer
    );

    const var__model_12_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_12_Split_output_1, var_model_12_m_0_cv1_conv_weight,
        {
            bias: var_model_12_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_12_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_12_m_0_cv1_conv_Conv_output_0);

    const var__model_12_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_12_m_0_cv1_conv_Conv_output_0, var__model_12_m_0_cv1_act_Sigmoid_output_0);

    let var_model_12_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 22070784, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_12_m_0_cv2_conv_weight_buffer
    );

    let var_model_12_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 22660608, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_12_m_0_cv2_conv_bias_buffer
    );

    const var__model_12_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_12_m_0_cv1_act_Mul_output_0, var_model_12_m_0_cv2_conv_weight,
        {
            bias: var_model_12_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_12_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_12_m_0_cv2_conv_Conv_output_0);

    const var__model_12_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_12_m_0_cv2_conv_Conv_output_0, var__model_12_m_0_cv2_act_Sigmoid_output_0);

    const var__model_12_Concat_output_0 = builder.concat(
        [var__model_12_Split_output_0, var__model_12_Split_output_1, var__model_12_m_0_cv2_act_Mul_output_0],
        1,
    );

    let var_model_12_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 21086208, 393216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 384, 1, 1]},
        var_model_12_cv2_conv_weight_buffer
    );

    let var_model_12_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 21479424, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_12_cv2_conv_bias_buffer
    );

    const var__model_12_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_12_Concat_output_0, var_model_12_cv2_conv_weight,
        {
            bias: var_model_12_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_12_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_12_cv2_conv_Conv_output_0);

    const var__model_12_cv2_act_Mul_output_0 = builder.mul(var__model_12_cv2_conv_Conv_output_0, var__model_12_cv2_act_Sigmoid_output_0);

    const var__model_13_Resize_output_0 = builder.resample2d(
        var__model_12_cv2_act_Mul_output_0,
        {
            mode: 'nearest-neighbor', scales: [2.0, 2.0], sizes: undefined
        }
    );

    const var__model_14_Concat_output_0 = builder.concat(
        [var__model_13_Resize_output_0, var__model_4_cv2_act_Mul_output_0],
        1,
    );

    let var_model_15_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 22661120, 196608 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 384, 1, 1]},
        var_model_15_cv1_conv_weight_buffer
    );

    let var_model_15_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 22857728, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_15_cv1_conv_bias_buffer
    );

    const var__model_15_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_14_Concat_output_0, var_model_15_cv1_conv_weight,
        {
            bias: var_model_15_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_15_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_15_cv1_conv_Conv_output_0);

    const var__model_15_cv1_act_Mul_output_0 = builder.mul(var__model_15_cv1_conv_Conv_output_0, var__model_15_cv1_act_Sigmoid_output_0);

    const var__model_15_Split_output_0_splits = builder.split(var__model_15_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
const var__model_15_Split_output_0 = var__model_15_Split_output_0_splits[0];
const var__model_15_Split_output_1 = var__model_15_Split_output_0_splits[1];

    let var_model_15_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 22957056, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_15_m_0_cv1_conv_weight_buffer
    );

    let var_model_15_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 23104512, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_15_m_0_cv1_conv_bias_buffer
    );

    const var__model_15_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_15_Split_output_1, var_model_15_m_0_cv1_conv_weight,
        {
            bias: var_model_15_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_15_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_15_m_0_cv1_conv_Conv_output_0);

    const var__model_15_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_15_m_0_cv1_conv_Conv_output_0, var__model_15_m_0_cv1_act_Sigmoid_output_0);

    let var_model_15_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 23104768, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_15_m_0_cv2_conv_weight_buffer
    );

    let var_model_15_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 23252224, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_15_m_0_cv2_conv_bias_buffer
    );

    const var__model_15_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_15_m_0_cv1_act_Mul_output_0, var_model_15_m_0_cv2_conv_weight,
        {
            bias: var_model_15_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_15_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_15_m_0_cv2_conv_Conv_output_0);

    const var__model_15_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_15_m_0_cv2_conv_Conv_output_0, var__model_15_m_0_cv2_act_Sigmoid_output_0);

    const var__model_15_Concat_output_0 = builder.concat(
        [var__model_15_Split_output_0, var__model_15_Split_output_1, var__model_15_m_0_cv2_act_Mul_output_0],
        1,
    );

    let var_model_15_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 22858240, 98304 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 192, 1, 1]},
        var_model_15_cv2_conv_weight_buffer
    );

    let var_model_15_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 22956544, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_15_cv2_conv_bias_buffer
    );

    const var__model_15_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_15_Concat_output_0, var_model_15_cv2_conv_weight,
        {
            bias: var_model_15_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_15_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_15_cv2_conv_Conv_output_0);

    const var__model_15_cv2_act_Mul_output_0 = builder.mul(var__model_15_cv2_conv_Conv_output_0, var__model_15_cv2_act_Sigmoid_output_0);

    let var_model_16_conv_weight_buffer = new Float32Array(weights_array_buffer, 23252480, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_16_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_16_conv_weight_buffer
    );

    let var_model_16_conv_bias_buffer = new Float32Array(weights_array_buffer, 23842304, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_16_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_16_conv_bias_buffer
    );

    const var__model_16_conv_Conv_output_0 = builder.conv2d(
        var__model_15_cv2_act_Mul_output_0, var_model_16_conv_weight,
        {
            bias: var_model_16_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_16_act_Sigmoid_output_0 = builder.sigmoid(var__model_16_conv_Conv_output_0);

    const var__model_16_act_Mul_output_0 = builder.mul(var__model_16_conv_Conv_output_0, var__model_16_act_Sigmoid_output_0);

    const var__model_17_Concat_output_0 = builder.concat(
        [var__model_16_act_Mul_output_0, var__model_12_cv2_act_Mul_output_0],
        1,
    );

    let var_model_18_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 23842816, 393216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 384, 1, 1]},
        var_model_18_cv1_conv_weight_buffer
    );

    let var_model_18_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 24236032, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_18_cv1_conv_bias_buffer
    );

    const var__model_18_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_17_Concat_output_0, var_model_18_cv1_conv_weight,
        {
            bias: var_model_18_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_18_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_18_cv1_conv_Conv_output_0);

    const var__model_18_cv1_act_Mul_output_0 = builder.mul(var__model_18_cv1_conv_Conv_output_0, var__model_18_cv1_act_Sigmoid_output_0);

    const var__model_18_Split_output_0_splits = builder.split(var__model_18_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
const var__model_18_Split_output_0 = var__model_18_Split_output_0_splits[0];
const var__model_18_Split_output_1 = var__model_18_Split_output_0_splits[1];

    let var_model_18_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 24631296, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_18_m_0_cv1_conv_weight_buffer
    );

    let var_model_18_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 25221120, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_18_m_0_cv1_conv_bias_buffer
    );

    const var__model_18_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_18_Split_output_1, var_model_18_m_0_cv1_conv_weight,
        {
            bias: var_model_18_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_18_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_18_m_0_cv1_conv_Conv_output_0);

    const var__model_18_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_18_m_0_cv1_conv_Conv_output_0, var__model_18_m_0_cv1_act_Sigmoid_output_0);

    let var_model_18_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 25221632, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_18_m_0_cv2_conv_weight_buffer
    );

    let var_model_18_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 25811456, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_18_m_0_cv2_conv_bias_buffer
    );

    const var__model_18_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_18_m_0_cv1_act_Mul_output_0, var_model_18_m_0_cv2_conv_weight,
        {
            bias: var_model_18_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_18_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_18_m_0_cv2_conv_Conv_output_0);

    const var__model_18_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_18_m_0_cv2_conv_Conv_output_0, var__model_18_m_0_cv2_act_Sigmoid_output_0);

    const var__model_18_Concat_output_0 = builder.concat(
        [var__model_18_Split_output_0, var__model_18_Split_output_1, var__model_18_m_0_cv2_act_Mul_output_0],
        1,
    );

    let var_model_18_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 24237056, 393216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 384, 1, 1]},
        var_model_18_cv2_conv_weight_buffer
    );

    let var_model_18_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 24630272, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_18_cv2_conv_bias_buffer
    );

    const var__model_18_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_18_Concat_output_0, var_model_18_cv2_conv_weight,
        {
            bias: var_model_18_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_18_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_18_cv2_conv_Conv_output_0);

    const var__model_18_cv2_act_Mul_output_0 = builder.mul(var__model_18_cv2_conv_Conv_output_0, var__model_18_cv2_act_Sigmoid_output_0);

    let var_model_19_conv_weight_buffer = new Float32Array(weights_array_buffer, 25811968, 2359296 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_19_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 3, 3]},
        var_model_19_conv_weight_buffer
    );

    let var_model_19_conv_bias_buffer = new Float32Array(weights_array_buffer, 28171264, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_19_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_19_conv_bias_buffer
    );

    const var__model_19_conv_Conv_output_0 = builder.conv2d(
        var__model_18_cv2_act_Mul_output_0, var_model_19_conv_weight,
        {
            bias: var_model_19_conv_bias, strides: [2, 2], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_19_act_Sigmoid_output_0 = builder.sigmoid(var__model_19_conv_Conv_output_0);

    const var__model_19_act_Mul_output_0 = builder.mul(var__model_19_conv_Conv_output_0, var__model_19_act_Sigmoid_output_0);

    const var__model_20_Concat_output_0 = builder.concat(
        [var__model_19_act_Mul_output_0, var__model_9_cv2_act_Mul_output_0],
        1,
    );

    let var_model_21_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 28172288, 1572864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [512, 768, 1, 1]},
        var_model_21_cv1_conv_weight_buffer
    );

    let var_model_21_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 29745152, 2048 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [512]},
        var_model_21_cv1_conv_bias_buffer
    );

    const var__model_21_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_20_Concat_output_0, var_model_21_cv1_conv_weight,
        {
            bias: var_model_21_cv1_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_21_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_21_cv1_conv_Conv_output_0);

    const var__model_21_cv1_act_Mul_output_0 = builder.mul(var__model_21_cv1_conv_Conv_output_0, var__model_21_cv1_act_Sigmoid_output_0);

    const var__model_21_Split_output_0_splits = builder.split(var__model_21_cv1_act_Mul_output_0, [256, 256], { axis: 1 });
const var__model_21_Split_output_0 = var__model_21_Split_output_0_splits[0];
const var__model_21_Split_output_1 = var__model_21_Split_output_0_splits[1];

    let var_model_21_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 31322112, 2359296 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 3, 3]},
        var_model_21_m_0_cv1_conv_weight_buffer
    );

    let var_model_21_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 33681408, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_21_m_0_cv1_conv_bias_buffer
    );

    const var__model_21_m_0_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_21_Split_output_1, var_model_21_m_0_cv1_conv_weight,
        {
            bias: var_model_21_m_0_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_21_m_0_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_21_m_0_cv1_conv_Conv_output_0);

    const var__model_21_m_0_cv1_act_Mul_output_0 = builder.mul(var__model_21_m_0_cv1_conv_Conv_output_0, var__model_21_m_0_cv1_act_Sigmoid_output_0);

    let var_model_21_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 33682432, 2359296 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 3, 3]},
        var_model_21_m_0_cv2_conv_weight_buffer
    );

    let var_model_21_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 36041728, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
        var_model_21_m_0_cv2_conv_bias_buffer
    );

    const var__model_21_m_0_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_21_m_0_cv1_act_Mul_output_0, var_model_21_m_0_cv2_conv_weight,
        {
            bias: var_model_21_m_0_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_21_m_0_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_21_m_0_cv2_conv_Conv_output_0);

    const var__model_21_m_0_cv2_act_Mul_output_0 = builder.mul(var__model_21_m_0_cv2_conv_Conv_output_0, var__model_21_m_0_cv2_act_Sigmoid_output_0);

    const var__model_21_Concat_output_0 = builder.concat(
        [var__model_21_Split_output_0, var__model_21_Split_output_1, var__model_21_m_0_cv2_act_Mul_output_0],
        1,
    );

    let var_model_21_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 29747200, 1572864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [512, 768, 1, 1]},
        var_model_21_cv2_conv_weight_buffer
    );

    let var_model_21_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 31320064, 2048 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [512]},
        var_model_21_cv2_conv_bias_buffer
    );

    const var__model_21_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_21_Concat_output_0, var_model_21_cv2_conv_weight,
        {
            bias: var_model_21_cv2_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_21_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_21_cv2_conv_Conv_output_0);

    const var__model_21_cv2_act_Mul_output_0 = builder.mul(var__model_21_cv2_conv_Conv_output_0, var__model_21_cv2_act_Sigmoid_output_0);

    let var_model_22_cv2_0_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 36042752, 294912 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 128, 3, 3]},
        var_model_22_cv2_0_0_conv_weight_buffer
    );

    let var_model_22_cv2_0_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 36337664, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_0_0_conv_bias_buffer
    );

    const var__model_22_cv2_0_cv2_0_0_conv_Conv_output_0 = builder.conv2d(
        var__model_15_cv2_act_Mul_output_0, var_model_22_cv2_0_0_conv_weight,
        {
            bias: var_model_22_cv2_0_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv2_0_cv2_0_0_conv_Conv_output_0);

    const var__model_22_cv2_0_cv2_0_0_act_Mul_output_0 = builder.mul(var__model_22_cv2_0_cv2_0_0_conv_Conv_output_0, var__model_22_cv2_0_cv2_0_0_act_Sigmoid_output_0);

    let var_model_22_cv2_0_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 36337920, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_0_1_conv_weight_buffer
    );

    let var_model_22_cv2_0_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 36485376, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_0_1_conv_bias_buffer
    );

    const var__model_22_cv2_0_cv2_0_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv2_0_cv2_0_0_act_Mul_output_0, var_model_22_cv2_0_1_conv_weight,
        {
            bias: var_model_22_cv2_0_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv2_0_cv2_0_1_conv_Conv_output_0);

    const var__model_22_cv2_0_cv2_0_1_act_Mul_output_0 = builder.mul(var__model_22_cv2_0_cv2_0_1_conv_Conv_output_0, var__model_22_cv2_0_cv2_0_1_act_Sigmoid_output_0);

    let var_model_22_cv2_0_2_weight_buffer = new Float32Array(weights_array_buffer, 36485632, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_2_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_22_cv2_0_2_weight_buffer
    );

    let var_model_22_cv2_0_2_bias_buffer = new Float32Array(weights_array_buffer, 36502016, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_2_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_0_2_bias_buffer
    );

    const var__model_22_cv2_0_cv2_0_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv2_0_cv2_0_1_act_Mul_output_0, var_model_22_cv2_0_2_weight,
        {
            bias: var_model_22_cv2_0_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_output_0 = builder.reshape(
        var__model_22_cv2_0_cv2_0_2_Conv_output_0,
        (() => {
        const shape = [1, 64, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv2_0_cv2_0_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    let var_model_22_cv2_1_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 36502272, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 256, 3, 3]},
        var_model_22_cv2_1_0_conv_weight_buffer
    );

    let var_model_22_cv2_1_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 37092096, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_1_0_conv_bias_buffer
    );

    const var__model_22_cv2_1_cv2_1_0_conv_Conv_output_0 = builder.conv2d(
        var__model_18_cv2_act_Mul_output_0, var_model_22_cv2_1_0_conv_weight,
        {
            bias: var_model_22_cv2_1_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv2_1_cv2_1_0_conv_Conv_output_0);

    const var__model_22_cv2_1_cv2_1_0_act_Mul_output_0 = builder.mul(var__model_22_cv2_1_cv2_1_0_conv_Conv_output_0, var__model_22_cv2_1_cv2_1_0_act_Sigmoid_output_0);

    let var_model_22_cv2_1_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 37092352, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_1_1_conv_weight_buffer
    );

    let var_model_22_cv2_1_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 37239808, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_1_1_conv_bias_buffer
    );

    const var__model_22_cv2_1_cv2_1_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv2_1_cv2_1_0_act_Mul_output_0, var_model_22_cv2_1_1_conv_weight,
        {
            bias: var_model_22_cv2_1_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv2_1_cv2_1_1_conv_Conv_output_0);

    const var__model_22_cv2_1_cv2_1_1_act_Mul_output_0 = builder.mul(var__model_22_cv2_1_cv2_1_1_conv_Conv_output_0, var__model_22_cv2_1_cv2_1_1_act_Sigmoid_output_0);

    let var_model_22_cv2_1_2_weight_buffer = new Float32Array(weights_array_buffer, 37240064, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_2_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_22_cv2_1_2_weight_buffer
    );

    let var_model_22_cv2_1_2_bias_buffer = new Float32Array(weights_array_buffer, 37256448, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_2_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_1_2_bias_buffer
    );

    const var__model_22_cv2_1_cv2_1_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv2_1_cv2_1_1_act_Mul_output_0, var_model_22_cv2_1_2_weight,
        {
            bias: var_model_22_cv2_1_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_1_output_0 = builder.reshape(
        var__model_22_cv2_1_cv2_1_2_Conv_output_0,
        (() => {
        const shape = [1, 64, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv2_1_cv2_1_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    let var_model_22_cv2_2_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 37256704, 1179648 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 512, 3, 3]},
        var_model_22_cv2_2_0_conv_weight_buffer
    );

    let var_model_22_cv2_2_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 38436352, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_2_0_conv_bias_buffer
    );

    const var__model_22_cv2_2_cv2_2_0_conv_Conv_output_0 = builder.conv2d(
        var__model_21_cv2_act_Mul_output_0, var_model_22_cv2_2_0_conv_weight,
        {
            bias: var_model_22_cv2_2_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv2_2_cv2_2_0_conv_Conv_output_0);

    const var__model_22_cv2_2_cv2_2_0_act_Mul_output_0 = builder.mul(var__model_22_cv2_2_cv2_2_0_conv_Conv_output_0, var__model_22_cv2_2_cv2_2_0_act_Sigmoid_output_0);

    let var_model_22_cv2_2_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 38436608, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_2_1_conv_weight_buffer
    );

    let var_model_22_cv2_2_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 38584064, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_2_1_conv_bias_buffer
    );

    const var__model_22_cv2_2_cv2_2_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv2_2_cv2_2_0_act_Mul_output_0, var_model_22_cv2_2_1_conv_weight,
        {
            bias: var_model_22_cv2_2_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv2_2_cv2_2_1_conv_Conv_output_0);

    const var__model_22_cv2_2_cv2_2_1_act_Mul_output_0 = builder.mul(var__model_22_cv2_2_cv2_2_1_conv_Conv_output_0, var__model_22_cv2_2_cv2_2_1_act_Sigmoid_output_0);

    let var_model_22_cv2_2_2_weight_buffer = new Float32Array(weights_array_buffer, 38584320, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_2_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_22_cv2_2_2_weight_buffer
    );

    let var_model_22_cv2_2_2_bias_buffer = new Float32Array(weights_array_buffer, 38600704, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_2_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_cv2_2_2_bias_buffer
    );

    const var__model_22_cv2_2_cv2_2_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv2_2_cv2_2_1_act_Mul_output_0, var_model_22_cv2_2_2_weight,
        {
            bias: var_model_22_cv2_2_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_2_output_0 = builder.reshape(
        var__model_22_cv2_2_cv2_2_2_Conv_output_0,
        (() => {
        const shape = [1, 64, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv2_2_cv2_2_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Concat_output_0 = builder.concat(
        [var__model_22_Reshape_output_0, var__model_22_Reshape_1_output_0, var__model_22_Reshape_2_output_0],
        2,
    );

    let var_model_22_cv3_0_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 38600960, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_22_cv3_0_0_conv_weight_buffer
    );

    let var_model_22_cv3_0_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 39190784, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_cv3_0_0_conv_bias_buffer
    );

    const var__model_22_cv3_0_cv3_0_0_conv_Conv_output_0 = builder.conv2d(
        var__model_15_cv2_act_Mul_output_0, var_model_22_cv3_0_0_conv_weight,
        {
            bias: var_model_22_cv3_0_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv3_0_cv3_0_0_conv_Conv_output_0);

    const var__model_22_cv3_0_cv3_0_0_act_Mul_output_0 = builder.mul(var__model_22_cv3_0_cv3_0_0_conv_Conv_output_0, var__model_22_cv3_0_cv3_0_0_act_Sigmoid_output_0);

    let var_model_22_cv3_0_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 39191296, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_22_cv3_0_1_conv_weight_buffer
    );

    let var_model_22_cv3_0_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 39781120, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_cv3_0_1_conv_bias_buffer
    );

    const var__model_22_cv3_0_cv3_0_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv3_0_cv3_0_0_act_Mul_output_0, var_model_22_cv3_0_1_conv_weight,
        {
            bias: var_model_22_cv3_0_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv3_0_cv3_0_1_conv_Conv_output_0);

    const var__model_22_cv3_0_cv3_0_1_act_Mul_output_0 = builder.mul(var__model_22_cv3_0_cv3_0_1_conv_Conv_output_0, var__model_22_cv3_0_cv3_0_1_act_Sigmoid_output_0);

    let var_model_22_cv3_0_2_weight_buffer = new Float32Array(weights_array_buffer, 39781632, 40960 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_2_weight = builder.constant(
        {dataType: 'float32', shape: [80, 128, 1, 1]},
        var_model_22_cv3_0_2_weight_buffer
    );

    let var_model_22_cv3_0_2_bias_buffer = new Float32Array(weights_array_buffer, 39822592, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_2_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
        var_model_22_cv3_0_2_bias_buffer
    );

    const var__model_22_cv3_0_cv3_0_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv3_0_cv3_0_1_act_Mul_output_0, var_model_22_cv3_0_2_weight,
        {
            bias: var_model_22_cv3_0_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_3_output_0 = builder.reshape(
        var__model_22_cv3_0_cv3_0_2_Conv_output_0,
        (() => {
        const shape = [1, 80, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv3_0_cv3_0_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    let var_model_22_cv3_1_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 39822912, 1179648 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 256, 3, 3]},
        var_model_22_cv3_1_0_conv_weight_buffer
    );

    let var_model_22_cv3_1_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 41002560, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_cv3_1_0_conv_bias_buffer
    );

    const var__model_22_cv3_1_cv3_1_0_conv_Conv_output_0 = builder.conv2d(
        var__model_18_cv2_act_Mul_output_0, var_model_22_cv3_1_0_conv_weight,
        {
            bias: var_model_22_cv3_1_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv3_1_cv3_1_0_conv_Conv_output_0);

    const var__model_22_cv3_1_cv3_1_0_act_Mul_output_0 = builder.mul(var__model_22_cv3_1_cv3_1_0_conv_Conv_output_0, var__model_22_cv3_1_cv3_1_0_act_Sigmoid_output_0);

    let var_model_22_cv3_1_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 41003072, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_22_cv3_1_1_conv_weight_buffer
    );

    let var_model_22_cv3_1_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 41592896, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_cv3_1_1_conv_bias_buffer
    );

    const var__model_22_cv3_1_cv3_1_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv3_1_cv3_1_0_act_Mul_output_0, var_model_22_cv3_1_1_conv_weight,
        {
            bias: var_model_22_cv3_1_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv3_1_cv3_1_1_conv_Conv_output_0);

    const var__model_22_cv3_1_cv3_1_1_act_Mul_output_0 = builder.mul(var__model_22_cv3_1_cv3_1_1_conv_Conv_output_0, var__model_22_cv3_1_cv3_1_1_act_Sigmoid_output_0);

    let var_model_22_cv3_1_2_weight_buffer = new Float32Array(weights_array_buffer, 41593408, 40960 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_2_weight = builder.constant(
        {dataType: 'float32', shape: [80, 128, 1, 1]},
        var_model_22_cv3_1_2_weight_buffer
    );

    let var_model_22_cv3_1_2_bias_buffer = new Float32Array(weights_array_buffer, 41634368, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_2_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
        var_model_22_cv3_1_2_bias_buffer
    );

    const var__model_22_cv3_1_cv3_1_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv3_1_cv3_1_1_act_Mul_output_0, var_model_22_cv3_1_2_weight,
        {
            bias: var_model_22_cv3_1_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_4_output_0 = builder.reshape(
        var__model_22_cv3_1_cv3_1_2_Conv_output_0,
        (() => {
        const shape = [1, 80, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv3_1_cv3_1_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    let var_model_22_cv3_2_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 41634688, 2359296 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 512, 3, 3]},
        var_model_22_cv3_2_0_conv_weight_buffer
    );

    let var_model_22_cv3_2_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 43993984, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_cv3_2_0_conv_bias_buffer
    );

    const var__model_22_cv3_2_cv3_2_0_conv_Conv_output_0 = builder.conv2d(
        var__model_21_cv2_act_Mul_output_0, var_model_22_cv3_2_0_conv_weight,
        {
            bias: var_model_22_cv3_2_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv3_2_cv3_2_0_conv_Conv_output_0);

    const var__model_22_cv3_2_cv3_2_0_act_Mul_output_0 = builder.mul(var__model_22_cv3_2_cv3_2_0_conv_Conv_output_0, var__model_22_cv3_2_cv3_2_0_act_Sigmoid_output_0);

    let var_model_22_cv3_2_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 43994496, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_22_cv3_2_1_conv_weight_buffer
    );

    let var_model_22_cv3_2_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 44584320, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_cv3_2_1_conv_bias_buffer
    );

    const var__model_22_cv3_2_cv3_2_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv3_2_cv3_2_0_act_Mul_output_0, var_model_22_cv3_2_1_conv_weight,
        {
            bias: var_model_22_cv3_2_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv3_2_cv3_2_1_conv_Conv_output_0);

    const var__model_22_cv3_2_cv3_2_1_act_Mul_output_0 = builder.mul(var__model_22_cv3_2_cv3_2_1_conv_Conv_output_0, var__model_22_cv3_2_cv3_2_1_act_Sigmoid_output_0);

    let var_model_22_cv3_2_2_weight_buffer = new Float32Array(weights_array_buffer, 44584832, 40960 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_2_weight = builder.constant(
        {dataType: 'float32', shape: [80, 128, 1, 1]},
        var_model_22_cv3_2_2_weight_buffer
    );

    let var_model_22_cv3_2_2_bias_buffer = new Float32Array(weights_array_buffer, 44625792, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_2_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
        var_model_22_cv3_2_2_bias_buffer
    );

    const var__model_22_cv3_2_cv3_2_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv3_2_cv3_2_1_act_Mul_output_0, var_model_22_cv3_2_2_weight,
        {
            bias: var_model_22_cv3_2_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_5_output_0 = builder.reshape(
        var__model_22_cv3_2_cv3_2_2_Conv_output_0,
        (() => {
        const shape = [1, 80, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv3_2_cv3_2_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Concat_1_output_0 = builder.concat(
        [var__model_22_Reshape_3_output_0, var__model_22_Reshape_4_output_0, var__model_22_Reshape_5_output_0],
        2,
    );

    let var_model_22_cv4_0_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 46086016, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 128, 3, 3]},
        var_model_22_cv4_0_0_conv_weight_buffer
    );

    let var_model_22_cv4_0_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 46233472, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_0_0_conv_bias_buffer
    );

    const var__model_22_cv4_0_cv4_0_0_conv_Conv_output_0 = builder.conv2d(
        var__model_15_cv2_act_Mul_output_0, var_model_22_cv4_0_0_conv_weight,
        {
            bias: var_model_22_cv4_0_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv4_0_cv4_0_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv4_0_cv4_0_0_conv_Conv_output_0);

    const var__model_22_cv4_0_cv4_0_0_act_Mul_output_0 = builder.mul(var__model_22_cv4_0_cv4_0_0_conv_Conv_output_0, var__model_22_cv4_0_cv4_0_0_act_Sigmoid_output_0);

    let var_model_22_cv4_0_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 46233600, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_22_cv4_0_1_conv_weight_buffer
    );

    let var_model_22_cv4_0_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 46270464, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_0_1_conv_bias_buffer
    );

    const var__model_22_cv4_0_cv4_0_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv4_0_cv4_0_0_act_Mul_output_0, var_model_22_cv4_0_1_conv_weight,
        {
            bias: var_model_22_cv4_0_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv4_0_cv4_0_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv4_0_cv4_0_1_conv_Conv_output_0);

    const var__model_22_cv4_0_cv4_0_1_act_Mul_output_0 = builder.mul(var__model_22_cv4_0_cv4_0_1_conv_Conv_output_0, var__model_22_cv4_0_cv4_0_1_act_Sigmoid_output_0);

    let var_model_22_cv4_0_2_weight_buffer = new Float32Array(weights_array_buffer, 46270592, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_2_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_22_cv4_0_2_weight_buffer
    );

    let var_model_22_cv4_0_2_bias_buffer = new Float32Array(weights_array_buffer, 46274688, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_2_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_0_2_bias_buffer
    );

    const var__model_22_cv4_0_cv4_0_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv4_0_cv4_0_1_act_Mul_output_0, var_model_22_cv4_0_2_weight,
        {
            bias: var_model_22_cv4_0_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_6_output_0 = builder.reshape(
        var__model_22_cv4_0_cv4_0_2_Conv_output_0,
        (() => {
        const shape = [1, 32, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv4_0_cv4_0_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    let var_model_22_cv4_1_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 46274816, 294912 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 256, 3, 3]},
        var_model_22_cv4_1_0_conv_weight_buffer
    );

    let var_model_22_cv4_1_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 46569728, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_1_0_conv_bias_buffer
    );

    const var__model_22_cv4_1_cv4_1_0_conv_Conv_output_0 = builder.conv2d(
        var__model_18_cv2_act_Mul_output_0, var_model_22_cv4_1_0_conv_weight,
        {
            bias: var_model_22_cv4_1_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv4_1_cv4_1_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv4_1_cv4_1_0_conv_Conv_output_0);

    const var__model_22_cv4_1_cv4_1_0_act_Mul_output_0 = builder.mul(var__model_22_cv4_1_cv4_1_0_conv_Conv_output_0, var__model_22_cv4_1_cv4_1_0_act_Sigmoid_output_0);

    let var_model_22_cv4_1_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 46569856, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_22_cv4_1_1_conv_weight_buffer
    );

    let var_model_22_cv4_1_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 46606720, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_1_1_conv_bias_buffer
    );

    const var__model_22_cv4_1_cv4_1_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv4_1_cv4_1_0_act_Mul_output_0, var_model_22_cv4_1_1_conv_weight,
        {
            bias: var_model_22_cv4_1_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv4_1_cv4_1_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv4_1_cv4_1_1_conv_Conv_output_0);

    const var__model_22_cv4_1_cv4_1_1_act_Mul_output_0 = builder.mul(var__model_22_cv4_1_cv4_1_1_conv_Conv_output_0, var__model_22_cv4_1_cv4_1_1_act_Sigmoid_output_0);

    let var_model_22_cv4_1_2_weight_buffer = new Float32Array(weights_array_buffer, 46606848, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_2_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_22_cv4_1_2_weight_buffer
    );

    let var_model_22_cv4_1_2_bias_buffer = new Float32Array(weights_array_buffer, 46610944, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_2_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_1_2_bias_buffer
    );

    const var__model_22_cv4_1_cv4_1_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv4_1_cv4_1_1_act_Mul_output_0, var_model_22_cv4_1_2_weight,
        {
            bias: var_model_22_cv4_1_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_7_output_0 = builder.reshape(
        var__model_22_cv4_1_cv4_1_2_Conv_output_0,
        (() => {
        const shape = [1, 32, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv4_1_cv4_1_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    let var_model_22_cv4_2_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 46611072, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 512, 3, 3]},
        var_model_22_cv4_2_0_conv_weight_buffer
    );

    let var_model_22_cv4_2_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 47200896, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_2_0_conv_bias_buffer
    );

    const var__model_22_cv4_2_cv4_2_0_conv_Conv_output_0 = builder.conv2d(
        var__model_21_cv2_act_Mul_output_0, var_model_22_cv4_2_0_conv_weight,
        {
            bias: var_model_22_cv4_2_0_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv4_2_cv4_2_0_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv4_2_cv4_2_0_conv_Conv_output_0);

    const var__model_22_cv4_2_cv4_2_0_act_Mul_output_0 = builder.mul(var__model_22_cv4_2_cv4_2_0_conv_Conv_output_0, var__model_22_cv4_2_cv4_2_0_act_Sigmoid_output_0);

    let var_model_22_cv4_2_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 47201024, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_22_cv4_2_1_conv_weight_buffer
    );

    let var_model_22_cv4_2_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 47237888, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_2_1_conv_bias_buffer
    );

    const var__model_22_cv4_2_cv4_2_1_conv_Conv_output_0 = builder.conv2d(
        var__model_22_cv4_2_cv4_2_0_act_Mul_output_0, var_model_22_cv4_2_1_conv_weight,
        {
            bias: var_model_22_cv4_2_1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_cv4_2_cv4_2_1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_cv4_2_cv4_2_1_conv_Conv_output_0);

    const var__model_22_cv4_2_cv4_2_1_act_Mul_output_0 = builder.mul(var__model_22_cv4_2_cv4_2_1_conv_Conv_output_0, var__model_22_cv4_2_cv4_2_1_act_Sigmoid_output_0);

    let var_model_22_cv4_2_2_weight_buffer = new Float32Array(weights_array_buffer, 47238016, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_2_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_22_cv4_2_2_weight_buffer
    );

    let var_model_22_cv4_2_2_bias_buffer = new Float32Array(weights_array_buffer, 47242112, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_2_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_cv4_2_2_bias_buffer
    );

    const var__model_22_cv4_2_cv4_2_2_Conv_output_0 = builder.conv2d(
        var__model_22_cv4_2_cv4_2_1_act_Mul_output_0, var_model_22_cv4_2_2_weight,
        {
            bias: var_model_22_cv4_2_2_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_Reshape_8_output_0 = builder.reshape(
        var__model_22_cv4_2_cv4_2_2_Conv_output_0,
        (() => {
        const shape = [1, 32, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_cv4_2_cv4_2_2_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Concat_2_output_0 = builder.concat(
        [var__model_22_Reshape_6_output_0, var__model_22_Reshape_7_output_0, var__model_22_Reshape_8_output_0],
        2,
    );

    const var__model_22_dfl_Reshape_output_0 = builder.reshape(
        var__model_22_Concat_output_0,
        (() => {
        const shape = [1, 4, 16, 8400];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_Concat_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_dfl_Transpose_output_0 = builder.transpose(
        var__model_22_dfl_Reshape_output_0,
        { permutation: [0, 2, 1, 3] }
    );

    const var__model_22_dfl_Softmax_output_0 = builder.softmax(
        var__model_22_dfl_Transpose_output_0,
        1
    );

    let var_model_22_dfl_conv_weight_buffer = new Float32Array(weights_array_buffer, 44626112, 64 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_dfl_conv_weight = builder.constant(
        {dataType: 'float32', shape: [1, 16, 1, 1]},
        var_model_22_dfl_conv_weight_buffer
    );

    const var__model_22_dfl_conv_Conv_output_0 = builder.conv2d(
        var__model_22_dfl_Softmax_output_0, var_model_22_dfl_conv_weight,
        {
            bias: undefined, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_dfl_Reshape_1_output_0 = builder.reshape(
        var__model_22_dfl_conv_Conv_output_0,
        (() => {
        const shape = [1, 4, 8400];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_dfl_conv_Conv_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Slice_output_0 = builder.slice(var__model_22_dfl_Reshape_1_output_0, [0, 0, 0], [1, 2, 8400]);

    const var__model_22_Slice_1_output_0 = builder.slice(var__model_22_dfl_Reshape_1_output_0, [0, 2, 0], [1, 2, 8400]);

    let var__model_22_Constant_15_output_0_buffer = new Float32Array(weights_array_buffer, 47276084, 67200 / Float32Array.BYTES_PER_ELEMENT);
    const var__model_22_Constant_15_output_0 = builder.constant(
        {dataType: 'float32', shape: [1, 2, 8400]},
        var__model_22_Constant_15_output_0_buffer
    );

    const var__model_22_Sub_output_0 = builder.sub(var__model_22_Constant_15_output_0, var__model_22_Slice_output_0);

    const var__model_22_Add_1_output_0 = builder.add(var__model_22_Constant_15_output_0, var__model_22_Slice_1_output_0);

    const var__model_22_Add_2_output_0 = builder.add(var__model_22_Sub_output_0, var__model_22_Add_1_output_0);

    let var__model_22_Constant_17_output_0_buffer = new Float32Array(weights_array_buffer, 47242464, 4 / Float32Array.BYTES_PER_ELEMENT);
    const var__model_22_Constant_17_output_0 = builder.constant(
        {dataType: 'float32', shape: []},
        var__model_22_Constant_17_output_0_buffer
    );

    const var__model_22_Div_1_output_0 = builder.div(var__model_22_Add_2_output_0, var__model_22_Constant_17_output_0);

    const var__model_22_Sub_1_output_0 = builder.sub(var__model_22_Add_1_output_0, var__model_22_Sub_output_0);

    const var__model_22_Concat_3_output_0 = builder.concat(
        [var__model_22_Div_1_output_0, var__model_22_Sub_1_output_0],
        1,
    );

    let var__model_22_Constant_18_output_0_buffer = new Float32Array(weights_array_buffer, 47242468, 33600 / Float32Array.BYTES_PER_ELEMENT);
    const var__model_22_Constant_18_output_0 = builder.constant(
        {dataType: 'float32', shape: [1, 8400]},
        var__model_22_Constant_18_output_0_buffer
    );

    const var__model_22_Mul_2_output_0 = builder.mul(var__model_22_Concat_3_output_0, var__model_22_Constant_18_output_0);

    const var__model_22_Sigmoid_output_0 = builder.sigmoid(var__model_22_Concat_1_output_0);

    const output0 = builder.concat(
        [var__model_22_Mul_2_output_0, var__model_22_Sigmoid_output_0, var__model_22_Concat_2_output_0],
        1,
    );

    let var_model_22_proto_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 44626176, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_22_proto_cv1_conv_weight_buffer
    );

    let var_model_22_proto_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 45216000, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_proto_cv1_conv_bias_buffer
    );

    const var__model_22_proto_cv1_conv_Conv_output_0 = builder.conv2d(
        var__model_15_cv2_act_Mul_output_0, var_model_22_proto_cv1_conv_weight,
        {
            bias: var_model_22_proto_cv1_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_proto_cv1_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_proto_cv1_conv_Conv_output_0);

    const var__model_22_proto_cv1_act_Mul_output_0 = builder.mul(var__model_22_proto_cv1_conv_Conv_output_0, var__model_22_proto_cv1_act_Sigmoid_output_0);

    let var_model_22_proto_upsample_weight_buffer = new Float32Array(weights_array_buffer, 45216512, 262144 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_upsample_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 2, 2]},
        var_model_22_proto_upsample_weight_buffer
    );

    let var_model_22_proto_upsample_bias_buffer = new Float32Array(weights_array_buffer, 45478656, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_upsample_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_proto_upsample_bias_buffer
    );

    const var__model_22_proto_upsample_ConvTranspose_output_0 = builder.convTranspose2d(
        var__model_22_proto_cv1_act_Mul_output_0, var_model_22_proto_upsample_weight,
        {
            bias: var_model_22_proto_upsample_bias, strides: [2, 2], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, outputSizes: undefined
        }
    );

    let var_model_22_proto_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 45479168, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_22_proto_cv2_conv_weight_buffer
    );

    let var_model_22_proto_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 46068992, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
        var_model_22_proto_cv2_conv_bias_buffer
    );

    const var__model_22_proto_cv2_conv_Conv_output_0 = builder.conv2d(
        var__model_22_proto_upsample_ConvTranspose_output_0, var_model_22_proto_cv2_conv_weight,
        {
            bias: var_model_22_proto_cv2_conv_bias, strides: [1, 1], padding: [1, 1, 1, 1], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_proto_cv2_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_proto_cv2_conv_Conv_output_0);

    const var__model_22_proto_cv2_act_Mul_output_0 = builder.mul(var__model_22_proto_cv2_conv_Conv_output_0, var__model_22_proto_cv2_act_Sigmoid_output_0);

    let var_model_22_proto_cv3_conv_weight_buffer = new Float32Array(weights_array_buffer, 46069504, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv3_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 128, 1, 1]},
        var_model_22_proto_cv3_conv_weight_buffer
    );

    let var_model_22_proto_cv3_conv_bias_buffer = new Float32Array(weights_array_buffer, 46085888, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv3_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
        var_model_22_proto_cv3_conv_bias_buffer
    );

    const var__model_22_proto_cv3_conv_Conv_output_0 = builder.conv2d(
        var__model_22_proto_cv2_act_Mul_output_0, var_model_22_proto_cv3_conv_weight,
        {
            bias: var_model_22_proto_cv3_conv_bias, strides: [1, 1], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1
        }
    );

    const var__model_22_proto_cv3_act_Sigmoid_output_0 = builder.sigmoid(var__model_22_proto_cv3_conv_Conv_output_0);

    const output1 = builder.mul(var__model_22_proto_cv3_conv_Conv_output_0, var__model_22_proto_cv3_act_Sigmoid_output_0);

    // Build graph with output operands.

    this.graph_ = await builder.build({'output0': output0, 'output1': output1});

    // Create graph output tensors.

    this.outputTensors_['output0'] = await this.context_.createTensor(
        {dataType: output0.dataType, shape: output0.shape, readable: true}
    );

    this.outputTensors_['output1'] = await this.context_.createTensor(
        {dataType: output1.dataType, shape: output1.shape, readable: true}
    );

  }

  async run(inputs) {

    // Set input buffers to input tensors using writeTensor (sync)

    for (const name in inputs) {

      if (!(name in this.inputTensors_)) throw new Error(`Unknown input: ${name}`);

      this.context_.writeTensor(this.inputTensors_[name], inputs[name]);

    }

    // Compute the graph

    await this.context_.dispatch(this.graph_, this.inputTensors_, this.outputTensors_);

    // Read output tensors to buffers using readTensor (async)

    const outputs = {};

    for (const name in this.outputTensors_) {

      const tensor = this.outputTensors_[name];

      const buffer = await this.context_.readTensor(tensor);

      let typedArrayCtor;

      switch (tensor.dataType) {

        case 'float32': typedArrayCtor = Float32Array; break;

        case 'float16': typedArrayCtor = Float16Array; break;

        case 'int32': typedArrayCtor = Int32Array; break;

        case 'uint8': typedArrayCtor = Uint8Array; break;

        case 'int8': typedArrayCtor = Int8Array; break;

        case 'uint32': typedArrayCtor = Uint32Array; break;

        case 'int64': typedArrayCtor = BigInt64Array; break;

        case 'uint64': typedArrayCtor = BigUint64Array; break;

        default: throw new Error(`Unhandled tensor dataType: ${tensor.dataType}`);

      }

      outputs[name] = new typedArrayCtor(buffer);

    }

    return outputs;

  }

}