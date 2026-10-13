class Yolov8nSegWebnn {

  constructor() {

    this.graph_ = null;

    this.context_ = null;

    this.inputTensors_ = {};

    this.outputTensors_ = {};

  }



  async build(contextOptions) {

    // Load weights ArrayBuffer from yolov8n-seg-webnn.bin
        async function loadWeightsArrayBuffer() {
            const response = await fetch('yolov8n-seg-webnn.bin');
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

    let var_model_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 0, 1728 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [16, 3, 3, 3]},
        var_model_0_conv_weight_buffer
    );

    let var_model_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 1728, 64 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [16]},
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

    let var_model_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 1792, 18432 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 16, 3, 3]},
        var_model_1_conv_weight_buffer
    );

    let var_model_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 20224, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_2_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 20352, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_2_cv1_conv_weight_buffer
    );

    let var_model_2_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 24448, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    const var__model_2_Split_output_0_splits = builder.split(var__model_2_cv1_act_Mul_output_0, [16, 16], { axis: 1 });
const var__model_2_Split_output_0 = var__model_2_Split_output_0_splits[0];
const var__model_2_Split_output_1 = var__model_2_Split_output_0_splits[1];

    let var_model_2_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 30848, 9216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [16, 16, 3, 3]},
        var_model_2_m_0_cv1_conv_weight_buffer
    );

    let var_model_2_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 40064, 64 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [16]},
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

    let var_model_2_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 40128, 9216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [16, 16, 3, 3]},
        var_model_2_m_0_cv2_conv_weight_buffer
    );

    let var_model_2_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 49344, 64 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [16]},
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

    let var_model_2_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 24576, 6144 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 48, 1, 1]},
        var_model_2_cv2_conv_weight_buffer
    );

    let var_model_2_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 30720, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_2_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_3_conv_weight_buffer = new Float32Array(weights_array_buffer, 49408, 73728 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_3_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 32, 3, 3]},
        var_model_3_conv_weight_buffer
    );

    let var_model_3_conv_bias_buffer = new Float32Array(weights_array_buffer, 123136, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_3_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_4_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 123392, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_4_cv1_conv_weight_buffer
    );

    let var_model_4_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 139776, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    const var__model_4_Split_output_0_splits = builder.split(var__model_4_cv1_act_Mul_output_0, [32, 32], { axis: 1 });
const var__model_4_Split_output_0 = var__model_4_Split_output_0_splits[0];
const var__model_4_Split_output_1 = var__model_4_Split_output_0_splits[1];

    let var_model_4_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 173056, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_4_m_0_cv1_conv_weight_buffer
    );

    let var_model_4_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 209920, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_4_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 210048, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_4_m_0_cv2_conv_weight_buffer
    );

    let var_model_4_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 246912, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_4_m_1_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 247040, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_4_m_1_cv1_conv_weight_buffer
    );

    let var_model_4_m_1_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 283904, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_4_m_1_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 284032, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_4_m_1_cv2_conv_weight_buffer
    );

    let var_model_4_m_1_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 320896, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_m_1_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_4_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 140032, 32768 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 128, 1, 1]},
        var_model_4_cv2_conv_weight_buffer
    );

    let var_model_4_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 172800, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_4_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_5_conv_weight_buffer = new Float32Array(weights_array_buffer, 321024, 294912 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_5_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 64, 3, 3]},
        var_model_5_conv_weight_buffer
    );

    let var_model_5_conv_bias_buffer = new Float32Array(weights_array_buffer, 615936, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_5_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_6_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 616448, 65536 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 1, 1]},
        var_model_6_cv1_conv_weight_buffer
    );

    let var_model_6_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 681984, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    const var__model_6_Split_output_0_splits = builder.split(var__model_6_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
const var__model_6_Split_output_0 = var__model_6_Split_output_0_splits[0];
const var__model_6_Split_output_1 = var__model_6_Split_output_0_splits[1];

    let var_model_6_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 814080, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_6_m_0_cv1_conv_weight_buffer
    );

    let var_model_6_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 961536, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_6_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 961792, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_6_m_0_cv2_conv_weight_buffer
    );

    let var_model_6_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 1109248, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_6_m_1_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 1109504, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_6_m_1_cv1_conv_weight_buffer
    );

    let var_model_6_m_1_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 1256960, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_6_m_1_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 1257216, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_6_m_1_cv2_conv_weight_buffer
    );

    let var_model_6_m_1_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 1404672, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_m_1_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_6_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 682496, 131072 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 256, 1, 1]},
        var_model_6_cv2_conv_weight_buffer
    );

    let var_model_6_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 813568, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_6_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_7_conv_weight_buffer = new Float32Array(weights_array_buffer, 1404928, 1179648 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_7_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 128, 3, 3]},
        var_model_7_conv_weight_buffer
    );

    let var_model_7_conv_bias_buffer = new Float32Array(weights_array_buffer, 2584576, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_7_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
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

    let var_model_8_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 2585600, 262144 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 256, 1, 1]},
        var_model_8_cv1_conv_weight_buffer
    );

    let var_model_8_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 2847744, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
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

    const var__model_8_Split_output_0_splits = builder.split(var__model_8_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
const var__model_8_Split_output_0 = var__model_8_Split_output_0_splits[0];
const var__model_8_Split_output_1 = var__model_8_Split_output_0_splits[1];

    let var_model_8_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 3243008, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_8_m_0_cv1_conv_weight_buffer
    );

    let var_model_8_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 3832832, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_8_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 3833344, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_8_m_0_cv2_conv_weight_buffer
    );

    let var_model_8_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 4423168, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_8_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 2848768, 393216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 384, 1, 1]},
        var_model_8_cv2_conv_weight_buffer
    );

    let var_model_8_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 3241984, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_8_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
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

    let var_model_9_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 4423680, 131072 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 256, 1, 1]},
        var_model_9_cv1_conv_weight_buffer
    );

    let var_model_9_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 4554752, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_9_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 4555264, 524288 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 512, 1, 1]},
        var_model_9_cv2_conv_weight_buffer
    );

    let var_model_9_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 5079552, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_9_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
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

    let var_model_12_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 5080576, 196608 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 384, 1, 1]},
        var_model_12_cv1_conv_weight_buffer
    );

    let var_model_12_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 5277184, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    const var__model_12_Split_output_0_splits = builder.split(var__model_12_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
const var__model_12_Split_output_0 = var__model_12_Split_output_0_splits[0];
const var__model_12_Split_output_1 = var__model_12_Split_output_0_splits[1];

    let var_model_12_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 5376512, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_12_m_0_cv1_conv_weight_buffer
    );

    let var_model_12_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 5523968, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_12_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 5524224, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_12_m_0_cv2_conv_weight_buffer
    );

    let var_model_12_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 5671680, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_12_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 5277696, 98304 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 192, 1, 1]},
        var_model_12_cv2_conv_weight_buffer
    );

    let var_model_12_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 5376000, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_12_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_15_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 5671936, 49152 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 192, 1, 1]},
        var_model_15_cv1_conv_weight_buffer
    );

    let var_model_15_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 5721088, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    const var__model_15_Split_output_0_splits = builder.split(var__model_15_cv1_act_Mul_output_0, [32, 32], { axis: 1 });
const var__model_15_Split_output_0 = var__model_15_Split_output_0_splits[0];
const var__model_15_Split_output_1 = var__model_15_Split_output_0_splits[1];

    let var_model_15_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 5746176, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_15_m_0_cv1_conv_weight_buffer
    );

    let var_model_15_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 5783040, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_15_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 5783168, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_15_m_0_cv2_conv_weight_buffer
    );

    let var_model_15_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 5820032, 128 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [32]},
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

    let var_model_15_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 5721344, 24576 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 96, 1, 1]},
        var_model_15_cv2_conv_weight_buffer
    );

    let var_model_15_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 5745920, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_15_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_16_conv_weight_buffer = new Float32Array(weights_array_buffer, 5820160, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_16_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_16_conv_weight_buffer
    );

    let var_model_16_conv_bias_buffer = new Float32Array(weights_array_buffer, 5967616, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_16_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_18_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 5967872, 98304 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 192, 1, 1]},
        var_model_18_cv1_conv_weight_buffer
    );

    let var_model_18_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 6066176, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    const var__model_18_Split_output_0_splits = builder.split(var__model_18_cv1_act_Mul_output_0, [64, 64], { axis: 1 });
const var__model_18_Split_output_0 = var__model_18_Split_output_0_splits[0];
const var__model_18_Split_output_1 = var__model_18_Split_output_0_splits[1];

    let var_model_18_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 6165504, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_18_m_0_cv1_conv_weight_buffer
    );

    let var_model_18_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 6312960, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_18_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 6313216, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_18_m_0_cv2_conv_weight_buffer
    );

    let var_model_18_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 6460672, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_18_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 6066688, 98304 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 192, 1, 1]},
        var_model_18_cv2_conv_weight_buffer
    );

    let var_model_18_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 6164992, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_18_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_19_conv_weight_buffer = new Float32Array(weights_array_buffer, 6460928, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_19_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_19_conv_weight_buffer
    );

    let var_model_19_conv_bias_buffer = new Float32Array(weights_array_buffer, 7050752, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_19_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_21_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 7051264, 393216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 384, 1, 1]},
        var_model_21_cv1_conv_weight_buffer
    );

    let var_model_21_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 7444480, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
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

    const var__model_21_Split_output_0_splits = builder.split(var__model_21_cv1_act_Mul_output_0, [128, 128], { axis: 1 });
const var__model_21_Split_output_0 = var__model_21_Split_output_0_splits[0];
const var__model_21_Split_output_1 = var__model_21_Split_output_0_splits[1];

    let var_model_21_m_0_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 7839744, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_21_m_0_cv1_conv_weight_buffer
    );

    let var_model_21_m_0_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 8429568, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_21_m_0_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 8430080, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [128, 128, 3, 3]},
        var_model_21_m_0_cv2_conv_weight_buffer
    );

    let var_model_21_m_0_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 9019904, 512 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_m_0_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [128]},
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

    let var_model_21_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 7445504, 393216 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [256, 384, 1, 1]},
        var_model_21_cv2_conv_weight_buffer
    );

    let var_model_21_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 7838720, 1024 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_21_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [256]},
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

    let var_model_22_proto_cv1_conv_weight_buffer = new Float32Array(weights_array_buffer, 12607616, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_proto_cv1_conv_weight_buffer
    );

    let var_model_22_proto_cv1_conv_bias_buffer = new Float32Array(weights_array_buffer, 12755072, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_22_proto_upsample_weight_buffer = new Float32Array(weights_array_buffer, 12755328, 65536 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_upsample_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 2, 2]},
        var_model_22_proto_upsample_weight_buffer
    );

    let var_model_22_proto_upsample_bias_buffer = new Float32Array(weights_array_buffer, 12820864, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_upsample_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
        var_model_22_proto_upsample_bias_buffer
    );

    const var__model_22_proto_upsample_ConvTranspose_output_0 = builder.convTranspose2d(
        var__model_22_proto_cv1_act_Mul_output_0, var_model_22_proto_upsample_weight,
        {
            bias: var_model_22_proto_upsample_bias, strides: [2, 2], padding: [0, 0, 0, 0], dilations: [1, 1], groups: 1, outputSizes: undefined
        }
    );

    let var_model_22_proto_cv2_conv_weight_buffer = new Float32Array(weights_array_buffer, 12821120, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv2_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_proto_cv2_conv_weight_buffer
    );

    let var_model_22_proto_cv2_conv_bias_buffer = new Float32Array(weights_array_buffer, 12968576, 256 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv2_conv_bias = builder.constant(
        {dataType: 'float32', shape: [64]},
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

    let var_model_22_proto_cv3_conv_weight_buffer = new Float32Array(weights_array_buffer, 12968832, 8192 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_proto_cv3_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 64, 1, 1]},
        var_model_22_proto_cv3_conv_weight_buffer
    );

    let var_model_22_proto_cv3_conv_bias_buffer = new Float32Array(weights_array_buffer, 12977024, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_0_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 12977152, 73728 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 64, 3, 3]},
        var_model_22_cv4_0_0_conv_weight_buffer
    );

    let var_model_22_cv4_0_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 13050880, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_0_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 13051008, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_22_cv4_0_1_conv_weight_buffer
    );

    let var_model_22_cv4_0_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 13087872, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_0_2_weight_buffer = new Float32Array(weights_array_buffer, 13088000, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_0_2_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_22_cv4_0_2_weight_buffer
    );

    let var_model_22_cv4_0_2_bias_buffer = new Float32Array(weights_array_buffer, 13092096, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    const var__model_22_Reshape_output_0 = builder.reshape(
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

    let var_model_22_cv4_1_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 13092224, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 128, 3, 3]},
        var_model_22_cv4_1_0_conv_weight_buffer
    );

    let var_model_22_cv4_1_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 13239680, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_1_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 13239808, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_22_cv4_1_1_conv_weight_buffer
    );

    let var_model_22_cv4_1_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 13276672, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_1_2_weight_buffer = new Float32Array(weights_array_buffer, 13276800, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_1_2_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_22_cv4_1_2_weight_buffer
    );

    let var_model_22_cv4_1_2_bias_buffer = new Float32Array(weights_array_buffer, 13280896, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    const var__model_22_Reshape_1_output_0 = builder.reshape(
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

    let var_model_22_cv4_2_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 13281024, 294912 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 256, 3, 3]},
        var_model_22_cv4_2_0_conv_weight_buffer
    );

    let var_model_22_cv4_2_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 13575936, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_2_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 13576064, 36864 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 3, 3]},
        var_model_22_cv4_2_1_conv_weight_buffer
    );

    let var_model_22_cv4_2_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 13612928, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv4_2_2_weight_buffer = new Float32Array(weights_array_buffer, 13613056, 4096 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv4_2_2_weight = builder.constant(
        {dataType: 'float32', shape: [32, 32, 1, 1]},
        var_model_22_cv4_2_2_weight_buffer
    );

    let var_model_22_cv4_2_2_bias_buffer = new Float32Array(weights_array_buffer, 13617152, 128 / Float32Array.BYTES_PER_ELEMENT);
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

    const var__model_22_Reshape_2_output_0 = builder.reshape(
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

    const var__model_22_Concat_output_0 = builder.concat(
        [var__model_22_Reshape_output_0, var__model_22_Reshape_1_output_0, var__model_22_Reshape_2_output_0],
        2,
    );

    let var_model_22_cv2_0_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 9020416, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_0_0_conv_weight_buffer
    );

    let var_model_22_cv2_0_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 9167872, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv2_0_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 9168128, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_0_1_conv_weight_buffer
    );

    let var_model_22_cv2_0_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 9315584, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv2_0_2_weight_buffer = new Float32Array(weights_array_buffer, 9315840, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_0_2_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_22_cv2_0_2_weight_buffer
    );

    let var_model_22_cv2_0_2_bias_buffer = new Float32Array(weights_array_buffer, 9332224, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv3_0_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 10546432, 184320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [80, 64, 3, 3]},
        var_model_22_cv3_0_0_conv_weight_buffer
    );

    let var_model_22_cv3_0_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 10730752, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
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

    let var_model_22_cv3_0_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 10731072, 230400 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [80, 80, 3, 3]},
        var_model_22_cv3_0_1_conv_weight_buffer
    );

    let var_model_22_cv3_0_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 10961472, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
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

    let var_model_22_cv3_0_2_weight_buffer = new Float32Array(weights_array_buffer, 10961792, 25600 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_0_2_weight = builder.constant(
        {dataType: 'float32', shape: [80, 80, 1, 1]},
        var_model_22_cv3_0_2_weight_buffer
    );

    let var_model_22_cv3_0_2_bias_buffer = new Float32Array(weights_array_buffer, 10987392, 320 / Float32Array.BYTES_PER_ELEMENT);
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

    const var__model_22_Concat_1_output_0 = builder.concat(
        [var__model_22_cv2_0_cv2_0_2_Conv_output_0, var__model_22_cv3_0_cv3_0_2_Conv_output_0],
        1,
    );

    let var_model_22_cv2_1_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 9332480, 294912 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 128, 3, 3]},
        var_model_22_cv2_1_0_conv_weight_buffer
    );

    let var_model_22_cv2_1_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 9627392, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv2_1_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 9627648, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_1_1_conv_weight_buffer
    );

    let var_model_22_cv2_1_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 9775104, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv2_1_2_weight_buffer = new Float32Array(weights_array_buffer, 9775360, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_1_2_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_22_cv2_1_2_weight_buffer
    );

    let var_model_22_cv2_1_2_bias_buffer = new Float32Array(weights_array_buffer, 9791744, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv3_1_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 10987712, 368640 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [80, 128, 3, 3]},
        var_model_22_cv3_1_0_conv_weight_buffer
    );

    let var_model_22_cv3_1_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 11356352, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
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

    let var_model_22_cv3_1_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 11356672, 230400 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [80, 80, 3, 3]},
        var_model_22_cv3_1_1_conv_weight_buffer
    );

    let var_model_22_cv3_1_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 11587072, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
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

    let var_model_22_cv3_1_2_weight_buffer = new Float32Array(weights_array_buffer, 11587392, 25600 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_1_2_weight = builder.constant(
        {dataType: 'float32', shape: [80, 80, 1, 1]},
        var_model_22_cv3_1_2_weight_buffer
    );

    let var_model_22_cv3_1_2_bias_buffer = new Float32Array(weights_array_buffer, 11612992, 320 / Float32Array.BYTES_PER_ELEMENT);
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

    const var__model_22_Concat_2_output_0 = builder.concat(
        [var__model_22_cv2_1_cv2_1_2_Conv_output_0, var__model_22_cv3_1_cv3_1_2_Conv_output_0],
        1,
    );

    let var_model_22_cv2_2_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 9792000, 589824 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 256, 3, 3]},
        var_model_22_cv2_2_0_conv_weight_buffer
    );

    let var_model_22_cv2_2_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 10381824, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv2_2_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 10382080, 147456 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 3, 3]},
        var_model_22_cv2_2_1_conv_weight_buffer
    );

    let var_model_22_cv2_2_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 10529536, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv2_2_2_weight_buffer = new Float32Array(weights_array_buffer, 10529792, 16384 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv2_2_2_weight = builder.constant(
        {dataType: 'float32', shape: [64, 64, 1, 1]},
        var_model_22_cv2_2_2_weight_buffer
    );

    let var_model_22_cv2_2_2_bias_buffer = new Float32Array(weights_array_buffer, 10546176, 256 / Float32Array.BYTES_PER_ELEMENT);
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

    let var_model_22_cv3_2_0_conv_weight_buffer = new Float32Array(weights_array_buffer, 11613312, 737280 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_0_conv_weight = builder.constant(
        {dataType: 'float32', shape: [80, 256, 3, 3]},
        var_model_22_cv3_2_0_conv_weight_buffer
    );

    let var_model_22_cv3_2_0_conv_bias_buffer = new Float32Array(weights_array_buffer, 12350592, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_0_conv_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
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

    let var_model_22_cv3_2_1_conv_weight_buffer = new Float32Array(weights_array_buffer, 12350912, 230400 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_1_conv_weight = builder.constant(
        {dataType: 'float32', shape: [80, 80, 3, 3]},
        var_model_22_cv3_2_1_conv_weight_buffer
    );

    let var_model_22_cv3_2_1_conv_bias_buffer = new Float32Array(weights_array_buffer, 12581312, 320 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_1_conv_bias = builder.constant(
        {dataType: 'float32', shape: [80]},
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

    let var_model_22_cv3_2_2_weight_buffer = new Float32Array(weights_array_buffer, 12581632, 25600 / Float32Array.BYTES_PER_ELEMENT);
    const var_model_22_cv3_2_2_weight = builder.constant(
        {dataType: 'float32', shape: [80, 80, 1, 1]},
        var_model_22_cv3_2_2_weight_buffer
    );

    let var_model_22_cv3_2_2_bias_buffer = new Float32Array(weights_array_buffer, 12607232, 320 / Float32Array.BYTES_PER_ELEMENT);
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

    const var__model_22_Concat_3_output_0 = builder.concat(
        [var__model_22_cv2_2_cv2_2_2_Conv_output_0, var__model_22_cv3_2_cv3_2_2_Conv_output_0],
        1,
    );

    const var__model_22_Reshape_3_output_0 = builder.reshape(
        var__model_22_Concat_1_output_0,
        (() => {
        const shape = [1, 144, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_Concat_1_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Reshape_4_output_0 = builder.reshape(
        var__model_22_Concat_2_output_0,
        (() => {
        const shape = [1, 144, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_Concat_2_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Reshape_5_output_0 = builder.reshape(
        var__model_22_Concat_3_output_0,
        (() => {
        const shape = [1, 144, -1];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_Concat_3_output_0.shape.reduce((a, b) => a * b, 1);
            const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
            const idx = shape.indexOf(-1);
            shape[idx] = totalInput / known;
        }
        return shape;
    })()
    );

    const var__model_22_Concat_4_output_0 = builder.concat(
        [var__model_22_Reshape_3_output_0, var__model_22_Reshape_4_output_0, var__model_22_Reshape_5_output_0],
        2,
    );

    const var__model_22_Split_output_0_splits = builder.split(var__model_22_Concat_4_output_0, [64, 80], { axis: 1 });
const var__model_22_Split_output_0 = var__model_22_Split_output_0_splits[0];
const var__model_22_Split_output_1 = var__model_22_Split_output_0_splits[1];

    const var__model_22_dfl_Reshape_output_0 = builder.reshape(
        var__model_22_Split_output_0,
        (() => {
        const shape = [1, 4, 16, 8400];
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
            const count = shape.filter(v => v === -1).length;
            if (count !== 1) {
                throw new Error('Only one -1 is allowed in reshape shape');
            }
            const totalInput = var__model_22_Split_output_0.shape.reduce((a, b) => a * b, 1);
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

    let var_model_22_dfl_conv_weight_buffer = new Float32Array(weights_array_buffer, 12607552, 64 / Float32Array.BYTES_PER_ELEMENT);
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

    let var__model_22_Constant_12_output_0_buffer = new Float32Array(weights_array_buffer, 13651116, 67200 / Float32Array.BYTES_PER_ELEMENT);
    const var__model_22_Constant_12_output_0 = builder.constant(
        {dataType: 'float32', shape: [1, 2, 8400]},
        var__model_22_Constant_12_output_0_buffer
    );

    const var__model_22_Sub_output_0 = builder.sub(var__model_22_Constant_12_output_0, var__model_22_Slice_output_0);

    const var__model_22_Add_1_output_0 = builder.add(var__model_22_Constant_12_output_0, var__model_22_Slice_1_output_0);

    const var__model_22_Add_2_output_0 = builder.add(var__model_22_Sub_output_0, var__model_22_Add_1_output_0);

    let var__model_22_Constant_14_output_0_buffer = new Float32Array(weights_array_buffer, 13617496, 4 / Float32Array.BYTES_PER_ELEMENT);
    const var__model_22_Constant_14_output_0 = builder.constant(
        {dataType: 'float32', shape: []},
        var__model_22_Constant_14_output_0_buffer
    );

    const var__model_22_Div_1_output_0 = builder.div(var__model_22_Add_2_output_0, var__model_22_Constant_14_output_0);

    const var__model_22_Sub_1_output_0 = builder.sub(var__model_22_Add_1_output_0, var__model_22_Sub_output_0);

    const var__model_22_Concat_5_output_0 = builder.concat(
        [var__model_22_Div_1_output_0, var__model_22_Sub_1_output_0],
        1,
    );

    let var__model_22_Constant_15_output_0_buffer = new Float32Array(weights_array_buffer, 13617500, 33600 / Float32Array.BYTES_PER_ELEMENT);
    const var__model_22_Constant_15_output_0 = builder.constant(
        {dataType: 'float32', shape: [1, 8400]},
        var__model_22_Constant_15_output_0_buffer
    );

    const var__model_22_Mul_2_output_0 = builder.mul(var__model_22_Concat_5_output_0, var__model_22_Constant_15_output_0);

    const var__model_22_Sigmoid_output_0 = builder.sigmoid(var__model_22_Split_output_1);

    const output0 = builder.concat(
        [var__model_22_Mul_2_output_0, var__model_22_Sigmoid_output_0, var__model_22_Concat_output_0],
        1,
    );

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