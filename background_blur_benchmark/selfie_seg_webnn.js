// Pure WebNN MediaPipe Selfie Segmentation (256×256) graph builder.
// Extracted from selfie_segmentation/ — runs via navigator.ml + MLGraphBuilder.
// Input:  [1,3,256,256] NCHW float32, normalized (pixel-127.5)/127.5  → [-1,1]
// Output: [1,256,256,1] sigmoid person mask (Float32Array of 256*256)
class SelfieSegmentationWebNN {
  constructor() {
    this._B = null;
    this._wbuf = null; this._winfo = null;
    this._bbuf = null; this._binfo = null;
    this._addB = null; this._mulA = null;
    this._ctx = null; this._graph = null;
    this._inT = null; this._outT = null;
  }

  async load(deviceType) {
    const [wb, wj, bb, bj] = await Promise.all([
      fetch('weights_nchw.bin').then(r => r.arrayBuffer()),
      fetch('weights_nchw.json').then(r => r.json()),
      fetch('biases.bin').then(r => r.arrayBuffer()),
      fetch('biases.json').then(r => r.json()),
    ]);
    this._wbuf = wb; this._winfo = wj; this._bbuf = bb; this._binfo = bj;

    if (typeof MLTensorUsage === 'undefined') {
      window.MLTensorUsage = { WEBGPU_INTEROP: 1, READ: 2, WRITE: 4 };
    }

    const ctx = await navigator.ml.createContext({ deviceType });
    this._ctx = ctx;
    this._B = new MLGraphBuilder(ctx);
    this._addB = this._B.constant({ dataType: 'float32', shape: [1,1,1,1] }, new Float32Array([3]));
    this._mulA = this._B.constant({ dataType: 'float32', shape: [] }, new Float32Array([0.1666666716337204]));

    const out = this._buildGraph();
    this._graph = await this._B.build({ 'segment_back': out });

    this._inT = await ctx.createTensor({ dataType: 'float32', shape: [1,3,256,256], writable: true });
    this._outT = await ctx.createTensor({ dataType: 'float32', shape: [1,256,256,1], readable: true });
  }

  async run(inputNCHW) {
    this._ctx.writeTensor(this._inT, inputNCHW);
    this._ctx.dispatch(this._graph, { 'input': this._inT }, { 'segment_back': this._outT });
    const buf = await this._ctx.readTensor(this._outT);
    return new Float32Array(buf);
  }

  // conv2d helper with bias (weights keyed by conv index)
  _C(input, idx, act, opts) {
    opts = opts || {};
    const wi = this._winfo['conv' + idx];
    const w = this._B.constant({ shape: wi.shape, dataType: 'float32' },
      new Float32Array(this._wbuf.slice(wi.dataOffset, wi.dataOffset + wi.byteLength)));
    const bi = this._binfo['conv' + idx];
    opts.bias = this._B.constant({ shape: bi.shape, dataType: 'float32' },
      new Float32Array(this._bbuf.slice(bi.dataOffset, bi.dataOffset + bi.byteLength)));
    const c = this._B.conv2d(input, w, opts);
    if (act === 'relu') return this._B.relu(c);
    if (act === 'sigmoid') return this._B.sigmoid(c);
    return c;
  }

  // h-swish block
  _H(input, idx, opts) {
    const c = this._C(input, idx, '', opts);
    const a = this._B.add(c, this._addB);
    const cl = this._B.clamp(a, { minValue: 0, maxValue: 6 });
    const m = this._B.mul(this._mulA, cl);
    return this._B.mul(c, m);
  }

  // squeeze-and-excitation block
  _S(input, idx, str, skip) {
    const s = [str, str];
    const p = this._B.averagePool2d(input, { windowDimensions: s, strides: s });
    const c1 = this._C(p, idx, 'relu');
    const c2 = this._C(c1, idx + 1, 'sigmoid');
    return this._B.mul(skip || input, c2);
  }

  _buildGraph() {
    const B = this._B;
    const sv = [2, 2];

    const inp = B.input('input', { dataType: 'float32', shape: [1,3,256,256] });

    const s0 = this._H(inp, 0, { strides: sv, padding: [0,1,0,1] });
    const c1 = this._C(s0, 1, 'relu');
    const c2 = this._C(c1, 2, 'relu', { strides: sv, padding: [0,1,0,1], groups: 16 });
    let x = this._S(c2, 3, 64);
    const c5 = this._C(x, 5, '');
    const c6 = this._C(c5, 6, 'relu');
    const c7 = this._C(c6, 7, 'relu', { strides: sv, padding: [0,1,0,1], groups: 72 });
    const c8 = this._C(c7, 8, '');
    const c9 = this._C(c8, 9, 'relu');
    const c10 = this._C(c9, 10, 'relu', { padding: [1,1,1,1], groups: 88 });
    const c11 = this._C(c10, 11, '');
    const a0 = B.add(c11, c8);
    x = this._H(a0, 12);
    x = this._H(x, 13, { strides: sv, padding: [1,2,1,2], groups: 96 });
    x = this._S(x, 14, 16);
    const c16 = this._C(x, 16, '');
    x = this._H(c16, 17);
    x = this._H(x, 18, { padding: [2,2,2,2], groups: 128 });
    x = this._S(x, 19, 16);
    const c21 = this._C(x, 21, '');
    const a1 = B.add(c21, c16);
    x = this._H(a1, 22);
    x = this._H(x, 23, { padding: [2,2,2,2], groups: 128 });
    x = this._S(x, 24, 16);
    const c26 = this._C(x, 26, '');
    const a2 = B.add(c26, a1);
    x = this._H(a2, 27);
    x = this._H(x, 28, { padding: [2,2,2,2], groups: 96 });
    x = this._S(x, 29, 16);
    const c31 = this._C(x, 31, '');
    const a3 = B.add(c31, a2);
    x = this._H(a3, 32);
    x = this._H(x, 33, { padding: [2,2,2,2], groups: 96 });
    x = this._S(x, 34, 16);
    const c36 = this._C(x, 36, '');
    const a4 = B.add(c36, a3);
    const c37 = this._C(a4, 37, 'relu');
    const p = B.averagePool2d(a4, { windowDimensions: [16,16], strides: [16,16] });
    const c38 = this._C(p, 38, 'sigmoid');
    const m0 = B.mul(c37, c38);
    x = B.resample2d(m0, { sizes: [32,32], mode: 'linear', axes: [2,3] });
    const c39 = this._C(x, 39, '');
    const a5 = B.add(c39, a0);
    x = this._S(a5, 40, 32, a0);
    const a6 = B.add(x, c39);
    const c42 = this._C(a6, 42, 'relu');
    const c43 = this._C(c42, 43, 'relu', { padding: [1,1,1,1], groups: 24 });
    const a7 = B.add(c42, c43);
    x = B.resample2d(a7, { sizes: [64,64], mode: 'linear', axes: [2,3] });
    const c44 = this._C(x, 44, '');
    const a8 = B.add(c5, c44);
    x = this._S(a8, 45, 64, c5);
    const a9 = B.add(x, c44);
    const c47 = this._C(a9, 47, 'relu');
    const c48 = this._C(c47, 48, 'relu', { padding: [1,1,1,1], groups: 16 });
    const a10 = B.add(c47, c48);
    x = B.resample2d(a10, { sizes: [128,128], mode: 'linear', axes: [2,3] });
    const c49 = this._C(x, 49, '');
    const a11 = B.add(s0, c49);
    x = this._S(a11, 50, 128, s0);
    const a12 = B.add(x, c49);
    const c52 = this._C(a12, 52, 'relu');
    const c53 = this._C(c52, 53, 'relu', { padding: [1,1,1,1], groups: 16 });
    const a13 = B.add(c52, c53);

    // convTranspose → 256×256, sigmoid
    const ctwi = this._winfo.convTranspose0;
    const ctW = B.constant({ shape: ctwi.shape, dataType: 'float32' },
      new Float32Array(this._wbuf.slice(ctwi.dataOffset, ctwi.dataOffset + ctwi.byteLength)));
    const ctB = B.constant({ dataType: 'float32', shape: [1] }, new Float32Array([0.53271484375]));
    const ct = B.convTranspose2d(a13, ctW, { bias: ctB, strides: [2,2], outputSizes: [256,256], filterLayout: 'iohw' });
    const sig = B.sigmoid(ct);
    return B.reshape(sig, [1,256,256,1]);
  }
}
