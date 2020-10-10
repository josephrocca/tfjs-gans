class GAN {

  constructor(modelPath) {
    this.model = {};
    this.modelPath = modelPath;
    this.modelInfo = {};
  }

  async init() {
    this.modelInfo = await fetch(this.modelPath).then(r => r.json());
    this.latentSize = this.modelInfo.modelLatentDim;
    if(this.modelInfo.modelType === "graph") {
      this.model = await tf.loadGraphModel(this.modelInfo.model);
    } else if(this.modelInfo.modelType === "layers") {
      this.model = await tf.loadLayersModel(this.modelInfo.model);
    } else {
      throw new Error("You must specify 'graph' or 'layers' for the modelType parameter of `manifest.json`. See  js.tensorflow.org/api/latest/#loadGraphModel  and  js.tensorflow.org/api/latest/#loadLayersModel");
    }
  }

  async generate(latentVector) {
    if(!latentVector) latentVector = tf.randomNormal([1, this.latentSize]);
    else latentVector = tf.tensor(latentVector, [1, this.latentSize]);

    let transpose = this.modelInfo.transpose || [0, 1, 2]; // such that shape=[128, 128, 3]
    let imageTensor = this.model.predict(latentVector).squeeze().transpose(transpose);
    if(this.modelInfo.outputRange && this.modelInfo.outputRange[0] === -1) imageTensor = imageTensor.div(tf.scalar(2)).add(tf.scalar(0.5));

    const raw = await tf.browser.toPixels(imageTensor);
    const blob = await this.rawToBlob(raw, imageTensor.shape[0], imageTensor.shape[1]);

    imageTensor.dispose();

    return {raw, blob};

  }

  async rawToBlob(raws, x, y) {
    const arr = Array.from(raws)
    const canvas = new OffscreenCanvas(x, y);
    const ctx = canvas.getContext("2d");

    const imgData = ctx.createImageData(x, y);
    const { data } = imgData;

    for (let i = 0; i < x * y * 4; i += 1 ) data[i] = arr[i];
    ctx.putImageData(imgData, 0, 0);

    return canvas.convertToBlob({type: "image/jpeg", quality: 0.95});
  };

}