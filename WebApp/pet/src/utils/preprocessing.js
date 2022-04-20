import { Tensor } from 'onnxruntime-web';

export default  function Preprocessing(CanvasRef){

    const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());
    const imageBufferData = CanvasRef.current.getContext("2d").getImageData(0,0,224,224).data;
    
    for (let i = 0; i < imageBufferData.length; i += 4) {
        redArray.push(imageBufferData[i]);
        greenArray.push(imageBufferData[i + 1]);
        blueArray.push(imageBufferData[i + 2]);
    }
    
    const transposedData = redArray.concat(greenArray).concat(blueArray);
    const l = transposedData.length;
    let i = 0;
    
    const float32Data = new Float32Array(1 * 3 * 224 * 224);    // for pytorch [batch_size, channels, height, width].
    for (i = 0; i < l; i++) {
        float32Data[i] = transposedData[i] / 225; 
    }

    const inputTensor = new Tensor("float32", float32Data , [1 * 3 * 224 * 224]);
    return inputTensor;
}