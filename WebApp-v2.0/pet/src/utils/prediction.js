import { InferenceSession } from "onnxruntime-web";

export default async  function predict(inputTensor){

    try{
        const session = await InferenceSession.create("./cat_vs_dog.onnx",{executionProviders : ["webgl"],graphOptimizationLevel:["all"]});
        const feeds = { "input": inputTensor };
        const predicted = await session.run(feeds);
        return predicted;
    }
    catch (error) {
        return `failed to inference ONNX model: ${error}.`;
    }
}
    