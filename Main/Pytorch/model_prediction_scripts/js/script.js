// https://stackoverflow.com/questions/64931078/how-do-i-run-my-model-json-file-on-the-web-to-test-predictions

async function predict()
{
    user_image = document.getElementById("user-image");

    let input_tensor = tf.browser.fromPixels(user_image,3);
                                                            
    input_tensor = input_tensor.reshape([1,3,224,224]);
    input_tensor = input_tensor.toFloat();
                                                           
    const model = await tf.loadGraphModel("./tfjs_model/model.json");
    const prediction = model.predict(input_tensor).dataSync();
    console.log(prediction);
    
    let classes = {0:"cat",1:"dog"}

    document.getElementById("result").innerHTML = classes[tf.argMax(prediction).dataSync()[0]];
}

document.getElementById("predict-btn").addEventListener("click",() => {
    predict();
});