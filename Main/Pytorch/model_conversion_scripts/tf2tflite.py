
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("../weights/cat_vs_dog/")
tflite_model = converter.convert()

with open("../weights/cat_vs_dog.tflite","wb") as f:
    f.write(tflite_model)

