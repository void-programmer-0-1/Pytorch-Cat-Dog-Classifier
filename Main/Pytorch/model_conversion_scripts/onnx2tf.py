
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("../weights/cat_vs_dog.onnx")  
tf_rep = prepare(onnx_model)                                                        # prepare tf representation
tf_rep.export_graph("../weights/cat_vs_dog/")                                   # export the model (SavedModel format)



# https://stackoverflow.com/questions/58834684/how-could-i-convert-onnx-model-to-tensorflow-saved-model
# https://github.com/onnx/onnx-tensorflow

