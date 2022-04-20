
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime
import numpy as np
import os

classes = {0:"cat",1:"dog"}

transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
])

cat = []
dog = []

for path in os.listdir("../TestData/Cat/"):

    path = "../TestData/Cat/{}".format(path)
    image = Image.open(path)
    image = transformations(image)
    image = image.numpy()

    onnx_session = onnxruntime.InferenceSession("../weights/cat_vs_dog.onnx")
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    prediction = np.array(onnx_session.run(None,{input_name:image[None,...]})).argmax()
    cat.append(classes[prediction])


for path in os.listdir("../TestData/Dog/"):

    path = "../TestData/Dog/{}".format(path)
    image = Image.open(path)
    image = transformations(image)
    image = image.numpy()

    onnx_session = onnxruntime.InferenceSession("../weights/cat_vs_dog.onnx")
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    prediction = np.array(onnx_session.run(None,{input_name:image[None,...]})).argmax()
    dog.append(classes[prediction])

print(cat)
print(dog)


