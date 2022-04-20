
import onnx
from onnx_tf.backend import prepare
import numpy as np

import os
import torch
from model import NeuralNetwork
import torchvision.transforms as transforms
from PIL import Image

model = NeuralNetwork()
model.load_state_dict(torch.load("../weights/cat_vs_dog.pt"))
model.eval()

image_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

onnx_model = onnx.load("../weights/cat_vs_dog.onnx")  
tf_rep = prepare(onnx_model) 

cat = []
dog = []

for path in os.listdir("../TestData/Cat/"):
    path = "../TestData/Cat/{}".format(path)
    image = Image.open(path)
    image = image_transformation(image).numpy()

    prediction = np.array(tf_rep.run(image[None,...]))[0].argmax(axis=1).item()
    cat.append(prediction)

for path in os.listdir("../TestData/Dog/"):
    path = "../TestData/Dog/{}".format(path)
    image = Image.open(path)
    image = image_transformation(image).numpy()

    prediction = np.array(tf_rep.run(image[None,...]))[0].argmax(axis=1).item()
    dog.append(prediction)


print(cat)
print(dog)