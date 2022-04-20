
import os
import torch
from model import NeuralNetwork
import torchvision.transforms as transforms
from PIL import Image

model = NeuralNetwork()
model.load_state_dict(torch.load("cat_vs_dog_.pt"))
model.eval()

image_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

cat = []
dog = []

for path in os.listdir("../TestingData/Cat/"):
    path = "../TestingData/Cat/{}".format(path)
    image = Image.open(path)
    image = image_transformation(image)

    prediction = model(image[None,...]).argmax(dim=1).item()
    cat.append(prediction)

for path in os.listdir("../TestingData/Dog/"):
    path = "../TestingData/Dog/{}".format(path)
    image = Image.open(path)
    image = image_transformation(image)

    prediction = model(image[None,...]).argmax(dim=1).item()
    dog.append(prediction)


print(cat)
print(dog)