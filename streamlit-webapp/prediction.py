import torch
from model import NeuralNetwork
import torchvision.transforms as transforms

def predict(image):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("./cat_vs_dog.pt"))
    model.eval()

    image_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224))
    ])

    image = image_transformation(image)

    prediction = model(image[None,...]).argmax(dim=1).item()
    return prediction

