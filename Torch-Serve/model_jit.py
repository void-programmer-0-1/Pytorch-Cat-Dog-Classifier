from statistics import mode
import model
import torch

model = model.NeuralNetwork()
model.load_state_dict(torch.load("cat_vs_dog.pt"))

jit_model = torch.jit.script(model)
torch.jit.save(jit_model,"petClassifier.pt")
