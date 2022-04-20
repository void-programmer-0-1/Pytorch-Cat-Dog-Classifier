
import torch
from model import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("cat_vs_dog.pt"))
model.eval()
model_input = torch.zeros(1 * 3 * 224 * 224) #nchw
torch.onnx.export(model, model_input, 'cat_vs_dog.onnx',
                                    export_params=True,
                                    verbose=True,
                                    opset_version=14,
                                    input_names=["input"],
                                    output_names=["output"])
