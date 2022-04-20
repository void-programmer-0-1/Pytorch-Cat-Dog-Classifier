import torch
import torchvision.transforms as transforms
import os
from ts.torch_handler.base_handler import BaseHandler
import io
from PIL import Image

class ModelHandler(BaseHandler):

    def __init__(self):
        self.model = None

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir,serialized_file)
        self.model = torch.jit.load(model_pt_path,map_location="cpu")
    
    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        
        preprocessed_data = Image.open(io.BytesIO(preprocessed_data))
    
        transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        return transformation(preprocessed_data)

    def inference(self, data):
        prediction = self.model.forward(data)
        return prediction

    def postprocess(self, data):
        postprocessed_data = data 
        class_id = postprocessed_data.argmax(dim=1).tolist()
        return class_id

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input[None,...])
        prediction =  self.postprocess(model_output)
        return prediction

