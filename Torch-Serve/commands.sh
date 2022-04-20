

torch-model-archiver --model-name PetClassifier --version 1.0 --model-file model.py --serialized-file petClassifier.pt --export-path model_store --handler handler.py 
torchserve --start --model-store model_store --models PetClassifier.mar --ncs --ts-config config.properties