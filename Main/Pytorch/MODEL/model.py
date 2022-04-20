
import torch.nn as nn

class NeuralNetwork(nn.Module):

  def __init__(self):
    super(NeuralNetwork,self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(3,16,3,2,0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Dropout2d(p=0.25)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(16,32,3,2,0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Dropout2d(p=0.25)
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(32,64,3,2,0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Dropout2d(p=0.25)
    )

    self.FC = nn.Sequential(
        nn.Linear(576,512),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        nn.Linear(512,2),
    )    

  def forward(self,x):
    x = x.reshape(1,3,224,224)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = x.view(x.size(0),-1)
    x = self.FC(x)

    return x