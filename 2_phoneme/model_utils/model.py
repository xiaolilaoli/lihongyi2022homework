import torch
import torch.nn as nn

class myNet(nn.Module):
    def __init__(self, inDim, outDim):
        super(myNet,self).__init__()
        self.fc1 = nn.Linear(inDim, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, outDim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x

class myNet2(nn.Module):
    def __init__(self, inDim, outDim):
        super(myNet2,self).__init__()
        self.fc1 = nn.Linear(inDim, 1024)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(1024, 512)
        self.drop1 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(512, outDim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x
def init_para(model):
    def weights_init(model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
    model.apply(weights_init)
    return model