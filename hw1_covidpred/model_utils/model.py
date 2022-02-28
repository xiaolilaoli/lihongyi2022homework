import torch
import torch.nn as nn

class myNet(nn.Module):
    def __init__(self,inDim):
        super(myNet,self).__init__()
        self.fc1 = nn.Linear(inDim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x