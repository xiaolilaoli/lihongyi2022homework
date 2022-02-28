from model_utils.model import myNet
from model_utils.data import covidDataset
from model_utils.train import train_val
from model_utils.evaluate import evaluate
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = 'covid.train.csv'
test_path = 'covid.test.csv'


feature_dim = 6
trainset = covidDataset(train_path,'train',feature_dim=feature_dim)
valset = covidDataset(train_path,'val',feature_dim=feature_dim)
testset = covidDataset(test_path,'test',feature_dim=feature_dim)

def getLoss(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    ''' Calculate loss '''
    regularization_loss = 0
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        # regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)
    return loss(pred, target) + 0.00075 * regularization_loss

loss =  getLoss

config = {
    'n_epochs': 2000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'model_save/model.pth',  # your model will be saved here
}

model = myNet(feature_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)


trainloader = DataLoader(trainset,batch_size=config['batch_size'],shuffle=True)
valloader = DataLoader(valset,batch_size=config['batch_size'],shuffle=True)

train_val(model, trainloader,valloader,optimizer, loss, config['n_epochs'],device,save_=config['save_path'])
evaluate(config['save_path'], testset, 'pred.csv',device)
