import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split



class phonemeDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x)
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index].float(), self.y[index]
        else:
            return self.x[index].float()

    def __len__(self):
        return len(self.x)

def getDataset(path,mode):
    if mode == 'train' or mode == 'trainAndVal':
        traindataPath = path + '/' + 'train_11.npy'
        trainlabelPath = path + '/' + 'train_label_11.npy'
        x = np.load(traindataPath)
        y = np.load(trainlabelPath)
        # x = x[:, 3*39:8*39]
        y = y.astype(np.int)
        if mode == 'trainAndVal':
            trainAndValset = phonemeDataset(x, y)
            return trainAndValset
        else:
            train_x,val_x, train_y, val_y = train_test_split(x, y, test_size=0.05,shuffle=True,random_state=0)
            trainset = phonemeDataset(train_x, train_y)
            valset = phonemeDataset(val_x, val_y)
            return trainset, valset

    elif mode == 'test':
        testdataPath = path + '/' + 'test_11.npy'
        x = np.load(testdataPath)
        # x = x[:, 3*39:8*39]
        testset = phonemeDataset(x)
        return testset

def getDataLoader(path, mode, batchSize):
    assert mode in ['train', 'test', 'trainAndVal']
    if mode == 'train':
        trainset, valset = getDataset(path, mode)
        trainloader = DataLoader(trainset,batch_size=batchSize, shuffle=True)
        valloader = DataLoader(valset,batch_size=batchSize, shuffle=True)
        return trainloader,valloader
    elif mode == 'trainAndVal':
        trainAndValset = getDataset(path, mode)
        trainAndValloader = DataLoader(trainAndValset,batch_size=batchSize, shuffle=True)
        return trainAndValloader

    elif mode == 'test':
        testset = getDataset(path, mode)
        testLoader = DataLoader(testset, batch_size=batchSize, shuffle=False)
        return testLoader