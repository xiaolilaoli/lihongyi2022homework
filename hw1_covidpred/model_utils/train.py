import numpy as np
import torch
from torch.utils.data import  DataLoader
import time
import matplotlib.pyplot as plt
from torch import optim
def train_val(model, trainset, valset,optimizer, loss, epoch, device, save_):

    # trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    # valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    model = model.to(device)
    plt_train_loss = []
    plt_val_loss = []
    val_rel = []
    min_val_loss = 100000

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        # if i > 30:
        #     optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # regularization_loss = 0
        # for param in model.parameters():
        #     # TODO: you may implement L1/L2 regularization here
        #     # 使用L2正则项
        #     # regularization_loss += torch.sum(abs(param))
        #     regularization_loss += torch.sum(param ** 2)
        for data in trainset:
            optimizer.zero_grad()
            x , target = data[0].to(device), data[1].to(torch.float32).to(device)
            pred = model(x)
            bat_loss = loss(pred, target, model)
            bat_loss.backward()
            optimizer.step()
            train_loss += bat_loss.detach().cpu().item()

        plt_train_loss . append(train_loss/trainset.__len__())

        model.eval()
        with torch.no_grad():
            for data in valset:
                val_x , val_target = data[0].to(device), data[1].to(device)
                val_pred = model(val_x)
                val_bat_loss = loss(val_pred, val_target, model)
                val_loss += val_bat_loss
                val_rel.append(val_pred)
        if val_loss < min_val_loss:
            torch.save(model, save_)

        plt_val_loss . append(val_loss/valset.__len__())

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %3.6f | valLoss: %3.6f' % \
              (i, epoch, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1])
              )

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()
