from tqdm import tqdm
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from model_utils.data import samplePlot, get_semi_loader
import random
from torch.utils.data import Dataset, DataLoader
def train_val(para):

########################################################
    model = para['model']
    semi_loader = para['semi_loader']
    train_loader =para['train_loader']
    val_loader = para['val_loader']
    scheduler = para['scheduler']
    optimizer = para['optimizer']
    loss = para['loss']
    epoch = para['epoch']
    device = para['device']
    save_path = para['save_path']
    save_acc = para['save_acc']
    pre_path = para['pre_path']
    max_acc = para['max_acc']
    val_epoch = para['val_epoch']
    acc_thres = para['acc_thres']
    conf_thres = para['conf_thres']
    do_semi = para['do_semi']

    semi_epoch = 10
###################################################
    no_label_Loader = None
    if pre_path != None:
        model = torch.load(pre_path)
    model = model.to(device)
    # model = torch.nn.DataParallel(model).to(device)
    model.device_ids = [0,1]
    plt_train_loss = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_acc = []
    plt_semi_acc = []
    val_rel = []
    max_acc = 0

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_loss = 0.0
        semi_acc = 0

        for data in tqdm(train_loader):
            optimizer.zero_grad()
            x , target = data[0].to(device), data[1].to(device)
            num = random.randint(0,10000)
            if num == 99:
                samplePlot(train_loader,True,isbat=False,ori =True)
            pred = model(x)
            bat_loss = loss(pred, target)
            bat_loss.backward()
            scheduler.step()
            optimizer.step()
            train_loss += bat_loss.item()    #.detach 表示去掉梯度
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1)== data[1].numpy())

        if no_label_Loader != None:
            for data in tqdm(no_label_Loader):
                optimizer.zero_grad()
                x , target = data[0].to(device), data[1].to(device)
                num = random.randint(0, 10000)
                if num == 99:
                    samplePlot(train_loader,True,isbat=False,ori =True)
                pred = model(x)
                bat_loss = loss(pred, target)
                bat_loss.backward()
                scheduler.step()
                optimizer.step()

                semi_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1)== data[1].numpy())
            plt_semi_acc .append(semi_acc/no_label_Loader.dataset.__len__())
            print('semi_acc:', plt_semi_acc[-1])
        plt_train_loss . append(train_loss/train_loader.dataset.__len__())
        plt_train_acc.append(train_acc/train_loader.dataset.__len__())
        if i % val_epoch == 0:
            model.eval()
            with torch.no_grad():
                for valdata in val_loader:
                    val_x , val_target = valdata[0].to(device), valdata[1].to(device)
                    val_pred = model(val_x)
                    val_bat_loss = loss(val_pred, val_target)
                    val_loss += val_bat_loss.cpu().item()
                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == valdata[1].numpy())
                    val_rel.append(val_pred)
            val_acc = val_acc/val_loader.dataset.__len__()
            if val_acc > max_acc:
                torch.save(model, save_path)
                max_acc = val_acc
            plt_val_loss.append(val_loss/val_loader.dataset.__len__())
            plt_val_acc.append(val_acc)
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | valAcc: %3.6f valLoss: %3.6f  ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1])
                  )
        else:
            plt_val_loss.append(plt_val_loss[-1])
            plt_val_acc.append(plt_val_acc[-1])


        if do_semi and plt_val_acc[-1] > acc_thres and i % semi_epoch==0:
            no_label_Loader = get_semi_loader(semi_loader, model, device, conf_thres)


    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.savefig('acc.png')
    plt.show()
