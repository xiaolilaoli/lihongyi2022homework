
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from model_utils.model import model_Datapara
from sklearn.metrics import confusion_matrix


from torch.cuda.amp import autocast as autocast, GradScaler





def train_val(para):

    ########################################################
    model = para['model']
    train_loader =para['train_loader']
    val_loader = para['val_loader']
    scheduler = para['scheduler']
    optimizer = para['optimizer']
    loss = para['loss']
    epochs = para['epochs']
    device = para['device']
    save_path = para['save_path']
    save_acc = para['save_acc']
    pre_path = para['pre_path']
    val_epoch = para['val_epoch']
    ###################################################
    # model = model_Datapara(model, device, pre_path)
    model = model.to(device)
    #################################################
    plt_train_loss = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_acc = []
    val_rel = []
    max_acc = 0
    pat = 0
    for i in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_loss = 0.0
        for batch in tqdm(train_loader):
            model.zero_grad()
            mel, labels = batch[0], batch[1]
            mel = mel.to(device)
            labels = labels.to(device)
            with autocast():
                pred = model(mel)
                bat_loss = loss(pred, labels)
            bat_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss += bat_loss.item()    #.detach 表示去掉梯度
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1)== labels.cpu().numpy())
        plt_train_loss . append(train_loss/train_loader.dataset.__len__())
        plt_train_acc.append(train_acc/train_loader.dataset.__len__())
        if i % val_epoch == 0:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    val_mels , val_labels = batch[0], batch[1]
                    val_mels= val_mels.to(device)
                    val_labels = val_labels.to(device)
                    with autocast():
                        val_pred = model(val_mels)
                        val_bat_loss = loss(val_pred, val_labels)
                    val_loss += val_bat_loss.cpu().item()
                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == val_labels.cpu().numpy())
                    val_rel.append(val_pred)

            if val_acc > max_acc:
                pat = 0
                torch.save(model, save_path)
                max_acc = val_acc
                if save_acc:
                    with open(save_path+'acc.txt', 'w') as f:
                        f.write('-%.2f'%(max_acc/val_loader.dataset.__len__()))
            else:
                pat += 1
                if pat == 50:
                    break
            plt_val_loss.append(val_loss/val_loader.dataset.__len__())
            plt_val_acc.append(val_acc/val_loader.dataset.__len__())
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | valAcc: %3.6f valLoss: %3.6f  ' % \
                  (i, epochs, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1])
                  )
            if i % 50 == 0:
                torch.save(model, save_path+'-epoch:'+str(i)+ '-%.2f'%plt_val_acc[-1])
        else:
            plt_val_loss.append(plt_val_loss[-1])
            plt_val_acc.append(plt_val_acc[-1])
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f   ' % \
                  (i, epochs, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1])
                  )
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
