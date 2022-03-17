
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def train_val(model, trainloader, valloader,optimizer, scheduler, loss, epoch, device, save_):

    # trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    # valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    model = model.to(device)
    plt_train_loss = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_acc = []
    val_rel = []
    max_acc = 0

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            x , target = data[0].to(device), data[1].to(device)
            pred = model(x)
            bat_loss = loss(pred, target)
            bat_loss.backward()
            optimizer.step()
            train_loss += bat_loss.item()    #.detach 表示去掉梯度
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1)== data[1].numpy())
        plt_train_loss . append(train_loss/trainloader.dataset.__len__())
        plt_train_acc.append(train_acc/trainloader.dataset.__len__())

        if i % 5 == 0:
            model.eval()
            with torch.no_grad():
                for valdata in valloader:
                    val_x , val_target = valdata[0].to(device), valdata[1].to(device)
                    val_pred = model(val_x)
                    val_bat_loss = loss(val_pred, val_target)
                    val_loss += val_bat_loss.cpu().item()
                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == valdata[1].numpy())
                    val_rel.append(val_pred)

            if val_acc > max_acc:
                torch.save(model, save_)
                max_acc = val_acc
            plt_val_loss.append(val_loss/valloader.dataset.__len__())
            plt_val_acc.append(val_acc/valloader.dataset.__len__())
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | valAcc: %3.6f valLoss: %3.6f  ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1])
                  )
        else:
            plt_val_loss.append(plt_val_loss[-1])
            plt_val_acc.append(plt_val_acc[-1])
        scheduler.step()
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
