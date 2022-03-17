from model_utils.model import myNet, init_para,myNet2
from model_utils.data import getDataLoader
from model_utils.train import train_val
from model_utils.evaluate import evaluate
from torch import optim
import torch.nn as nn
import torch
import random
import numpy as np
import os
import csv
def seed_everything(seed=1):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True






#################################################################

batch_size = 512
learning_rate = 1e-4
seed_everything(1)

epoch = 1000
# w = 0.00001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##################################################################

dataPath = 'timit_11'
savePath = 'model_save/My'
trainloader, valloader = getDataLoader(dataPath, 'train', batchSize=batch_size)
test_loader = getDataLoader(dataPath, 'test', batchSize=batch_size)


#
# model = myNet(429, 39)
# model = init_para(model)
model = torch.load(savePath)
model = model.cuda()
optimizer = optim.SGD(model.parameters() , lr=learning_rate, weight_decay=0.0001,momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=1e-9, T_0=20)
criterion = nn.CrossEntropyLoss()
#
train_val(model,trainloader,valloader, optimizer=optimizer ,scheduler=scheduler, loss= criterion, epoch=epoch, device=device, save_=savePath)
#

evaluate(savePath, test_loader, 'myrel.csv',device)


predict = []
raw_output = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)
        for y in test_pred.cpu().numpy():
            predict.append(y)
        for output in outputs.cpu().numpy():
            raw_output.append(output)
raw_output = np.array(raw_output)

alllength = 451552
#transition matrix
data_root = 'timit_11/train_label_11.npy'
train_label = np.load(data_root)
print('size of train data:{}'.format(train_label.shape))
trans_table = np.zeros((39,39))
train_label = train_label.astype('int')

for i in range(len(train_label)-1):
    trans_table[train_label[i], train_label[i+1]] += 1

trans_table_norm = trans_table/ np.sum(trans_table,axis=1,keepdims=True)
trans_table_norm += 1e-17
trans_table_norm = np.log(trans_table_norm)

m = nn.Softmax(dim=1)
test_ln_softmax = m(torch.tensor(raw_output))
test_ln_softmax = np.array((test_ln_softmax))
test_ln_softmax = test_ln_softmax + 1e-17
test_ln_softmax = np.log(test_ln_softmax)


tracking = np.zeros((alllength, 39))
last_state = test_ln_softmax[0]
for i in range(1, len(test_ln_softmax)):
    prob = last_state.reshape(39,1) + trans_table_norm + test_ln_softmax[i]
    current_state = np.max(prob, axis=0)
    tracking[i] = np.argmax(prob, axis=0)
    last_state = current_state

pred_ls = [np.argmax(raw_output[-1])]

for i in range(0,alllength-1):
    back = tracking[alllength-i-1][int(pred_ls[-1])]
    pred_ls.append(int(back))

predict = pred_ls[::-1]

with open('hmm.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)        #百度的csv写法
    csv_writer.writerow(['id','Class'])
    for i in range(len(predict)):
        csv_writer.writerow([str(i),str(predict[i])])