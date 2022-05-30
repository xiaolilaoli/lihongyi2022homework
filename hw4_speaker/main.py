import argparse
import torch
import torch.nn as nn
from optimizers import get_optimizer, LR_Scheduler
from model_utils.model import myTransformer, Classifier, Classifier2, Classifier3,FocalSoftmax
from model_utils.train import train_val
from model_utils.evaluate import evaluate, evaluate_test, evaluate_big
from model_utils.lr_loss import get_cosine_schedule_with_warmup
from optimizers import LR_Scheduler
from model_utils.data import getDataloader, get_dataloader_test, getInferenceDataloader
import random
import  numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(1)
###############################################

num_class = 600
batchSize = 128
learning_rate = 2e-5
loss = nn.CrossEntropyLoss()
# loss = FocalSoftmax()
epochs = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##########################################

filepath = '/home/dataset/lhy/hw4/Dataset'
save_path = 'model_save/class_2_6layer.pth'

pre_path = save_path
# model = Classifier()
model = Classifier2()
model = torch.nn.parallel.DataParallel(model.to(device))
# model = myTransformer()
# model = torch.load(pre_path)
##########################

# def parse_args():
#     """arguments"""
#     config = {
#         "data_dir": '/home/dataset/lhy/hw4/Dataset',
#         "save_path": "model_save/model.ckpt",
#         "batch_size": 128,
#         "n_workers": 0,
#         "valid_steps": 2000,
#         "warmup_steps": 1000,
#         "save_steps": 10000,
#         "total_steps": 9990000,
#
#
#     }
#
#     return config

train_loader,val_loader = getDataloader(filepath,batchSize, 'train' )
# train_loader,val_loader,_ = get_dataloader_test(filepath,batchSize, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-4)
warmup_steps = 1000
# define lr scheduler
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, len(train_loader)*epochs)



trainpara = {
             'model': model,
             'train_loader': train_loader,
             'val_loader': val_loader,
             'scheduler': scheduler,
             'optimizer': optimizer,
            'batchSize' :batchSize,
             'loss': loss,
             'epochs': epochs,
             'device': device,
             'save_path': save_path,
             'save_acc': True,
             'max_acc': 0.5,
             'pre_path': None,
             'val_epoch' : 1
             }


# train_val(trainpara)

print('test')

test_loader = getDataloader(filepath,batchSize, 'test')

evaluate(save_path, test_loader, 'rel.csv', device)
test_loader_test = getInferenceDataloader('/home/dataset/lhy/hw4/Dataset')
# #
# # for batch in test_loader:
# #     for batch_2 in test_loader_test:
# #         print(batch)
# #         print(batch_2)
#
evaluate_test(save_path, test_loader_test, filepath, 'rel_others.csv', device)