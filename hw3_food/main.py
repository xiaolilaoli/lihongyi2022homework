import argparse
import torch
import torch.nn as nn
from optimizers import get_optimizer, LR_Scheduler
from model_utils.model import myTransformer, initialize_model, SimModel,get_backbone,load_backBone
from model_utils.train import train_val
from model_utils.evaluate import evaluate
from model_utils.data import getDataLoader
from model_utils.sim_train import sim_train_val
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from models.backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
# def get_args_parser():
#     parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms



###############################################
model_name = 'resnet'
num_class = 11
batchSize = 32
learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
epoch = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##########################################

filepath = 'food-11'
save_path = 'model_save/' + model_name+'afsemi_'
SimPrePath = '/home/lhy/hw3/72.model'
# SimPrePath = '/home/lhy/hw3/final.pth'
pre_path = 'model_save/resnetbeforesemi_'
##########################


backbone = get_backbone('resnet18_cifar_variant1')
backbone = load_backBone(backbone,SimPrePath,False)

classfier = nn.Linear(in_features=512, out_features=11, bias=True)



train_loader = getDataLoader(filepath, 'train', batchSize)
val_loader = getDataLoader(filepath, 'val', batchSize)
no_label_Loader = getDataLoader(filepath,'train_unl', batchSize)
# train_loader = None
# val_loader = None
# semi_loader = None
# lr=30*batchSize/256,
optimizer = get_optimizer(
    'sgd', classfier,
    lr=0.03*batchSize/256,
    momentum=0.9,
    weight_decay=0)

# define lr scheduler
scheduler = LR_Scheduler(
    optimizer,
    0, 0*batchSize/256,
    epoch, 30*batchSize/256, 0*batchSize/256,
    len(train_loader),
                             )





trainpara = {
            'backbone': backbone,
             'classfier': classfier,
             'train_loader': train_loader,
             'val_loader': val_loader,
             'no_label_Loader' : no_label_Loader,
             'scheduler': scheduler,
             'optimizer': optimizer,
            'batchSize' :batchSize,
             'loss': loss,
             'epoch': epoch,
             'device': device,
             'save_path': save_path,
             'save_acc': True,
             'max_acc': 0.5,
             'pre_path': None,
             'val_epoch' : 1,
             'acc_thres' : 0.8,
             'conf_thres' : 0.99,
             'do_semi' : False,
             'sim_pre_path' : SimPrePath
             }
if __name__ == '__main__':
    sim_train_val(trainpara)
    # train_val(trainpara)