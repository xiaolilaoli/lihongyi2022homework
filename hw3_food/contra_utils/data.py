import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import cv2
from torchvision.transforms import transforms,autoaugment
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt
from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda)
import albumentations as A
from torchvision.datasets import DatasetFolder
HW = 256

test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    autoaugment.AutoAugment(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
class foodDataset(Dataset):
    def __init__(self, path):
        y = None
        self.transform = train_transform
        imgPaths = path +'/training/unlabeled'
        self.x = self._readfile(imgPaths,False)

    def __getitem__(self, index):
        orix = self.x[index]

        x1 = self.transform(orix)
        x2 = self.transform(orix)
        return x1, x2


    def _readfile(self,path, label=True):
        if label:
            x, y = [], []
            for i in tqdm(range(11)):
                label = '/%02d/'%i
                imgDirpath = path+label
                imglist = os.listdir(imgDirpath)
                xi = np.zeros((len(imglist), HW, HW ,3),dtype=np.uint8)
                yi = np.zeros((len(imglist)),dtype=np.uint8)
                for j, each in enumerate(imglist):
                    imgpath = imgDirpath + each
                    img = Image.open(imgpath)
                    img = img.resize((HW, HW))
                    xi[j,...] = img
                    yi[j] = i
                if i == 0:
                    x = xi
                    y = yi
                else:
                    x = np.concatenate((x, xi), axis=0)
                    y = np.concatenate((y, yi), axis=0)
            print('读入有标签数据%d个 '%len(x))
            return x, y
        else:
            imgDirpath = path + '/00/'
            imgList = os.listdir(imgDirpath)
            x = np.zeros((len(imgList), HW, HW ,3),dtype=np.uint8)

            for i, each in enumerate(imgList):
                imgpath = imgDirpath + each
                img = Image.open(imgpath)
                img = img.resize((HW, HW))
                x[i,...] = img

            return x

    def __len__(self):
        return len(self.x)



def getDataLoader(path, batchSize):

    dataset = foodDataset(path)
    loader = DataLoader(dataset,batchSize,shuffle=True)
    return loader


def samplePlot(dataset, isloader=True,isbat=False,ori=None, **imshow_kwargs):
    if isloader:
        dataset = dataset.dataset
    rows = 3
    cols = 3
    num = rows*cols
    # if isbat:
    #     dataset = dataset * 225
    datalen = len(dataset)
    randomNum = []
    while len(randomNum) < num:
        temp = random.randint(0,datalen-1)
        if temp not in randomNum:
            randomNum.append(temp)
    fig, axs = plt.subplots(nrows=rows,ncols=cols,squeeze=False)
    index = 0
    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            if isbat:
                ax.imshow(np.array(dataset[randomNum[index]].cpu().permute(1,2,0)),**imshow_kwargs)
            else:
                ax.imshow(np.array(dataset[randomNum[index]][0].cpu().permute(1,2,0)),**imshow_kwargs)
            index += 1
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()
    plt.tight_layout()
    if ori != None:
        fig2, axs2 = plt.subplots(nrows=rows,ncols=cols,squeeze=False)
        index = 0
        for i in range(rows):
            for j in range(cols):
                ax = axs2[i, j]
                if isbat:
                    ax.imshow(np.array(dataset[randomNum[index]][-1]),**imshow_kwargs)
                else:
                    ax.imshow(np.array(dataset[randomNum[index]][-1]),**imshow_kwargs)
                index += 1
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        plt.tight_layout()





if __name__ == '__main__':
    filepath = '../food-11'
    val_loader = getDataLoader(filepath, 'val', 8)
    for i in range(100):
        samplePlot(val_loader,True,isbat=False,ori=True)
    ##########################

