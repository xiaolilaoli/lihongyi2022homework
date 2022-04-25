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
HW = 224
imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(HW),
    transforms.RandomHorizontalFlip(),
    autoaugment.AutoAugment(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


sim_train_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(HW, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_norm)
])

sim_test_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(HW*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
    transforms.CenterCrop(HW),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_norm)
])




class foodDataset(Dataset):
    def __init__(self, path, mode):
        y = None
        self.transform = None
        self.mode = mode
        pathDict = {'train':'training/labeled','train_unl':'training/unlabeled', 'val':'validation', 'test':'testing'}
        imgPaths = path +'/'+ pathDict[mode]

        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     # autoaugment.AutoAugment(),
        #     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        # ])

        if mode == 'test':
            x = self._readfile(imgPaths,False)
            self.transform = sim_test_trans
        elif mode == 'train':
            x, y =self._readfile(imgPaths,True)
            self.transform = sim_train_trans
        elif mode == 'val':
            x, y =self._readfile(imgPaths,True)
            # self.transform = test_transform
            self.transform = sim_test_trans
        elif mode == 'train_unl':
            x = self._readfile(imgPaths,False)
            self.transform = sim_test_trans
        if y is not None:
            y = torch.LongTensor(y)
        self.x, self.y = x, y

    def __getitem__(self, index):
        orix = self.x[index]
        if self.transform == None:
            xT = torch.tensor(orix).float()
        else:
            xT = self.transform(orix)
        if self.y is not None:
            y = self.y[index]
            return xT, y, orix
        else:
            return xT, orix



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

class albuDataset(Dataset):
    def __init__(self, path, mode):
        y = None
        self.transform = None
        self.mode = mode
        pathDict = {'train':'training/labeled','train_unl':'training/unlabeled', 'val':'validation', 'test':'testing'}
        imgPaths = path +'/'+ pathDict[mode]

        self.toTensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_transform = A.Compose([
            A.CenterCrop(224,224),
        ])

        train_transform = A.Compose([
            A.RandomResizedCrop(224,224),
            A.HorizontalFlip(),
            A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        if mode == 'test':
            x = self._readfile(imgPaths,False)
            self.transform = test_transform
        elif mode == 'train':
            x, y =self._readfile(imgPaths,True)
            self.transform = train_transform
        elif mode == 'val':
            x, y =self._readfile(imgPaths,True)
            self.transform = test_transform
        elif mode == 'train_unl':
            x = self._readfile(imgPaths,False)
            self.transform = test_transform
        if y is not None:
            y = torch.LongTensor(y)
        self.x, self.y = x, y

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform == None:
            xT = torch.tensor(x).float()
        else:
            xT = self.transform(image= x)
        if self.y is not None:
            y = self.y[index]
            return self.toTensor(xT['image']), y, x
        else:
            return self.toTensor(xT['image']), x



    def _readfile(self,path):
        if self.mode in ['train', 'val']:
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











class noLabDataset(Dataset):
    def __init__(self,train_dataset, dataloader, backBone,classifier, device, thres=0.85):
        super(noLabDataset, self).__init__()
        self.transformers = sim_train_trans
        self.backBone = backBone
        self.classifier = classifier
        self.device = device
        self.thres = thres
        x, y = self._model_pred(dataloader)
        self.x = np.concatenate((np.array(x),train_dataset.x),axis=0)
        self.y = torch.cat(((torch.LongTensor(y),train_dataset.y)),dim=0)

    def _model_pred(self, dataloader):
        backBone = self.backBone
        classifier = self.classifier
        device = self.device
        thres = self.thres
        pred_probs = []
        labels = []
        x = []
        y = []
        with torch.no_grad():
            for data in dataloader:
                imgs = data[0].to(device)
                feature = backBone(imgs)
                pred = classifier(feature)
                soft = torch.nn.Softmax(dim=1)
                pred_p = soft(pred)
                pred_max, preds = pred_p.max(1)
                pred_probs.extend(pred_max.cpu().numpy().tolist())
                labels.extend(preds.cpu().numpy().tolist())
        for index, prob in enumerate(pred_probs):
            if prob > thres:
                x.append(dataloader.dataset[index][1])
                y.append(labels[index])
        return x, y




    def __getitem__(self, index):
        x = self.x[index]
        x= self.transformers(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

def get_semi_loader(train_loader, dataloader, backBone,classifier, device, thres):
    semi_set = noLabDataset(train_loader.dataset, dataloader, backBone, classifier, device, thres)
    dataloader = DataLoader(semi_set, batch_size=dataloader.batch_size,shuffle=True)
    return dataloader

def getDataLoader(path, mode, batchSize, model = None):
    assert mode in ['train', 'train_unl', 'val', 'test']
    dataset = foodDataset(path, mode)
    if mode in ['test','train_unl']:
        shuffle = False
    else:
        shuffle = True
    loader = DataLoader(dataset,batchSize,shuffle=shuffle)
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

