import numpy as np
import torch
from torch.utils.data import  DataLoader

import csv

def evaluate(model_path, testloader, rel_path ,device):
    model = torch.load(model_path).to(device)
    # model = model_path
    # testloader = DataLoader(testset,batch_size=1,shuffle=False)  #放入loader 其实可能没必要 loader作用就是把数据形成批次而已
    test_rel = []
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(testloader):
            if i %10 == 0:
                print(i)
            x = data.to(device)
            pred = model(x)
            idx = np.argmax(pred.cpu().data.numpy(),axis=1)
            for each in list(idx):
                test_rel.append(each)
    print(test_rel)
    with open(rel_path, 'w',newline='') as f:
        csv_writer = csv.writer(f)        #百度的csv写法
        csv_writer.writerow(['id','Class'])
        for i in range(len(test_rel)):
            csv_writer.writerow([str(i),str(test_rel[i])])