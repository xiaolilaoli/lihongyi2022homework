import numpy as np
import torch
from torch.utils.data import  DataLoader
import time
import matplotlib.pyplot as plt
from torch import optim
import csv

def evaluate(model_path, testset, rel_path ,device):
    model = torch.load(model_path).to(device)
    testloader = DataLoader(testset,batch_size=1,shuffle=False)
    plt_train_loss = []
    plt_val_loss = []
    val_rel = []
    model.eval()
    with torch.no_grad():
        for data in testloader:
            x  = data.to(device)
            pred = model(x)
            val_rel.append(pred.item())
    print(val_rel)
    with open(rel_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id','tested_positive'])
        for i in range(len(testset)):
            csv_writer.writerow([str(i),str(val_rel[i])])