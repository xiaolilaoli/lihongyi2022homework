import numpy as np
import torch
from torch.utils.data import  DataLoader
from model_utils.model import model_Datapara
import csv
from  tqdm import tqdm
import json
from pathlib import Path
def evaluate(model_path, testloader, rel_path ,device):
    model = torch.load(model_path).to(device)
    model.device_ids = [0]
    results = [["Id", "Category"]]
    id2speaker = testloader.dataset.id2speaker
    model.eval()
    with torch.no_grad():
        for i,(feat_paths, mels) in enumerate(tqdm(testloader)):
                if i > 4000:
                    mels = mels.to(device)
                    outs = model(mels)
                    preds = outs.argmax(1).cpu().numpy()
                    for feat_path, pred in zip(feat_paths, preds):
                        results.append([feat_path, id2speaker[str(pred)]])

    with open(rel_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

def evaluate_test(model_path, testloader, datapath, rel_path ,device):
    mapping_path = Path(datapath) / "mapping.json"
    mapping = json.load(mapping_path.open())
    model = torch.load(model_path).to(device)
    model.device_ids = [0]
    results = [["Id", "Category"]]

    model.eval()
    for feat_paths, mels in tqdm(testloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(rel_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

def evaluate_big(model_path, testloader, rel_path ,device, flag):
    model = torch.load(model_path).to(device)
    model.device_ids = [0]
    if flag == 'fir':
        results = [["Id", "Category"]]
    else:
        results = []
    with open(rel_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for line in csvreader:
            results.append(line)
    id2speaker = testloader.dataset.id2speaker
    model.eval()
    with torch.no_grad():
        for i,(feat_paths, mels) in enumerate(tqdm(testloader)):
            if flag == 'fir':
                if i <= 4200:
                    mels = mels.to(device)
                    outs = model(mels)
                    preds = outs.argmax(1).cpu().numpy()
                    for feat_path, pred in zip(feat_paths, preds):
                        results.append([feat_path, id2speaker[str(pred)]])
            elif flag == 'sec':
                if i > 4200:
                    mels = mels.to(device)
                    outs = model(mels)
                    preds = outs.argmax(1).cpu().numpy()
                    for feat_path, pred in zip(feat_paths, preds):
                        results.append([feat_path, id2speaker[str(pred)]])
    with open(rel_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)