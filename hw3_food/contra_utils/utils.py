import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import cv2
from torchvision.transforms import transforms,autoaugment
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
HW = 256
import torchvision.models as models

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
        self.transform = train_transform
        imgPaths = path +'/training/unlabeled'
        self.x = self._readfile(imgPaths)

    def __getitem__(self, index):
        orix = self.x[index]
        x1 = self.transform(orix)
        x2 = self.transform(orix)
        return x1, x2

    def _readfile(self,path):
            imgDirpath = path + '/00/'
            imgList = os.listdir(imgDirpath)
            x = np.zeros((len(imgList), HW, HW ,3),dtype=np.uint8)
            # x = np.zeros((600, HW, HW ,3),dtype=np.uint8)
            for i, each in tqdm(enumerate(imgList)):
                imgpath = imgDirpath + each
                img = Image.open(imgpath)
                img = img.resize((HW, HW))
                x[i,...] = img
                # if i == 599:
                #     break
            return x
    def __len__(self):
        return len(self.x)





def get_loss(feature1, feature2, device):
    acc = 0
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
    feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)
    logits1_2 = feature1 @ feature2.t()
    logits2_1 = feature2 @ feature1.t()
    ground_truth = torch.arange(len(logits1_2)).long()
    ground_truth = ground_truth.to(device)
    bat_loss =  (loss1(logits1_2, ground_truth) + loss2(logits2_1, ground_truth)) /2
    acc += np.sum(np.argmax(logits1_2.cpu().data.numpy(), axis=1) == ground_truth.cpu().numpy())
    return bat_loss, acc


class my_dataset(Dataset):
    def __init__(self,x):
        self.x = x

    def __getitem__(self, item):
        return self.x[item][0], self.x[item][1]

    def __len__(self):
        return len(self.x)






def get_dataset(path):
    dataset = foodDataset(path)
    train, val = train_test_split(dataset,test_size=0.2)
    train_set = my_dataset(train)
    val_set = my_dataset(val)
    return train_set, val_set


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length
def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster






def evaluate(model, loss, epoch, args, tb_writer=None, steps=None):


    model.eval()

    dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, texts = batch
            tokens = tokenize(texts)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                tokens = {k: v.cuda(args.gpu, non_blocking=True) for k, v in tokens.items()}

            image_features, text_features, logit_scale = model(images, tokens)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                                 loss_img(logits_per_image, ground_truth)
                                 + loss_txt(logits_per_text, ground_truth)
                         ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = get_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale
        )
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics
