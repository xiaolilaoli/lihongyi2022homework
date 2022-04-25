from torch.utils.data import Dataset,DataLoader
from contra_utils.model import initialize_model
from contra_utils.utils import get_dataset, get_loss, cosine_lr
from contra_utils.train import train_val
from sklearn.model_selection import train_test_split
import torch

###############################################
model_name = 'resnet'
learning_rate = 1e-4
epochs = 1000
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
batch_size = 256
path = 'food-11'
save_path = 'contra_utils/con_model'
##########################################



train_set, val_set = get_dataset(path)
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=True, drop_last=True)
model, _ = initialize_model(model_name, batch_size, False, False)
total_steps = train_loader.batch_size * epochs
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0001)
scheduler = cosine_lr(optimizer, learning_rate, 5, total_steps)



trainpara = {
             'model': model,
             'train_loader': train_loader,
             'val_loader': val_loader,
             'scheduler': scheduler,
             'optimizer': optimizer,
             'loss': get_loss,
             'epochs': epochs,
             'device': device,
             'save_path': save_path,
             'save_acc': True,
             'max_acc': 0.5,
             'pre_path': None,
             'val_epoch' : 1,
             }

train_val(trainpara)