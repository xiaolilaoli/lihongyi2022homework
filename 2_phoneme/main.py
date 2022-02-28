import torch
import os
import numpy as np



train_path = 'timit_11/train_11.npy'
trainlabel_path = 'timit_11/train_label_11.npy'
test_path = 'timit_11/test_11.npy'

train_array = np.load(train_path)
print(train_array.shape)