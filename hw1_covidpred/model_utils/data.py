import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import csv
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



def get_feature_importance(feature_data, label_data, k =4,column = None):
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    """
    model = SelectKBest(chi2, k=k)#选择k个最佳特征
    X_new = model.fit_transform(feature_data, label_data)
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_
    p_values = model.pvalues_
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]
    if column:
        k_best_features = [column[i+1] for i in indices[0:k].tolist()]
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]




# with open(r'covid.train.csv', 'r') as f:
#     # train_data = list(csv.reader(f))
#     train_data = f.readlines()
#     column = [line.split('\n') for line in train_data][0]
#     column = column[0].split(',')
#     train_data = [line.split('\n') for line in train_data][1:]
#
#     train_data = [each[0].split(',') for each in train_data]
#     print(len(train_data[0]))
#     train_data = np.array(train_data)
#
# train_x = train_data[:,1:-1]
# train_y = train_data[:,-1]
# train_x,col_indices = get_feature_importance(train_x,train_y,feature_dim,column)
# col_indices = col_indices.tolist()
# print(col_indices)
#




class covidDataset(Dataset):
    def __init__(self, path, mode, feature_dim):
        with open(path,'r') as f:
            csv_data = list(csv.reader(f))
            column = csv_data[0]
            train_x = np.array(csv_data)[1:][:,1:-1]
            train_y = np.array(csv_data)[1:][:,-1]
            _,col_indices = get_feature_importance(train_x,train_y,feature_dim,column)
            col_indices = col_indices.tolist()
            csv_data = np.array(csv_data[1:])[:,1:].astype(float)
            if mode == 'train':
                indices = [i for i in range(len(csv_data)) if i % 5 != 0]
                self.y = torch.LongTensor(csv_data[indices,-1])
            elif mode == 'val':
                indices = [i for i in range(len(csv_data)) if i % 5 != 0]
                # data = torch.tensor(csv_data[indices,col_indices])
                self.y = torch.LongTensor(csv_data[indices,-1])
            else:
                indices = [i for i in range(len(csv_data))]
                # data = torch.tensor(csv_data[indices,col_indices])
            data = torch.tensor(csv_data[indices,:])
            self.data = data[:,col_indices]
            self.mode = mode
            self.data = (self.data - self.data.mean(dim=0,keepdim=True)) /self.data.std(dim=0,keepdim=True)
            assert feature_dim == self.data.shape[1]


            print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
                  .format(mode, len(self.data), feature_dim))

    def __getitem__(self, item):
        if self.mode == 'test':
            return self.data[item].float()
        else :
            return self.data[item].float(), self.y[item]
    def __len__(self):
        return len(self.data)

