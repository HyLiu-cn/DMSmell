import csv
from datetime import timedelta
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import codecs
from HGSmell import xlNet
from sklearn.utils import shuffle as reset
from HGSmell import pre_data_xlnet
from HGSmell import pre_data_class_xlnet
import pre_data_bert,pre_data_class_bert
import json

''' 数据预处理'''
PAD = ['PAD']
def get_dataload(config,data_json,types):
    print('初始数据总数:',len(data_json))
    print(type(data_json))
    train_data, test_data = train_test_split(data_json)
    print('train_data:',len(train_data))
    print('test_data:',len(test_data))

    if config.pretain == 'xlnet':
        traindataset = MyDataset(train_data, types)
        testdataset = MyDataset(test_data, types)
    if config.pretain == 'bert':
        traindataset = MyDataset(train_data, types)
        testdataset = MyDataset(test_data, types)

    train = DataLoader(traindataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
    test = DataLoader(testdataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)

    end_time = time.time()
    print('数据处理结束时间：', end_time)
    return train, test


class MyDataset(Dataset):
    def __init__(self, data, type):
        self.data = data
        self.type = type
        if type == 'method':
            self.padding_size = 20
        else:
            self.padding_size = 20

    def clean_data(self):
        for i in range(len(self.data)):
            x = self.data.iloc[i, :].values
            x2 = x[-2:34]
            try:
                x2 = [float(x) for x in x2]
                x2 = torch.tensor(x2)
            except:
                print(i + 1)
                print(x2)

    def __getitem__(self, index):
        line = self.data.iloc[index].values.tolist()
        x1 = line[:-22]
        x2 = line[-22:-2]
        # label = line[-3:43]
        label = line[-2:42]
        try:
            mask = self.build_data(x1)
        except:
            print(x1)
        x1 = torch.tensor(x1).long()
        mask = torch.tensor(mask).long()

        x2 = [float(x) for x in x2]
        x2 = torch.tensor(x2)
        try:
            label = [float(x) for x in label]
        except:
            print(index)
        label = torch.tensor(label)

        return x1, mask, x2, label

    def __len__(self):
        return len(self.data)

    def collate(self, data):
        x1 = torch.stack([x[0] for x in data], dim=0)
        mask = torch.stack([x[1] for x in data], dim=0)
        x2 = torch.stack([x[2] for x in data], dim=0)
        all_label = []
        i = 0
        for l in data:
            l = l[3]
            if len(l) == 2:
                all_label.append(l)
            else:
                print(i)
                print(l)
            i += 1
        label = torch.stack(all_label, dim=0)
        return x1, mask, x2, label

    def build_data(self, data):
        # tonkens = []
        # data_str = [str(x) for x in data]
        data_str = [x for x in data]
        tonkens = 0
        for content in data_str:
            # if content != '0.0':
            if content != 0:
                tonkens += 1
        pad_size = self.padding_size
        mask = [1] * tonkens
        mask_len = len(mask)
        if pad_size:
            if mask_len <= pad_size:
                mask = mask + [0] * (pad_size - mask_len)
            if mask_len > pad_size:
                mask = mask[:pad_size]

        return mask

def data_pre(data):
    x = []
    for item in data:
        x1, x2, label = item['features']['x1'], item['features']['x2'], item['labels']
        x.append(x1 + x2 + label)

    x_new = torch.tensor(x)
    df = pd.DataFrame(x_new)
    return df

def train_test_split( data, train_size=0.7, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=random_state)

    train = data[:int(len(data) * train_size)].reset_index(drop=True)
    test = data[int(len(data) * train_size):].reset_index(drop=True)
    return train, test

if __name__ == '__main__':
    config = xlNet.config('a','method')
    train, test = get_dataload(config,'G:\softwareinstall\Pycharm\code\MultiSmell\data/FE_LPL_LM.json','method')