from importlib import import_module
import torch
import numpy as np
import argparse
import utils
import utils_2
import time
import train
import bert
import xlNet
from matplotlib import pyplot as plt
import warnings
import sys
import os
import pandas as pd

# class Logger(object):
#     def __init__(self, filename, stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'w')
#         # w:文件存在则覆盖，否则创建新文件
#         # a:文件存在则追加，否则创建新文件
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         return True

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # 方法级
    model_name = "FE_LM"
    path = 'G:\softwareinstall\Pycharm\code\MultiSmell\data'
    type = 'method'

    print(path)
    config = xlNet.config(model_name, type)
    print(config.device)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('数据集加载')

    path = 'E:\刘海洋\图\graph\中文\data/'
    data_list = os.listdir(path)
    print(data_list)

    data = pd.DataFrame([])
    for i in range(len(data_list)):
        subPath = os.path.join(path, data_list[i])
        data_json = pd.read_json(subPath)
        data = pd.concat([data, data_json])

    train_iter, test_iter = utils_2.get_dataload(config, data, config.type)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据总时间：", time_dif)

    start_train = time.time()
    model = xlNet.RIFRE_SEN(config).to(config.device)
    print(model)
    # train
    train_acc_list,train_loss_list,fpr_list,tpr_list = train.train(config, model, train_iter, test_iter)
    t_acc_max,t_acc_min = max(train_acc_list),min(train_acc_list)

    # test
    train.test(config, model, test_iter,type='method')
    end_model = utils.get_time_dif(start_train)

    print('模型测试结束', end_model)
    print("123")
