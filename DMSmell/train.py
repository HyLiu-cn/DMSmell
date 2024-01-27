import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import utils
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss,roc_curve,precision_recall_curve,jaccard_score,roc_auc_score
from transformers.optimization import *
import matplotlib.pyplot as plt
import sys

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
        # w:文件存在则覆盖，否则创建新文件
        # a:文件存在则追加，否则创建新文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        return True

# def train(config, model, train_iter, vaild_iter, test_iter):
def train(config, model, train_iter, test_iter):
    start_time = time.time()
    model.train()
    # 获取model 参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减参数
    no_decay = ['bias', 'LayerNorm', 'LayerNorm.weight']
    optimizer_growped_paramters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(params=optimizer_growped_paramters,
                           lr=config.learning_rate)
    flag = False  # 记录是否很久没有提升效果

    model.train()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    train_acc_list = np.array([], dtype=int)
    train_loss_list = np.array([], dtype=int)
    fpr_list = np.array([],dtype=int)
    tpr_list = np.array([],dtype=int)
    for epoch in range(config.num_epochs):
        print('Epoch{}/{}'.format(epoch + 1, config.num_epochs),"test_data:{}".format(len(train_iter.dataset.data)))
        epoch_train_acc_list_1 = np.array([],dtype=int)
        epoch_train_acc_list_2 = np.array([],dtype=int)
        epoch_train_loss_list = np.array([],dtype=int)
        epoch_fpr_list = np.array([],dtype=int)
        epoch_tpr_list = np.array([],dtype=int)

        for i, (x1, mask, x2, labels) in enumerate(train_iter):
            x1 = x1.to(config.device)
            mask = mask.to(config.device)
            x2 = x2.to(config.device)
            labels = labels.to(config.device)

            outputs = model(x1, mask, x2, labels)
            model.zero_grad()
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            loss = loss.detach().cpu().numpy()
            epoch_train_loss_list = np.append(epoch_train_loss_list, loss)
            optimizer.step()
            predict = (outputs >= 0.5).long()  # 将输出阈值设置为0.5 [0.49,0.58]
            labels = labels.data.cpu().numpy()
            predict = predict.cpu().numpy()

            if i != 0:
                j = len(predict_all)
                labels_all = np.insert(labels_all,j,labels,axis=0)
                predict_all = np.insert(predict_all,j,predict,axis=0)
            else:
                j = len(predict)
                labels_all = labels
                predict_all = predict
            acc1 = metrics.jaccard_score(labels_all[:, ], predict_all[:, ],average='samples')
            acc2 = metrics.jaccard_score(labels_all[:, ], predict_all[:, ],average='macro')
            epoch_train_acc_list_1 = np.append(epoch_train_acc_list_1, acc1)
            epoch_train_acc_list_2 = np.append(epoch_train_acc_list_2, acc2)

        print("accuracy samples:{} macro:{} \t loss:{} \t ".format(np.mean(epoch_train_acc_list_1),np.mean(epoch_train_acc_list_2),loss.item()))
        # 每个epoch中平均值
        train_acc_list = np.append(train_acc_list,np.mean(epoch_train_acc_list_1))
        train_loss_list = np.append(train_loss_list,np.mean(epoch_train_loss_list))
        fpr_list = np.append(fpr_list,np.mean(epoch_fpr_list))
        tpr_list = np.append(tpr_list,np.mean(epoch_tpr_list))
        if flag:
            break

    return train_acc_list,train_loss_list,fpr_list,tpr_list


def evaluate(config, model, vaild_iter,test=False,type='method'):
    model.eval()
    dev_loss_all = np.array([],dtype=int)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    dev_loss = np.array([],dtype=int)
    with torch.no_grad():
        for i, (x1, mask, x2, labels) in enumerate(vaild_iter):
            x1 = x1.to(config.device)
            mask = mask.to(config.device)
            x2 = x2.to(config.device)
            labels = labels.to(config.device)

            outputs = model(x1, mask, x2, labels)
            predict = (outputs >= 0.5).long()
            epoch_dev_loss = F.binary_cross_entropy(outputs, labels)
            epoch_dev_loss = epoch_dev_loss.detach().cpu().numpy()
            labels = labels.data.cpu().numpy()
            predict = predict.cpu().numpy()

            if i != 0:
                j = len(predict_all)
                labels_all = np.insert(labels_all,j,labels,axis=0)
                predict_all = np.insert(predict_all,j,predict,axis=0)
            else:
                labels_all = labels
                predict_all = predict

            dev_loss_all = np.append(dev_loss_all,epoch_dev_loss)

    if type == 'method':
        # label_name = ['FE','LPL','LM']
        label_name = ['FE','LM']
        # sw = [1,1,1]
    else:
        label_name = ['GC','CC','MC']
        # sw = [1,1,1]
    if test:
        hammingloss,accuracy_1,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,macroAuc,microAuc = metric(labels_all,predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=label_name, digits=4)
        multilabel_confusion = metrics.multilabel_confusion_matrix(labels_all,predict_all)
        return hammingloss,accuracy_1,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,macroAuc,microAuc,report,multilabel_confusion

    acc1 = metrics.accuracy_score(labels_all[:, ], predict_all[:, ])
    dev_loss = np.append(dev_loss,np.mean(dev_loss_all))

    report = metrics.classification_report(labels_all, predict_all)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    hamming_loss = metrics.hamming_loss(labels_all, predict_all)
    precision = precision_score(labels_all,predict_all)
    recall = recall_score(labels_all,predict_all)
    f1 = f1_score(labels_all,predict_all)
    return acc1, dev_loss, report, confusion, hamming_loss,precision,recall,f1

def test(config, model, test_iter,type):
    # model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # ,tets=True ,test_report,test_confusion
    hammingloss,accuracy_1,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,macroAuc,microAuc,report,multilabel_confusion = evaluate(config, model, test_iter, True,type)
    # msg='Test loss:{0>3.2}, Test Acc:{1>3.2%}' ,test_report,test_confusion
    print("#" * 20, '测试结果:{}'.format(len(test_iter.dataset.data)), "#" * 20)
    print("test_hamming_loss:", hammingloss)
    print("test_accuracy samples:", accuracy_1)
    print("test_accuracy macro:", accuracy_2)
    print("test_subAccuracy:",subsetAccuracy)
    print("test_MacroPrecision:", macroPrecision)
    print("test_MacroRecall:", macroRecall)
    print("test_MacroF1:", macroF1)
    print("test_MicroPrecision:", microPrecision)
    print("test_MicroRecall:", microRecall)
    print("test_MicroF1:", microF1)
    print("macroAUC:",macroAuc)
    print("microAUC:",microAuc)
    print("test_report:")
    print(report)
    print("multilabel Confusion Maxtrix:")
    print(multilabel_confusion)

def metric(y_true,y_pre,sample_weight=None):
    # 基于实例的指标：汉明损失、准确率、子集准确率
    def Hloss():
        hammingloss = hamming_loss(y_true,y_pre)
        return hammingloss
    def Accuracy_1():
        accuracy_1 = jaccard_score(y_true,y_pre,average='samples') # Jaccard系数值越大，样本相似度越高
        # accuracy = jaccard_score(y_true,y_pre) # Jaccard系数值越大，样本相似度越高
        return accuracy_1
    def Accuracy_2():
        accuracy_2 = jaccard_score(y_true, y_pre, average='macro')  # Jaccard系数值越大，样本相似度越高
        return accuracy_2
    def SubAccuracy():
        subsetAccuracy = accuracy_score(y_true,y_pre)
        return subsetAccuracy

    # 基于标签的指标：宏查准率、宏查全率、宏F1；微查准率、微查全率、微F1,AUC
    def MacPrecision():
        macroPrecision = precision_score(y_true,y_pre,average='macro')
        return macroPrecision
    def MacRecall():
        macroRecall = recall_score(y_true,y_pre,average='macro')
        return macroRecall
    def MacF1():
        macroF1 = f1_score(y_true,y_pre,average='macro')
        return macroF1
    def MicPrecision():
        microPrecision = precision_score(y_true,y_pre,average='micro')
        return microPrecision
    def MicRecall():
        microRecall = recall_score(y_true,y_pre,average='micro')
        return microRecall
    def MicF1():
        microF1 = f1_score(y_true,y_pre,average='micro')
        return microF1

    def macroAUC():
        macroAuc = roc_auc_score(y_true,y_pre,average='macro')
        return macroAuc
    def microAUC():
        microAuc = roc_auc_score(y_true,y_pre,average='micro')
        return microAuc

    return Hloss(),Accuracy_1(),Accuracy_2(),SubAccuracy(),MacPrecision(),MacRecall(),MacF1(),MicPrecision(),MicRecall(),MicF1(),macroAUC(),microAUC()