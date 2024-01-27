import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig,BertModel
from transformers import XLNetTokenizer,XLNetModel,RobertaTokenizer,RobertaModel
import transformers
import math
from torch.nn import functional as F

print("transformers 版本:",transformers.__version__)

# 模型合并,输入
class config(object):
    def __init__(self, model_name,type):
        self.model_name = model_name
        self.type = type
        self.save_path = 'E:/刘海洋/图/graph/saved_dict/' + self.model_name + '.ckpt'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvment = 1000
        self.pretain = 'xlnet'
        # 标签数
        self.num_class = 2
        # 迭代次数
        self.num_epochs = 30
        # 批次数目
        self.batch_size = 128
        # 每条信息长度，长截取 段补充
        self.padding_size = 60
        # 学习率
        self.learning_rate = 1e-4
        # bert与训练位置
        self.bert_path = 'E:\Pre-training\Bert\BERT12'
        self.xlnet_path = 'DE:\Pre-training\XLNet'
        # bert分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # gru 隐藏层
        self.gru_hidden = 256
        # rnn 隐藏层
        self.rnn_hidden = 256
        # bert 隐层数量
        self.hidden_size = 128
        # 卷积核数量
        self.num_filter = 256
        self.class_nums = 2
        self.pool_type = max
        # 隐藏层数量
        self.num_layer = 2
        self.gat_layers = 3
        # drptout
        self.dropout = 0.5
        self.linear = 128

class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()
        self.lstm_1 = torch.nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        self.dropout = torch.nn.Dropout(0.4)
        self.linear_1 = torch.nn.Linear(256,128)
        self.act = torch.nn.Tanh()
        self.flatten = torch.nn.Flatten()

    def forward(self,inputs):
        inputs = inputs.reshape(inputs.shape[0],1,-1)
        out,_ = self.lstm_1(inputs)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.linear_1(out)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            torch.nn.Conv1d(in_channels=20, out_channels=128, kernel_size=3,padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=1),
        )
    def forward(self,inputs):
        inputs = inputs.view(inputs.shape[0],-1,1)
        out = self.cnn(inputs)
        return out

# 阈值定义0.5，计算比较结果，随机初始化权重
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.xlnet = RobertaModel.from_pretrained('E:\Pre-training\CodeBERT')
        for param in self.xlnet.parameters():
            param.requires_grad = False
        self.bi_lstm = Bi_LSTM()
        self.cnn = CNN()

        self.dropout = torch.nn.Dropout(0.4)
        self.act = torch.nn.ReLU()
        self.q = nn.Linear(256, 256)
        self.k = nn.Linear(256, 256)
        self.v = nn.Linear(256, 256)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256,2)


    def forward(self, x1, mask, x2, label):
        # Check size of tensors
        encoder_out = self.xlnet(x1, attention_mask=mask)[0]  # [128, 50, 768]
        encoder_out, _ = torch.max(encoder_out, dim=1)  # [128, 768]
        out_1 = self.bi_lstm(encoder_out)

        x2 = x2.view(x2.shape[0],-1,1)
        out_2 = self.cnn(x2)
        out_2 = self.flatten(out_2)
        x = torch.cat([out_1,out_2],dim=1)
        x = x.unsqueeze(1)
        query, key, value = self.q(x), self.k(x), self.v(x)
        attention = F.softmax(query @ key.transpose(1, 2) / math.sqrt(query.size(2)), dim=-1) @ value
        out = self.flatten(attention)
        out = self.fc(out)
        logits = torch.sigmoid(out)
        return logits