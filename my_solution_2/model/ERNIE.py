import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

import config as cnf


class ERNIE_MODULE_1(nn.Module):
    def __init__(self):
        super(ERNIE_MODULE_1, self).__init__()
        self.bert = BertModel.from_pretrained(cnf.ernie_path)
        self.bert_dim = 768
        self.hidden_dim = 256
        self.k = 8
        self.dropout = 0.4
        self.token_att_query = nn.Linear(self.bert_dim, 1)
        self.p_lstm = nn.LSTM(input_size=self.bert_dim, hidden_size=int(self.bert_dim / 2), bidirectional=True)
        self.pool = nn.MaxPool2d(kernel_size=(self.k, 1))
        # self.reset_para()

    def reset_para(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, 1.141)
                nn.init.constant_(layer.bias, 0)

    def forward(self, tokens):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        input_ids = input_ids.view(-1, cnf.max_seq_len)
        attention_mask = attention_mask.view(-1, cnf.max_seq_len)
        embed = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

        token_score = self.token_att_query(embed).view(-1, cnf.max_seq_len)
        token_score = torch.mul(token_score, 1.0 / math.sqrt(float(self.hidden_dim)))
        adder = (1 - attention_mask) * -10000.0
        token_score = token_score + adder
        token_score = F.softmax(token_score, dim=-1).view(-1, cnf.max_seq_len, 1)
        embed = torch.bmm(embed.permute(0, 2, 1), token_score).view(-1, self.bert_dim)
        embed = F.elu(embed)
        embed = F.dropout(embed, p=self.dropout).view(-1, self.k, self.bert_dim)
        embed, _ = self.p_lstm(embed)

        # 合并不同段落
        pool_out = self.pool(embed).view(-1, self.bert_dim)
        return pool_out


class ERNIE_MODULE_2(nn.Module):
    def __init__(self):
        super(ERNIE_MODULE_2, self).__init__()
        self.bert = BertModel.from_pretrained(cnf.ernie_path)
        self.bert_dim = 768
        self.hidden_dim = 256
        self.dropout = 0.4
        self.lstm = nn.LSTM(input_size=self.bert_dim, hidden_size=int(self.bert_dim / 2), bidirectional=True)
        self.token_att_query = nn.Linear(self.bert_dim, 1)
        # self.reset_para()

    def reset_para(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, 1.141)
                nn.init.constant_(layer.bias, 0)

    def forward(self, tokens):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        embed = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        embed, _ = self.lstm(embed)

        token_score = self.token_att_query(embed).view(-1, cnf.max_seq_len)
        token_score = torch.mul(token_score, 1.0 / math.sqrt(float(self.hidden_dim)))
        adder = (1 - attention_mask) * -10000.0
        token_score = token_score + adder
        token_score = F.softmax(token_score, dim=-1).view(-1, cnf.max_seq_len, 1)
        embed = torch.bmm(embed.permute(0, 2, 1), token_score).view(-1, self.bert_dim)
        embed = F.elu(embed)
        embed = F.dropout(embed, p=self.dropout).view(-1, self.bert_dim)

        return embed


class ERNIE(nn.Module):
    def __init__(self):
        super(ERNIE, self).__init__()
        self.extractor = ERNIE_MODULE_2()
        self.dropout = 0.4
        self.feature_dim = 768
        self.hidden_dim_1 = 512
        self.hidden_dim_2 = 32
        self.output_dim = 2
        self.fc1 = nn.Linear(3 * self.feature_dim, self.hidden_dim_1)
        self.fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.output_dim)

    def forward(self, q1, q2):
        y1 = self.extractor(q1)  # (batch, feature_dim)
        y2 = self.extractor(q2)  # (batch, feature_dim)
        y1_y2 = torch.abs(y1 - y2)
        cat = torch.cat([y1, y2, y1_y2], dim=1)  # (batch, 3*feature_dim)
        out = F.dropout(F.leaky_relu(self.fc1(cat)), p=self.dropout)  # (batch, hidden_dim_1)
        out = F.dropout(F.leaky_relu(self.fc2(out)), p=self.dropout)  # (batch, hidden_dim_2)
        out = self.fc3(out)  # (batch, output_dim)
        return out
