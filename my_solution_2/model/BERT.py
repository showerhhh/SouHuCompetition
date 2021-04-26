import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

import config as cnf


class BERT_MODULE(nn.Module):
    def __init__(self):
        super(BERT_MODULE, self).__init__()
        self.bert = BertModel.from_pretrained(cnf.bert_path)
        # self.model_config = BertConfig.from_pretrained(cnf.bert_path)
        # self.model_config.output_hidden_states = True  # 设置返回所有隐层输出
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, tokens):
        # last_hidden层的输出，shape为[batch, seq_len, hidden_size]
        # pooler层的输出，shape为[batch, hidden_size]
        last_hidden = self.bert(**tokens).last_hidden_state
        pooler = self.bert(**tokens).pooler_output
        return pooler


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.extractor = BERT_MODULE()
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
