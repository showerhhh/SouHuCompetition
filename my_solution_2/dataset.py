import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import config as cnf


class MyDataset(Dataset):
    def __init__(self, type, mode):
        super(MyDataset, self).__init__()
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(cnf.ernie_path)
        self.tokenizer.padding_side = 'right'

        change_type = {'ssA': '短短匹配A类/', 'ssB': '短短匹配B类/',
                       'slA': '短长匹配A类/', 'slB': '短长匹配B类/',
                       'llA': '长长匹配A类/', 'llB': '长长匹配B类/'}
        base_dir = cnf.data_path + change_type[type]

        if mode == 'train':
            self.df = pd.read_json(base_dir + "train.txt", lines=True)
            self.df.columns = ['text_a', 'text_b', 'label']
        elif mode == 'evaluate':
            self.df = pd.read_json(base_dir + "valid.txt", lines=True)
            self.df.columns = ['text_a', 'text_b', 'label']
        elif mode == 'test':
            self.df = pd.read_json(base_dir + "test_with_id.txt", lines=True)
            self.df.columns = ['text_a', 'text_b', 'id']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.loc[index, :]
        item = {}
        q1_tokens = self.tokenizer.tokenize(row['text_a'])  # 检查q1具体的分词情况
        q2_tokens = self.tokenizer.tokenize(row['text_b'])  # 检查q2具体的分词情况
        q1_tokens = self.tokenizer(row['text_a'], max_length=cnf.max_seq_len, padding='max_length', truncation=True,
                                   return_tensors='pt')
        q2_tokens = self.tokenizer(row['text_b'], max_length=cnf.max_seq_len, padding='max_length', truncation=True,
                                   return_tensors='pt')
        q1_tokens = {k: v.squeeze(0).cuda() for k, v in q1_tokens.items()}  # 将第0维去掉
        q2_tokens = {k: v.squeeze(0).cuda() for k, v in q2_tokens.items()}  # 将第0维去掉
        item['q1'] = q1_tokens
        item['q2'] = q2_tokens
        if self.mode in ['train', 'evaluate']:
            item['label'] = torch.tensor(row['label'], dtype=torch.long).cuda()
        elif self.mode == 'test':
            item['id'] = row['id']
        return item


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    dataset = MyDataset(type='ssA', mode='test')
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size, shuffle=False)

    for index, data in enumerate(dataloader):
        print(data)
