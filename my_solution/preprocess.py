import os

import pandas as pd


def data_process(base_dir, save_dir):
    train_df = pd.read_json(base_dir + "train.txt", lines=True)
    train_df.columns = ['text_a', 'text_b', 'label']
    train_df.to_csv(save_dir + "train.tsv", index=False, encoding='utf_8_sig', sep='\t')

    valid_df = pd.read_json(base_dir + "valid.txt", lines=True)
    valid_df.columns = ['text_a', 'text_b', 'label']
    valid_df.to_csv(save_dir + "valid.tsv", index=False, encoding='utf_8_sig', sep='\t')

    test_df = pd.read_json(base_dir + "test_with_id.txt", lines=True)
    test_df.columns = ['text_a', 'text_b', 'id']
    test_df.to_csv(save_dir + "test.tsv", index=False, encoding='utf_8_sig', sep='\t')


base = ["短短匹配A类/", "短短匹配B类/", "短长匹配A类/", "短长匹配B类/", "长长匹配A类/", "长长匹配B类/"]
save = ["ssA/", "ssB/", "slA/", "slB/", "llA/", "llB/"]
for i in range(len(save)):
    base_dir = './sohu2021_open_data_clean/' + base[i]
    save_dir = './sohu2021_open_data_clean/' + save[i]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_process(base_dir, save_dir)
