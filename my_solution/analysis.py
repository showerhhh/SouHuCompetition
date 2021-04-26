import pandas as pd


def func(path):
    df = pd.read_csv('./sohu2021_open_data_clean/' + path, sep='\t')
    df.columns = ['text_a', 'text_b', 'label']

    df['text_a_len'] = df['text_a'].map(lambda x: len(x))
    df['text_b_len'] = df['text_b'].map(lambda x: len(x))

    print(df.describe())


paths = ["ssA/train.tsv", "ssB/train.tsv", "slA/train.tsv", "slB/train.tsv", "llA/train.tsv", "llB/train.tsv"]
for path in paths:
    print("-----------------" + path + "-----------------")
    func(path)
