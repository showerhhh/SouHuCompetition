import pandas as pd

types = ['ssA', 'ssB', 'slA', 'slB', 'llA', 'llB']

df = pd.DataFrame(columns=['id', 'label'])
for type in types:
    path = './results/{}.csv'.format(type)
    new = pd.read_csv(path, sep=',')
    df = df.append(new, ignore_index=True)
df.to_csv('./results/result.csv', header=True, sep=',', index=False)
