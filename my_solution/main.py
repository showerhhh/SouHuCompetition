# 复制到ai studio上运行
# !pip install paddlehub==1.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

import paddlehub as hub
import pandas as pd
from paddlehub.dataset.base_nlp_dataset import TextMatchingDataset


class Config(object):
    type = None
    max_seq_len = None
    train_and_eval_batch = None
    test_batch = None


def main(type, cnf):
    class SouhuCompetition(TextMatchingDataset):
        def __init__(self, tokenizer=None, max_seq_len=None):
            base_path = './data'
            if type in ['ssA', 'slA', 'llA']:
                train_file = 'data78383/{}_train.tsv'.format(type)
                dev_file = 'data78383/{}_valid.tsv'.format(type)
            elif type in ['ssB', 'slB', 'llB']:
                train_file = 'data78384/{}_train.tsv'.format(type)
                dev_file = 'data78384/{}_valid.tsv'.format(type)
            super(SouhuCompetition, self).__init__(
                is_pair_wise=False,  # 文本匹配类型，是否为pairwise
                base_path=base_path,
                train_file=train_file,  # 相对于base_path的文件路径
                dev_file=dev_file,  # 相对于base_path的文件路径
                train_file_with_header=True,
                dev_file_with_header=True,
                label_list=["0", "1"],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len)

    module = hub.Module(name="ernie")

    # pointwise任务需要: query, title_left (2 slots)
    inputs, outputs, program = module.context(trainable=True, max_seq_len=cnf.max_seq_len, num_slots=2)

    tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path(), tokenize_chinese_chars=True)
    dataset = SouhuCompetition(tokenizer=tokenizer, max_seq_len=cnf.max_seq_len)

    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=0.01,
        warmup_proportion=0.1,
        learning_rate=1e-5)
    config = hub.RunConfig(
        eval_interval=300,
        use_cuda=True,
        num_epoch=10,
        batch_size=cnf.train_and_eval_batch,
        checkpoint_dir='./ckpt_ernie_pointwise_matching_{}'.format(type),
        strategy=strategy)
    # 构建迁移网络，使用ernie的token-level输出
    query = outputs["sequence_output"]
    title = outputs['sequence_output_2']
    # 创建pointwise文本匹配任务
    pointwise_matching_task = hub.PointwiseTextMatchingTask(
        dataset=dataset,
        query_feature=query,
        title_feature=title,
        tokenizer=tokenizer,
        config=config)
    run_states = pointwise_matching_task.finetune_and_eval()

    # # 预测数据样例
    # text_pairs = [
    #     [
    #         "小孩吃了百令胶囊能打预防针吗",  # query
    #         "小孩吃了百令胶囊能不能打预防针",  # title
    #     ],
    #     [
    #         "请问呕血与咯血有什么区别?",  # query
    #         "请问呕血与咯血异同？",  # title
    #     ]
    # ]
    save_df = pd.DataFrame(columns=['id', 'label'])

    def predict(text_pairs):
        results = pointwise_matching_task.predict(
            data=text_pairs,
            max_seq_len=cnf.max_seq_len,
            label_list=dataset.get_labels(),
            return_result=True,
            accelerate_mode=False)
        return results

    if type in ['ssA', 'slA', 'llA']:
        test_file = './data/data78383/{}_test.tsv'.format(type)
    elif type in ['ssB', 'slB', 'llB']:
        test_file = './data/data78384/{}_test.tsv'.format(type)
    test_df = pd.read_csv(test_file, sep='\t')
    test_df.columns = ['text_a', 'text_b', 'id']
    text_pairs = []
    ids = []
    for index, row in test_df.iterrows():
        text_pairs.append([row['text_a'], row['text_b']])
        ids.append(row['id'])
        if len(text_pairs) == cnf.test_batch:
            results = predict(text_pairs)
            for i in range(len(ids)):
                new = pd.DataFrame({'id': ids[i], 'label': results[i]}, index=[0])
                save_df = save_df.append(new, ignore_index=True)
            text_pairs = []
            ids = []
    if len(text_pairs) != 0:
        results = predict(text_pairs)
        for i in range(len(ids)):
            new = pd.DataFrame({'id': ids[i], 'label': results[i]}, index=[0])
            save_df = save_df.append(new, ignore_index=True)

    save_df.to_csv('./results/{}.csv'.format(type), header=True, sep=',', index=False)


if __name__ == '__main__':
    params_dict = {
        # 'ssA': [128, 64, 64],
        # 'ssB': [128, 64, 64],
        # 'slA': [512, 8, 8],
        'slB': [512, 8, 8],
        'llA': [512, 8, 8],
        'llB': [512, 8, 8],
    }
    for type, params_list in params_dict.items():
        cnf = Config()
        cnf.type = type
        cnf.max_seq_len = params_list[0]
        cnf.train_and_eval_batch = params_list[1]
        cnf.test_batch = params_list[2]
        main(type, cnf)
