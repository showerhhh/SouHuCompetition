# import logging
#
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
# logging.basicConfig(filename='record.log',
#                     filemode='a',
#                     level=logging.INFO,
#                     format=LOG_FORMAT,
#                     datefmt=DATE_FORMAT)
# logger = logging.getLogger()

run_type = 'ssA'  # 当前处理类型

lr = 5e-5
batch_size = 16
num_epochs = 80

max_seq_len = 256  # 一句话的长度

data_path = './sohu2021_open_data_clean/'

result_path = './results/'
checkpoint_path = './checkpoint/'

bert_path = './bert-base-chinese/'
ernie_path = './ernie-1.0/'
