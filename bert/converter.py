'''
If you are having trouble multiprocessing inside Notebooks, give this script a shot.

Written by Thilina Rajapakse
https://github.com/ThilinaRajapakse/BERT_binary_text_classification/blob/master/converter.py
'''

import pickle, sys

from tqdm import tqdm, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

from multiprocessing import Pool, cpu_count
from tools import *
import convert_examples_to_features

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "data/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-uncased'

# The name of the task to train.
TASK_NAME = sys.argv[1]

# The output directory where the model predictions and checkpoints will be written.
OUTPUT_DIR = f'outputs/{TASK_NAME}/'

CACHE_DIR = 'cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 128

RANDOM_SEED = 42
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"

output_mode = OUTPUT_MODE

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

processor = BinaryClassificationProcessor()
train_examples = processor.get_train_examples(DATA_DIR)
train_examples_len = len(train_examples)

label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)
print('num_labels: ', num_labels)

label_map = {label: i for i, label in enumerate(label_list)}
train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, output_mode) for example in train_examples]

process_count = 10
if __name__ ==  '__main__':
    print(f'Preparing to convert {train_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        train_features = list(tqdm(p.imap(convert_examples_to_features.convert_example_to_feature, train_examples_for_processing), total=train_examples_len))

with open(DATA_DIR + "train_features.pkl", 'wb') as f:
          pickle.dump(train_features, f)

tokenizer.save_vocabulary(OUTPUT_DIR)

