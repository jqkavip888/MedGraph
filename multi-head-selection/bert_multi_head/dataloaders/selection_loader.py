import os
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from typing import Dict, List, Tuple, Set, Optional
from transformers import BertTokenizer

# Dataset是一个抽象类, 自定义的Dataset需要继承它并且实现两个成员方法:
# __getitem__() 第一个最为重要, 即每次怎么读数据
# __len__() 返回整个数据集的长度

PAD, CLS, SEP= '[PAD]', '[CLS]', '[SEP]'

class Selection_Dataset(Dataset):
    def __init__(self, config, dataset):
        self.config = config
        self.data_root = config['data_root']
        self.bert_model = config['bert_model']
        self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []

        # bert分词器
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip('\n')
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        if self.config['cell_name'] == 'bert':
            # 1.数据载入部分
            # 初始化bert分词器, 对输入句子预处理, 加上特殊标记符[cls],[pad],[seq], 分词
            text, bio, selection = self.pad_bert(text, bio, selection)
            tokens_id = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2id(text)
        bio_id = self.bio2id(bio)
        # self.relation_vocab 作为返回，以便后面批处理的时候将selection变成table
        return tokens_id, bio_id, selection, len(text), spo, text, bio, self.relation_vocab

    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text, bio, selection):
        # for [CLS] and [SEP]
        text = ['[CLS]'] + list(text) + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] +
                      1, 'predicate': triplet['predicate']} for triplet in selection]
        
        assert len(text) <= self.config['max_text_len']
        
        text = text + ['[PAD]'] * (self.config['max_text_len'] - len(text))
        bio = bio + ['O'] * (self.config['max_text_len'] - len(bio))
        
        return text, bio, selection

    # 将词转换为id表示
    def text2id(self, text):
        oov = self.word_vocab['oov']
        text_id_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        return torch.tensor(text_id_list)

    def bio2id(self, bio):
        bio_id_list = list(map(lambda x: self.bio_vocab[x], bio))
        return torch.tensor(bio_id_list)


class Batch_reader(object):
    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        transposed_data = list(zip(*data))
        self.length = transposed_data[3]
        # tokens_id, bio_id, selection_id, spo, text, bio

        # word字典中pad的id是0，pad_sequence默认padding_value是0，这里可以不指定了
        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        self.bio_id = pad_sequence(transposed_data[1], batch_first=True)

        batch_max_text_len = self.tokens_id.size()[1]
        relation_vocab = transposed_data[7][0]
        self.selection_id = self.selection2table(batch_max_text_len, transposed_data[2], relation_vocab)

        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]
        self.bio = transposed_data[6]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        return self

    def selection2table(self, batch_max_text_len, selection, relation_vocab):
        # s p o
        batch_size = len(selection)
        result = torch.zeros(batch_size, batch_max_text_len, len(relation_vocab), batch_max_text_len)
        
        NA = relation_vocab['N']
        for b in range(batch_size):
            result[b, :, NA, :] = 1
            for triplet in selection[b]:
                object = triplet['object']
                subject = triplet['subject']
                predicate = triplet['predicate']

                result[b, subject, predicate, object] = 1
                result[b, subject, NA, object] = 0

        return result


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)

