import os
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from typing import Dict, List, Tuple, Set, Optional


# Dataset是一个抽象类, 自定义的Dataset需要继承它并且实现两个成员方法:
# __getitem__() 第一个最为重要, 即每次怎么读数据
# __len__() 返回整个数据集的长度
class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper
        self.data_root = hyper['data_root']
        self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []
        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
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
        if self.hyper['cell_name'] == 'bert':
            pass
        else:
            tokens_id = self.text2id(text)
        
        bio_id = self.bio2id(bio)

        #self.relation_vocab 作为返回，以便后面批处理的时候将selection变成table
        return tokens_id, bio_id, selection, len(text), spo, text, bio, self.relation_vocab

    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text, bio, selection):
        text = ['[CLS]'] + text + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] + 1, 
                      'predicate': triplet['predicate']} for triplet in selection]
        
        assert len(text) <= self.hyper.max_text_len
        text = text + ['[PAD]'] * (self.hyper.max_text_len - len(text))
        
        return text, bio, selection


    #将词转换为id表示
    def text2id(self, text):
        # text: 如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈
        
        oov = self.word_vocab['oov']
        text_id_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        
        # text_id_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 3, 14, 5, 15, 16, 17, 18, 13, 19, 20, 21, 22, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 21, 33, 7, 34, 35, 36, 37]
        
        return torch.tensor(text_id_list)

    def bio2id(self, bio):
        bio_id_list = list(map(lambda x: self.bio_vocab[x], bio))
        return torch.tensor(bio_id_list)


class Batch_reader(object):
    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        transposed_data = list(zip(*data))
        # 训练代码中的sample.length的取值就来源于这里.
        self.length = transposed_data[3]
        # tokens_id, bio_id, selection_id, spo, text, bio

        # word字典中pad的id是0, pad_sequence默认padding_value是0, 这里可以不指定了
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
        
        # selection: ([{'subject': 11, 'predicate': 14, 'object': 30}, {'subject': 11, 'predicate': 17, 'object': 16}, {'subject': 11, 'predicate': 21, 'object': 34}], [{'subject': 5, 'predicate': 20, 'object': 20}], [{'subject': 1, 'predicate': 26, 'object': 6}, {'subject': 1, 'predicate': 3, 'object': 20}, {'subject': 1, 'predicate': 36, 'object': 55}, {'subject': 1, 'predicate': 17, 'object': 14}], [{'subject': 20, 'predicate': 12, 'object': 38}], ......, [{'subject': 6, 'predicate': 8, 'object': 10}, {'subject': 10, 'predicate': 24, 'object': 6}])

        NA = relation_vocab['N']
        # NA: 49

        for b in range(batch_size):
            result[b, :, NA, :] = 1
            for triplet in selection[b]:
                object = triplet['object']
                subject = triplet['subject']
                predicate = triplet['predicate']

                result[b, subject, predicate, object] = 1
                result[b, subject, NA, object] = 0

        '''
          result: tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            ...,
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [1., 1., 1.,  ..., 1., 1., 1.]],


                           [[0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            ...,
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [0., 0., 0.,  ..., 0., 0., 0.],
                            [1., 1., 1.,  ..., 1., 1., 1.]]]])
        '''

        return result


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)

