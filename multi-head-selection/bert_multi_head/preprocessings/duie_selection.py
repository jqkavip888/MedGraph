# 导入工具包
import os
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from functools import cached_property


# 构造数据预处理的类
class DuIE_selection_preprocessing(object):
    def __init__(self, config):
        self.config = config
        # 原始数据存放的路径文件夹
        self.raw_data_root = config['raw_data_root']
        # 预处理后的数据存放的路径文件夹
        self.data_root = config['data_root']
        # 关系元数据定义文件
        self.schema_path = os.path.join(self.raw_data_root, 'all_50_schemas')

        # 鲁棒性检查元数据文件
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError('schema file not found, please check your downloaded data!')
        
        # 鲁棒性检查数据路径
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        # 关系定义字典文件
        self.relation_vocab_path = os.path.join(self.data_root, config['relation_vocab'])

    @cached_property
    def relation_vocab(self):
        # 关系数据定义字典如果存在, 则不处理; 如果不存在, 则生成字典
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r'))

    # 生成NER任务所需的BIO映射字典
    def gen_bio_vocab(self):
        result = {'B': 0, 'I': 1, 'O': 2}
        json.dump(result, open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    # 生成RE任务所需的关系映射字典
    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0

        for line in open(self.schema_path, 'r'):
            # 读取关系定义元数据文件, 并累加1构造映射字典
            relation = json.loads(line)['predicate']
            if relation not in relation_vocab:
                relation_vocab[relation] = i
                i += 1
        
        # 最后将没有定义的关系统一设置为'N', 并放置于最后一个编号位置
        relation_vocab['N'] = i
        # 将生成好的关系字典做永久化保存
        json.dump(relation_vocab, open(self.relation_vocab_path, 'w'), ensure_ascii=False)

    # 生成单词映射字典
    def gen_vocab(self, min_freq):
        # 字典的构造采用训练集数据作为语料
        source = os.path.join(self.raw_data_root, self.config['train'])
        target = os.path.join(self.data_root, 'word_vocab.json')

        counter = Counter()
        with open(source, 'r') as s:
            for line in s:
                line = line.strip("\n")
                if not line:
                    return None

                # 每一行数据单独处理, 采用"字映射"的模式
                instance = json.loads(line)
                text = list(instance['text'])
                counter.update(text)
        
        # 将特殊字符<pad>放置于字典0号编码位置
        result = {'<pad>': 0}
        i = 1
        for k, v in counter.items():
            # 构造字典过程中, 依然采取过滤掉低频词的经典模式
            if v > min_freq:
                result[k] = i
                i += 1

        # 最后一个编码位置给'oov'单词, 以处理UNK的情况
        result['oov'] = i
        # 将生成好的字典做永久化保存
        json.dump(result, open(target, 'w'), ensure_ascii=False)

    # 读取一行训练数据并做格式化处理
    def _read_line(self, line):
        line = line.strip('\n')
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text']

        # 初始化BER任务和RE任务的数据变量
        bio = None
        selection = None

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            # 做合法数据检测
            if not self._check_valid(text, spo_list):
                return None

            # 将SPO列表数据依次提取为字典格式
            spo_list = [{'predicate': spo['predicate'],
                         'object': spo['object'],
                         'subject': spo['subject']
                        } for spo in spo_list]

            # 按照SPO列表数据做NER实体和RE三元组的数据提取
            entities = self.spo_to_entities(text, spo_list)
            relations = self.spo_to_relations(text, spo_list)

            # 再次将NER实体做BIO模式的映射
            bio = self.spo_to_bio(text, entities)
            # 再次将RE三元组做selection模式的映射
            selection = self.spo_to_selection(text, spo_list)

        result = {'text': text,
                  'spo_list': spo_list,
                  'bio': bio,
                  'selection': selection}

        return json.dumps(result, ensure_ascii=False)

    # 生成数据
    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write('\n')

    # 生成训练集数据和验证集数据
    def gen_all_data(self):
        self._gen_one_data(self.config['train'])
        self._gen_one_data(self.config['dev'])

    # 检查数据合法性的函数
    def _check_valid(self, text, spo_list):
        # SPO列表为空, 非法
        if spo_list == []:
            return False

        # 文本长度超过最大限定长度, 非法
        if len(text) > self.config['max_text_len']:
            return False

        # 任意的主体, 或客体, 不在文本text中, 非法
        for t in spo_list:
            if t['object'] not in text or t['subject'] not in text:
                return False

        # 除了上述3种情况, 数据都合法
        return True

    # 从SPO列表中提取实体的集合, 并返回列表格式
    def spo_to_entities(self, text, spo_list):
        entities = set(t['object'] for t in spo_list) | set(t['subject'] for t in spo_list)
        return list(entities)

    # 从SPO列表中宏提取关系的集合, 并返回列表格式
    def spo_to_relations(self, text, spo_list):
        return [t['predicate'] for t in spo_list]

    # 通过SPO列表构造数字化的selection列表
    def spo_to_selection(self, text, spo_list):

        selection = []
        # 遍历SPO列表
        for triplet in spo_list:
            # 读取主体, 客体
            object = triplet['object']
            subject = triplet['subject']

            # 客体最后一个字符的位置index作为object_pos
            object_pos = text.find(object) + len(object) - 1
            # 关系映射字典中的编码直接作为relation_pos
            relation_pos = self.relation_vocab[triplet['predicate']]
            # 主体最后一个字符的位置index作为subject_pos
            subject_pos = text.find(subject) + len(subject) - 1

            # 向结果列表中添加三元组
            selection.append({'subject': subject_pos,
                              'predicate': relation_pos,
                              'object': object_pos
                             })

        return selection

    # 通过SPO列表构造数字化的bio列表
    def spo_to_bio(self, text, entities):
        # 初始化BIO列表, 全部都是'O'
        bio = ['O'] * len(text)
        # 遍历实体列表
        for e in entities:
            # 查找起始位置begin, 终止位置end
            begin = text.find(e)
            end = begin + len(e) - 1

            # 确认实体结束位置没有超过文本长度
            assert end <= len(text)

            # 起始位置标记'B'
            bio[begin] = 'B'
            # 从第二个字符到最后一个字符位置标记'I'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        # 返回标记好的bio列表
        return bio

