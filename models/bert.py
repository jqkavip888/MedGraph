import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer,BertConfig


class Config(object):
    def __init__(self, dataset):
        self.model_name = "bert"
        self.data_path = "/root/knowledge_graph/data"
        self.train_path = self.data_path + "/train.txt"  # 训练集
        self.dev_path = self.data_path + "/dev.txt"  # 验证集
        self.test_path = self.data_path + "/test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path + "/class.txt").readlines()] # 类别名单
        
        self.save_path = "/root/knowledge_graph/models/src"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果
        

        # 模型训练+预测的时候, 放开下一行代码, 在GPU上运行.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 4  # mini-batch大小
        self.pad_size = 40  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "/root/knowledge_graph/multi-head-selection/bert_multi_head/bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/config.json')
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids = input_ids.to('cuda')
        # attention_mask = attention_mask.to('cuda')
        # token_type_ids = token_type_ids.to('cuda')

        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.fc(output.pooler_output)
        
        return out

