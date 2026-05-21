# 构建idcnn_crf核心类，把idcnn输出和crf输入做融合

import torch
import torch.nn as nn
from torch.autograd import Variable
from idcnn.crf import CRF
from idcnn.cnn import IDCNN


class IDCNN_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, num_filters=64, dropout=0.4, use_cuda=True):
        super(IDCNN_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.num_filters = num_filters
        self.use_cuda = use_cuda

        # 设置词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 实例化idcnn，词嵌入层，卷积核num
        self.idcnn = IDCNN(input_size=embedding_dim, filters=num_filters)
        # 定义dropout
        self.dropout = nn.Dropout(p=dropout)
        # 定义fc层，从crf映射到发射矩阵
        self.linear = nn.Linear(num_filters, tagset_size + 2)
        # 定义crf层=转移矩阵
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)


    # 卷积计算，获取发射矩阵输出张量
    def get_output_score(self, sentence, attention_mask=None):
        # 单句提取维度
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        # 流向embedding层，准备卷积
        embeds = self.embedding(sentence)

        # 进卷积
        output = self.idcnn(embeds, attention_mask)
        output = self.dropout(output)

        # fc层映射+维度变换
        output = self.linear(output)
        feats = output.contiguous().view(batch_size, seq_length, -1)

        return feats


    # 推理阶段前向传播计算
    def forward(self, sentence, masks):
        # 得到发射矩阵的张量
        feats = self.get_output_score(sentence)
        # 解码
        scores, tag_seq = self.crf._viterbi_decode(feats, masks.bool())

        # 分数不需要，直接舍弃
        return tag_seq



    # 训练阶段前向传播计算
    def neg_log_likelihood(self, sentence, mask, tag):
        # 得到发射矩阵的张量
        feats = self.get_output_score(sentence)
        # 计算loss
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tag)

        # 计算average batch size，然后计算average loss
        batch_size = feats.size(0)
        loss_value /= float(batch_size)

        return loss_value

