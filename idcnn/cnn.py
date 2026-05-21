# IDCNN核心代码
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from idcnn.config_idcnn import *

class IDCNN(nn.Module):
    def __init__(self,input_size,filters,kernel_size=3,num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [{'dilation':1},{'dilation':1},{'dilation':3}]         # 定义3个子层膨胀率
        net = nn.Sequential()           # 定义一个空的子结构层net，用来填充block
        norms_1 = nn.ModuleList([nn.LayerNorm(256) for _ in range(len(self.layers))])       # net层正则化
        norms_2 = nn.ModuleList([nn.LayerNorm(256) for _ in range(num_block)])              # idcnn正则化


        # 构建block
        for i in range(len(self.layers)):
            dilation = self.layers[i]['dilation']     # 设定膨胀率
            # 定义卷积层
            single_block = nn.Conv1d(in_channels=filters,out_channels=filters,
                      kernel_size=kernel_size,dilation=dilation,padding=kernel_size//2 + dilation -1)

            # 添加到net层，每一个net层包含conv卷积层，relu激活层，norms正则化层
            net.add_module('layer%d'%i,single_block)
            net.add_module('relu',nn.ReLU())
            net.add_module('norm%d'%i,norms_1[i])

        # 定义全连接层
        self.linear = nn.Linear(input_size,filters)
        # 定义idcnn结构层，用来填充net
        self.idcnn = nn.Sequential(net)

        # 构建4个block，包含net层，relu激活层，norms正则化层
        for i in range(num_block):
            self.idcnn.add_module('block%i'%i,net)
            self.idcnn.add_module('relu',nn.ReLU())
            self.idcnn.add_module('norm%i'%i,norms_2[i])


    # 前向传播
    def forward(self,embeddings,length):
        embeddings = self.linear(embeddings)        # 对词嵌入维度映射到卷积层维度上
        embeddings = embeddings.permute(0,2,1)      # 维度对调，适配api
        output = self.idcnn(embeddings).permute(0,2,1)     # 进行特征抽取后再调换维度
        return output



# 使用经典transformer方法实现layernorm正则化策略
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        # 初始化全1全0张量
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 计算均值和方差，注意是横向dim，跟batch norm做出区分
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 套入公式
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
