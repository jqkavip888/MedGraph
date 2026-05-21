# 导入工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import copy
from functools import partial
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence
from torchcrf import CRF




# multi-head-selection模型的实现类代码
class MultiHeadSelection(nn.Module):
    def __init__(self, config):
        super(MultiHeadSelection, self).__init__()
        self.config = config
        self.data_root = config['data_root']
        self.gpu = config['gpu']
        self.bert_model = config['bert_model']

        # 读取原始文件数据
        self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}

        # Word, 词汇嵌入张量
        self.word_embeddings = nn.Embedding(num_embeddings=len(self.word_vocab), embedding_dim=config['emb_size'])

        # Relation, 关系嵌入张量
        self.relation_emb = nn.Embedding(num_embeddings=len(self.relation_vocab), embedding_dim=config['rel_emb_size'])
        # NER, 命名实体嵌入张量
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab), embedding_dim=config['bio_emb_size'])

        # 编码器Encoder的不同定义, 可以采用GRU, LSTM, BERT
        if config['cell_name'] == 'gru':
            self.encoder = nn.GRU(config['emb_size'],
                                  config['hidden_size'],
                                  bidirectional=True,
                                  batch_first=True)
        elif config['cell_name'] == 'lstm':
            self.encoder = nn.LSTM(config['emb_size'],
                                   config['hidden_size'],
                                   bidirectional=True,
                                   batch_first=True)
        elif config['cell_name'] == 'bert':
            # 当采用BERT作为编码器时, 直接引入预训练模型即可
            self.encoder = BertModel.from_pretrained(self.bert_model)
            # 此处即使不执行for循环, 默认的所有encoder参数都参与反向传播和参数更新
            # 为了清晰起见此处明确设置为True 
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError('cell name should be gru/lstm/bert!')

        # 设置激活函数为relu或者tanh
        if config['activation'].lower() == 'relu':
            self.activation = nn.ReLU()
        elif config['activation'].lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        # 最上层设置一层CRF, 针对于NER任务优化
        self.tagger = CRF(len(self.bio_vocab), batch_first=True)

        # 按照multi-head-selection的计算公式, 共有3个矩阵, 定义如下:
        self.selection_u = nn.Linear(config['hidden_size'] + config['bio_emb_size'], config['rel_emb_size'])
        self.selection_v = nn.Linear(config['hidden_size'] + config['bio_emb_size'], config['rel_emb_size'])
        self.selection_uv = nn.Linear(2 * config['rel_emb_size'], config['rel_emb_size'])
        
        # 定义NER任务的发射矩阵
        self.emission = nn.Linear(config['hidden_size'], len(self.bio_vocab))

    # 推理阶段的函数, 用于获取多头提取出来的三元组
    def inference(self, mask, text_list, decoded_tag, selection_logits):
        # 将Mask矩阵设置为[batch_size, seq_len, relation_size, seq_len]的形状
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2)
        selection_mask = selection_mask.expand(-1, -1, len(self.relation_vocab), -1)

        # print('selection_mask:', selection_mask)
        # selection_mask: tensor([[[[False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False],
        #   ...,
        #   [False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False]],

        #  [[False,  True,  True,  ..., False, False, False],
        #   [False,  True,  True,  ..., False, False, False],
        #   [False,  True,  True,  ..., False, False, False],
        
        # selection_mask.shape: torch.Size([4, 200, 50, 200])
        # selection_logits.shape: torch.Size([4, 200, 50, 200])

        # 对多头模型的输出张量执行sigmoid二分类后, 并进行掩码. 最后的0, 1标签通过和阈值做对比得出
        selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > self.config['threshold']

        # 获取多头模型的结果三元组
        selection_triplets = self.selection_decode(text_list, decoded_tag, selection_tags)
        
        return selection_triplets

    # 计算BCELoss值
    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        # 将Mask矩阵设置为[batch_size, seq_len, relation_size, seq_len]的形状
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2)
        selection_mask = selection_mask.expand(-1, -1, len(self.relation_vocab), -1)

        # 直接利用多标签分类损失函数, 计算得出损失值
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
        
        # 对损失张量进行mask掩码计算
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        # 对损失值做归一化处理
        selection_loss /= mask.sum()
        
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return 'L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:'.format(
                output['loss'].item(), output['crf_loss'].item(),
                output['selection_loss'].item(), epoch, epoch_num)

    def forward(self, sample, is_train):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 对应于文本sentence ids
        tokens = sample.tokens_id.to(device)
        # 对应于关系三元组的ground truth
        selection_gold = sample.selection_id.to(device)
        # 对应于NER的ground truth
        bio_gold = sample.bio_id.to(device)

        # 原始数据中的text, spo_list, bio_list
        text_list = sample.text
        spo_gold = sample.spo_gold
        bio_text = sample.bio

        if self.config['cell_name'] in ('gru', 'lstm'):
            # mask: [batch_size, seq_len]
            mask = tokens != self.word_vocab['<pad>']
            bio_mask = mask
        elif self.config['cell_name'] in ('bert'):
            # 构造BERT的padding mask, 此处也要将[CSL], [SEP]也给mask掉
            not_pad = tokens != 0
            not_cls = tokens != 101
            not_sep = tokens != 102
            # 做'与'关系的处理, 结果为mask张量
            mask = not_pad & not_cls & not_sep
            # 针对于NER任务, 进行CRF计算时第一个字符[CLS]不能被mask掉
            bio_mask = not_pad & not_sep
        else:
            raise ValueError('unexpected encoder name!')

        # 如果采用LSTM, GRU作为处理单元
        if self.config['cell_name'] in ('lstm', 'gru'):
            # 首先进行词嵌入的操作
            embedded = self.word_embeddings(tokens)
            
            # 将同一个batch中不同长度的语句进行pack_padded操作, 提升Pytorch的运算效率
            pack_padded_embedded = pack_padded_sequence(embedded, sample.length, batch_first=True)
            
            # 送入编码器, 得到输出张量和隐藏层张量
            o, h = self.encoder(pack_padded_embedded)
            
            # 再对张量结果进行pad_packed还原操作
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
            
            # 相当于双向LSTM的两个张量, 取平均值的结果作为最后的o张量
            o = (lambda a: sum(a) / 2)(torch.split(o, self.config['hidden_size'], dim=2))
        # 如果采用BERT作为处理单元
        elif self.config['cell_name'] == 'bert':
            # 2.模型部分
            # 编码器部分改为bert, mask的处理, last hidden of BERT
            o = self.encoder(tokens, attention_mask=mask)[0]
        else:
            raise ValueError('unexpected encoder name!')
        
        # 编码器的输出张量, 直接送入发射矩阵中, 得到发射张量emi
        emi = self.emission(o)
        output = {}
        crf_loss = 0

        # 训练阶段
        if is_train:
            # 训练阶段, 直接将发射张良emi送入CRF中, 和NER任务的标签计算损失
            crf_loss = -self.tagger(emi, bio_gold, mask=bio_mask, reduction='mean')
        # 预测阶段
        else:
            # 预测阶段, 直接将发射张量emi送入CRF中, 进行decode解码
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            # 将数字化解码结果, 转换成真实BIO标签
            output['decoded_tag'] = [list(map(lambda x: self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_text

            # 将解码结果做深度拷贝
            temp_tag = copy.deepcopy(decoded_tag)
            # 提取序列长度值cur_len
            cur_len = o.size()[1]
            # 将解码序列中长度不足的部分, 补'O'
            for line in temp_tag:
                line.extend([self.bio_vocab['O']] * (cur_len - len(line)))
            # 将解码结果放置于GPU设备上(或CPU)
            bio_gold = torch.tensor(temp_tag).to(device)

        # 将CRF的解码结果, 送入NER嵌入矩阵中
        tag_emb = self.bio_emb(bio_gold)
        # 将编码器encoder的输出张量o, 和NER嵌入矩阵的输出张量tag_emb, 在最后一个维度上进行拼接
        o = torch.cat((o, tag_emb), dim=2)

        # multi-head-selection的核心代码段
        # B = 32, L = 80, H = 100
        B, L, H = o.size()
        # 下面三行代码依次得到原始公式中U, W, V的结果张量
        u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # 公式中的V, 和关系嵌入张量, 执行爱因斯坦求和
        # uv: [32, 80, 80, 100], relation_emb: [50, 100]
        # torch.einsum()计算后, 希望得到selection_logits: [32, 80, 50, 80]
        selection_logits = torch.einsum('bijh,rh->birj', uv, self.relation_emb.weight)

        # 推理阶段, 直接利用selection_logits进行inference的计算, 得到预测的三元组结果
        if not is_train:
            # 在类内函数inference内部会调用最关键的解码函数selection_decode()
            output['selection_triplets'] = self.inference(mask, text_list, decoded_tag, selection_logits)
            output['spo_gold'] = spo_gold

        selection_loss = 0
        # 训练阶段, 直接计算带mask效果的BCELoss, 这个loss针对于三元组抽取任务
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits, selection_gold)

        # 模型训练的总损失 = NER任务的损失(crf_loss) + 三元组抽取任务的损失(selection_loss)
        loss = crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)
        return output

    # 三元组抽取任务的解码函数
    def selection_decode(self, text_list, sequence_tags, selection_tags):
        # text_list: 批次样本中的原始中文文本, 已经过数字化张量处理
        # sequence_tags: 对应于inference函数中的decoded_tag, 是CRF的解码结果
        # selection_tags: 对应于inference函数中关系矩阵经历爱因斯坦求和后的selection_logits张量
        reversed_relation_vocab = {v: k for k, v in self.relation_vocab.items()}

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))
        # text_list: [['[CLS]', '查', '尔', '斯', '·', '阿', '兰', '基', '斯', '（', 'C', 'h', 'a', 'r', 'l', 'e', 's', ' ', 'A', 'r', 'á', 'n', 'g', 'u', 'i', 'z', '）', '，', '1', '9', '8', '9', '年', '4', '月', '1', '7', '日', '出', '生', '于', '智', '利', '圣', '地', '亚', '哥', '，', '智', '利', '职', '业', '足', '球', '运', '动', '员', '，', '司', '职', '中', '场', '，', '效', '力', '于', '德', '国', '足', '球', '甲', '级', '联', '赛', '勒', '沃', '库', '森', '足', '球', '俱', '乐', '部', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],.....]
        # print('len(text_list):', len(text_list))
        # len(text_list): 4

        def find_entity(pos, text, sequence_tags):
            entity = []
            # print('pos:', pos)

            # NER标记为'B', 'O'时, 代表单字符实体, 或者非实体字符
            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                # 采用逆向遍历的方法, 先检测'I', 直到发现'B'时一个完整的实体提取完毕
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    # 1: 检测到文本的最前面, 实体提取完毕
                    if pos < 0:
                        break
                    # 2: 检测到'B', 实体提取完毕
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                # 逆向后的列表, 就是正向的实体
                entity = list(reversed(temp_entity))
            # print('entity:', entity)
            # entity: ['智', '利', '圣', '地', '亚', '哥']
            # entity: ['查', '尔', '斯', '·', '阿', '兰', '基', '斯']

            # 以连续字符串的格式返回实体
            return ''.join(entity)

        # sequence_tags: [[2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2], [2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
        batch_num = len(sequence_tags)
        # batch_num: 4

        result = [[] for _ in range(batch_num)]
        # print('selection_tags:', selection_tags)
        # selection_tags: tensor([[[[False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False],
        #   ...,
        #   [False, False, False,  ..., False, False, False],
        #   [False, False, False,  ..., False, False, False],
        #   [False,  True,  True,  ..., False, False, False]]]])

        # selection_tags.shape: torch.Size([4, 200, 50, 200])
       
        # 将4维张量中, 等于True对应位置的下标提取成真实样本预测值idx
        idx = torch.nonzero(selection_tags.cpu(), as_tuple=False)
        # idx.shape: torch.Size([14822, 4])
        # idx: tensor([[ 0,  1, 49,  1],
        #              [ 0,  1, 49,  2],
        #              [ 0,  1, 49,  3],
        #              ...,
        #              [ 3, 48, 49, 46],
        #              [ 3, 48, 49, 47],
        #              [ 3, 48, 49, 48]])

        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()
            # print('idx[i]:', idx[i].tolist())
            # idx[i]: [0, 1, 49, 1]

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            
            # idx[i]: [0, 8, 3, 46]
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            # print('tags:', tags)
            # tags: ['O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

            # print('text_list[b]:', text_list[b])
            # text_list[b]: ['[CLS]', '查', '尔', '斯', '·', '阿', '兰', '基', '斯', '（', 'C', 'h', 'a', 'r', 'l', 'e', 's', ' ', 'A', 'r', 'á', 'n', 'g', 'u', 'i', 'z', '）', '，', '1', '9', '8', '9', '年', '4', '月', '1', '7', '日', '出', '生', '于', '智', '利', '圣', '地', '亚', '哥', '，', '智', '利', '职', '业', '足', '球', '运', '动', '员', '，', '司', '职', '中', '场', '，', '效', '力', '于', '德', '国', '足', '球', '甲', '级', '联', '赛', '勒', '沃', '库', '森', '足', '球', '俱', '乐', '部', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

            # 提取客体object的实体字符串
            object = find_entity(o, text_list[b], tags)
            # print('object:', object)
            # object: 智利圣地亚哥
            
            # 提取主体subject的实体字符串
            subject = find_entity(s, text_list[b], tags)
            # print('subject:', subject)
            # subject: 查尔斯·阿兰基斯

            # 确认主体和客体都部不为空
            assert object != '' and subject != ''

            # 提取出来一个三元组, 追加进结果列表中一次
            triplet = {'object': object, 'predicate': predicate, 'subject': subject}
            result[b].append(triplet)
        
        return result

    # def get_metrics(self, reset=False):
    #     pass

