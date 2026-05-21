import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import ahocorasick
import numpy as np
import time
import torch
from importlib import import_module
import onnxruntime

UNK,PAD,CLS = "[UNK]","[PAD]","[CLS]"


# 实现问题分类的主类代码
class QuestionClassiferONNX:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.disease_path = os.path.join(cur_dir, 'dict/disease.txt')
        self.drug_path = os.path.join(cur_dir, 'dict/drug.txt')
        self.food_path = os.path.join(cur_dir, 'dict/food.txt')
        self.symptom_path = os.path.join(cur_dir, 'dict/symptom.txt')

        # 加载特征词
        self.disease_words = [i.strip() for i in open(self.disease_path, encoding='utf-8') if i.strip()]
        self.drug_words = [i.strip() for i in open(self.drug_path, encoding='utf-8') if i.strip()]
        self.food_words = [i.strip() for i in open(self.food_path, encoding='utf-8') if i.strip()]
        self.symptom_words = [i.strip() for i in open(self.symptom_path, encoding='utf-8') if i.strip()]
        self.region_words = set(self.disease_words + self.drug_words + self.food_words + self.symptom_words)
        # 构造领域actree, 可以加速关键词匹配查找
        self.region_tree = self.build_actree(list(self.region_words))
        # 构造词典
        self.wdtype_dict = self.build_wdtype_dict()
        self.CLS = ['CLS']
        self.pad_size = 40
        # 初始化bert模型，用于意图识别
        self.init_bert()
        # 初始化onnxruntime推理模型
        self.init_onnx()

        # 问句疑问词, 当前版本仅支持症状, 食品, 药品的查询
        # self.symptom_request = ['症状', '表征', '现象', '症候', '表现', '不良反应', '副作用', '炎症', '发炎', '难受', '不舒服']
        # self.food_request = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱',
        #                      '菜谱', '食用', '食物','饭', '发物', '能不能吃', '能不能喝']
        # self.drug_request = ['药', '药品', '药物', '胶囊', '针剂', '片剂', '吊瓶','疫苗','消炎药', '抗生素','怎么治','怎么办',
        #                      '点滴','贴剂','内用','外用','内服','外敷','口服液', '炎片','吃药', '用药', '服药']


    # bert初始化
    def init_bert(self):
        model_name = 'bert'
        x = import_module("models."  + model_name)
        config = x.Config('red_spider')
        self.model = x.Model(config).to(config.device)
        self.model.load_state_dict(torch.load(config.save_path))
        self.tokenizer = config.tokenizer

        print('model init finished......')


    # onnx推理加速器初始化
    def init_onnx(self):
        self.onnx_model_path = "/root/knowledge_graph/models/bert.onnx"
        operator_export_type = torch._C._onnx.OperatorExportTypes.ONNX
        onnx_input = self.make_onnx_infer_input()

        # 动态轴
        dynamic_axes = {'input_ids':[1],'attention_mask':[1],'token_type_ids':[1]}
        # 偷懒写法dynamic_axes = {'input_ids':[0,1,2],'attention_mask':[0,1,2],'token_type_ids':[0,1,2]}
        out = torch.onnx.export(self.model,
                                onnx_input,
                                self.onnx_model_path,
                                export_params=True,
                                verbose=False,
                                operator_export_type=operator_export_type,
                                opset_version=11,   # 低于10不支持动态轴,the version must be > 10 to support dynamic_axes
                                input_names=['input_ids', 'attention_mask', 'token_type_ids'],
                                dynamic_axes=dynamic_axes)

        # providers = ['CPUExecutionProvider', 'CUDAExecutionProvider','TensorrtExecutionProvider']
        providers = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(self.onnx_model_path, providers=providers)


    # 定义onnx形状,example the shape of onnx, you could fill any number out
    def make_onnx_infer_input(self):
        onnx_input_ids = torch.LongTensor([[15,23,33,41,52,68,72,80,108]])
        onnx_input_mask = torch.LongTensor([[1,1,1,1,1,0,0,0,0]])
        onnx_token_type_ids = torch.LongTensor([[1,1,1,1,1,1,1,1,1]])
        onnx_input_mask = onnx_input_mask.to('cpu')
        onnx_token_type_ids = onnx_token_type_ids.to('cpu')

        return (onnx_input_ids, onnx_input_mask, onnx_token_type_ids)


    # onnx推理
    def onnx_infer(self,question):
        onnx_input = self.tokenizer.encode_plus(text=list(question), return_tensors='pt')
        pred_onnx = self.onnx_session.run(None,
                                          {'input_ids': onnx_input['input_ids'].numpy(),
                                           'attention_mask': onnx_input['attention_mask'].numpy(),
                                           'token_type_ids': onnx_input['token_type_ids'].numpy()})

        # pred_onnx: [array([[-1.0477493 , -0.949046  ,  1.3996799 ,  0.47712737]], dtype=float32)]
        predict_res = np.argmax(pred_onnx[0][0])


        return predict_res

    def question_class(self,question):
        self.model.eval()
        # 数据预处理
        tokens = self.tokenizer.tokenize(question)
        tokens = self.CLS + tokens
        mask = []

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        length = len(token_ids)

        # 补齐长度构造mask
        if length < self.pad_size:
            mask = [1] * length + [0] * (self.pad_size - length)
            token_ids += [0] * (self.pad_size - length)
        else:
            mask = [1] * length
            token_ids = token_ids[:self.pad_size]

        # 类型封装input_ids,mask,token_type_ids
        input_ids = torch.LongTensor(token_ids).unsqueeze(0)
        mask = torch.LongTensor(mask).unsqueeze(0)
        token_type_ids = [1] * self.pad_size
        token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0)

        # 直接输入模型，得到最后一层cls输出
        output = self.model(input_ids,mask,token_type_ids)
        # tensor([[-1.0107, -0.9093, 4.3177, -2.3370]],
        # 贪心算法取最大
        pre_res = torch.argmax(output.data,1).cpu().numpy()

        return pre_res[0]



    # 分类主函数
    def classify(self, question):
        data = {}
        medical_dict = self.check_medical(question)
        if not medical_dict:
            return {}

        data['args'] = medical_dict
        types = []
        for type_ in medical_dict.values():
            types += type_
        question_type = 'others'

        question_types = []

        # # 症状
        # if self.check_words(self.symptom_request, question) and ('disease' in types):
        #     question_type = 'disease_symptom'
        #     question_types.append(question_type)
        #
        # # 推荐食品
        # if self.check_words(self.food_request, question) and ('disease' in types):
        #     question_type = 'disease_food'
        #     question_types.append(question_type)
        #
        # # 推荐药品
        # if self.check_words(self.drug_request, question) and ('disease' in types):
        #     question_type = 'disease_drug'
        #     question_types.append(question_type)
        #
        # # 如果没有查到相关的外部查询信息, 则将该疾病的描述信息返回
        # if question_types == [] and 'symptom' in types:
        #     question_types = ['disease_symptom']
        #
        # # 将多个分类结果进行合并处理, 组装成一个字典
        # data['question_types'] = question_types

        # V2.0版本将question_class()函数利用BERT训练好的4分类器直接进行判断分类

        # V2.5版本将onnx_infer()函数利用BERT训练好的4分类器直接进行分类, 并采用onnx加速机制
        res = self.onnx_infer(question)
        # 疾病-症状查询类型==2
        if res == 2 and ('disease' in types):
            question_type = 'disease_symptom'
            question_types.append(question_type)

        # 疾病-推荐食品类型==1
        if res == 1 and ('disease' in types):
            question_type = 'disease_food'
            question_types.append(question_type)

        # 疾病-推荐药品类型==0
        if res == 0 and ('disease' in types):
            question_type = 'disease_drug'
            question_types.append(question_type)

        # 疾病-治疗方法类型==3
        if res == 3 and ('disease' in types):
            question_type = 'disease_cureway'
            question_types.append(question_type)

        # 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
        if question_types == [] and ('symptom' in types):
            question_types.append('disease_symptom')

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    # 构造关键词对应的节点类型
    def build_wdtype_dict(self):
        word_dict = dict()
        for word in self.region_words:
            word_dict[word] = []
            # 检查是否有疾病关键词
            if word in self.disease_words:
                word_dict[word].append('disease')

            # 检查是否有药品关键词
            if word in self.drug_words:
                word_dict[word].append('drug')

            # 检查是否有食品关键词
            if word in self.food_words:
                word_dict[word].append('food')

            # 检查是否有症状关键词
            if word in self.symptom_words:
                word_dict[word].append('symptom')

        return word_dict

    # 构造actree进行加速过滤
    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))

        actree.make_automaton()
        return actree

    # 问句检查
    def check_medical(self, question):
        region_words = []
        # 利用actree加速查询关键词
        for i in self.region_tree.iter(question):
            word = i[1][1]
            region_words.append(word)

        stop_words = []
        # 子词进入停用词表
        for word1 in region_words:
            for word2 in region_words:
                if word1 in word2 and word1 != word2:
                    stop_words.append(word1)

        final_words = [i for i in region_words if i not in stop_words]
        final_words = {i: self.wdtype_dict.get(i) for i in final_words}

        return final_words

    # 基于特征词进行问句检测, 并进行问句类型的规则分类
    def check_words(self, words, sent):
        for word in words:
            if word in sent:
                return True

        return False


if __name__ == '__main__':
    qc = QuestionClassiferONNX()
    while True:
        question = input('input an question: ')
        data = qc.classify(question)
        print(data)
        break
