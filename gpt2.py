# 使用gpt语言模型

import torch
import os
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline,pipeline

abs_path = os.path.dirname(os.path.abspath(__file__))

class GPT2_Red_Spider:
    def __init__(self,model_path):
        self.tokenizer = BertTokenizer.from_pretrained(abs_path + '/gpt2_chinese_base')
        self.model = GPT2LMHeadModel.from_pretrained(abs_path + '/gpt2_chinese_base')
        self.generator = TextGenerationPipeline(self.model, self.tokenizer,device=0)


    def chat(self,input_sentence):
        # max_length不能太短，3轮对话之后redis可能会报错
        output = self.generator(input_sentence,max_length=300,
                                pad_token_id=self.tokenizer.pad_token_id,do_sample=True)

        # 截断，限制总长度
        text = output[0]['generated_text'][:50]

        return text