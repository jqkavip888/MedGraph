# gpt外层封装
import sys
sys.path.insert(0, '/root/knowledge_graph')
import os
import torch
# from unit_robot import *
from gpt2 import GPT2_Red_Spider
from neo4j_utils.neo4j_config import NEO4J_CONFIG
from llama_cpp import Llama
from llama_model import Llama_Red_Spider




class ChatGPT():
    def __init__(self,flag='unit',model_path='./gpt2_chinese_base'):
        self.flag = flag
        # 如果是百度则调用百度unit接口
        if flag == 'unit':
            self.generator = unit_chat

        elif flag == 'gpt2':
            self.generator = GPT2_Red_Spider(model_path)

        elif flag == 'llama':
            self.generator = Llama_Red_Spider(model_path)

        # 调用封神榜模型
        # 调用qwen模型


    def chat(self,input_sentence):
        if self.flag == 'unit':
            res = self.generator.generator(input_sentence)

        else:
            res = self.generator.chat(input_sentence)

        return res