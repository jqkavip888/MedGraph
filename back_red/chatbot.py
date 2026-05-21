# 联调测试相关模块

import sys
sys.path.insert(0, '/root/knowledge_graph')

import os
import time
from answer_search import *
from chatgpt import ChatGPT
from question_classifier import *
from question_parser import *
from chatgpt import ChatGPT

import atexit


class Red_Spider:
    def __init__(self,flag='unit',model_path='./gpt2_chinese_base'):
        # 问题分类器
        print('初始化分类器QuestionClassifer...')
        self.classifier = QuestionClassiferONNX()
        # 问题解析器
        print('初始化翻译器QuestionPaser...')
        self.parser = QuestionPaser()
        # 答案搜索器
        print('初始化搜索器AnswerSearch...')
        self.searcher = AnswerSearch()
        # chatGPT
        print('初始化Llama...')
        self.generator = ChatGPT(flag,model_path)


    def chat_main(self,sentence):
        # 设置客套话
        answer = '您好，我是8号楼郝大夫，值此新春佳节来临之际，祝您身体健康，好运常在'
        t1 = time.time()
        res_classify = self.classifier.classify(sentence)
        t2 = time.time()
        print('classifier问题分类耗时: {}ms'.format((t2 - t1) * 1000))

        # 首先对问题进行分类，如果无法分类，则回复客套话，或使用gpt回复
        res_classify = self.classifier.classify(sentence)
        if not res_classify:
            # return answer
            return self.generator.chat(sentence)

        # 对问题分类进行解析，组装neo4j查询语句
        res_sql = self.parser.parser_main(res_classify)

        # 使用查询语句，调用答案搜索器查询neo4j，得到答案
        final_answers = self.searcher.search_main(res_sql)

        # 将答案进行分行返回，避免输出问题，如果找不到答案则返回客套话
        if not final_answers:
            return self.generator.chat(sentence)
        else:
            return '\n'.join(final_answers)

    # 问答关闭模块
    def close(self):
        self.searcher.driver.close()
        self.generator.generator.llm.close()



if __name__ == '__main__':
    print('实例化AI助手...')
    start_time = time.time()
    flag = 'llama'
    # model_path = './gpt2_chinese_base'
    model_path = '/root/knowledge_graph/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
    redspider = Red_Spider(flag, model_path)
    end_time = time.time()
    print('cost time: ',end_time-start_time)
    print('您好，我是8号楼郝大夫，值此新春佳节来临之际，祝您身体健康，你问我什么都可以')
    # 循环多轮对话
    while True:
        try:
            question = input('用户输入: ')
        except UnicodeDecodeError:
            print('输入有误，请重新输入')
            continue
        if question == 'exit' or question == '退出':
            print('再见！')
            atexit.register(redspider.close)
            break
        answer = redspider.chat_main(question)
        print('AI助理: ', answer)


