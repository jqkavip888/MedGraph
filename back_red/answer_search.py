import sys
sys.path.insert(0, '/root/knowledge_graph')
import os
import json
from neo4j import GraphDatabase
from neo4j_utils.neo4j_config import NEO4J_CONFIG

# 答案查询主类
class AnswerSearch:
    def __init__(self):
        self.num_limit = 10
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)


    def search_main(self,sqls):
        final_answers = []

        # 开启回话
        with self.driver.session() as session:
            for sql_ in sqls:
                question_type = sql_['question_type']
                queries = sql_['sql']
                answers = []


                # 遍历查询cypher，把结果添加到list
                for query in queries:
                    ress = session.run(query).data()
                    answers += ress

                # 调用精准回复模版
                final_answer = self.answer_prettify(question_type, answers)
                if final_answer:
                    final_answers.append(final_answer)

        return final_answers

    # 根据对应的question_types调用和组装相应的回复模版
    def answer_prettify(self,question_type,answers):
        final_answers = []
        if not answers:
            return ''

        if question_type == 'disease_symptom':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answers = '{0}的症状包括：{1}'.format(subject,';'.join(list(set(desc))[:self.num_limit]))

        if question_type == 'disease_food':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answers = '{0}的食谱包括：{1}'.format(subject,';'.join(list(set(desc))[:self.num_limit]))

        if question_type == 'disease_drug':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answers = '{0}的药品包括：{1}'.format(subject,';'.join(list(set(desc))[:self.num_limit]))


        return final_answers

if __name__ == '__main__':
    ans = AnswerSearch()
    print(ans)
