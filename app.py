import sys
sys.path.insert(0, '/root/knowledge_graph')

from back_red.question_classifier import *
from back_red.question_parser import *
from back_red.answer_search import *
import time
from chatgpt import ChatGPT


# 导入flask框架
from flask import Flask, render_template, request
app = Flask(__name__)


LLAMA_MODEL_PATH = '/root/knowledge_graph/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf'


class Red_Spider:
    def __init__(self):
        # 问题分类器
        self.classifier = QuestionClassiferONNX()
        # 问题解析器
        self.parser = QuestionPaser()
        # 答案搜索器
        self.searcher = AnswerSearch()
        # Llama 兜底
        print('初始化Llama...')
        self.generator = ChatGPT(flag='llama', model_path=LLAMA_MODEL_PATH)

    def chat_man(self,sentence):
        total_start = time.time()
        llama_stats = None

        # ── 1. 分类 ──────────────────────────────────────────
        t1 = time.time()
        # 设置客套话set a default answer
        answer = '您好，我是8号楼郝大夫，值此新春佳节来临之际，祝您身体健康，好运常在'

        # 首先对问题进行分类，如果无法分类，则回复客套话/Llama 兜底
        # if the question is not be classify, send the response by llama
        res_classify = self.classifier.classify(sentence)

        classify_ms = (time.time() - t1) * 1000
        print(f'[Classifier] 分类耗时: {classify_ms:.2f}ms')

        if not res_classify:
            answer, llama_stats = self.generator.chat(sentence)
            self._print_summary(classify_ms, None, llama_stats, time.time() - total_start)
            return answer

        # 对问题分类进行解析，组装neo4j查询语句
        # analysis the question and integrate the whole query cypher of neo4j
        t2 = time.time()

        res_sql = self.parser.parser_main(res_classify)

        # 使用查询语句，调用答案搜索器查询neo4j，得到答案
        final_answers = self.searcher.search_main(res_sql)

        neo4j_ms = (time.time() - t2) * 1000
        print(f'[Neo4j]      查询耗时: {neo4j_ms:.2f}ms')

        # 将答案进行分行返回，避免输出问题，如果找不到答案则返回客套话/Llama 兜底
        if not final_answers:
            # return self.generator.chat(sentence)
            answer, llama_stats = self.generator.chat(sentence)
            self._print_summary(classify_ms, neo4j_ms, llama_stats, time.time() - total_start)
            return answer
        else:
            self._print_summary(classify_ms, neo4j_ms, None, time.time() - total_start)
            return '\n'.join(final_answers)

    def _print_summary(self, classify_ms, neo4j_ms, llama_stats, total_s):
        print('─' * 55)
        print(f'  分类耗时   : {classify_ms:.2f} ms')
        if neo4j_ms is not None:
            print(f'  Neo4j耗时  : {neo4j_ms:.2f} ms')
        if llama_stats:
            print(f'  Llama耗时  : {llama_stats["elapsed"]:.3f} s')
            print(f'  生成tokens : {llama_stats["completion_tokens"]}')
            print(f'  推理速度   : {llama_stats["tokens_per_sec"]} tokens/s  ← 简历指标')
        print(f'  端到端延迟 : {total_s * 1000:.1f} ms')
        print('─' * 55)

# 实例化ai机器人
start_time = time.time()
red_spider = Red_Spider()
end_time = time.time()
print('cost time: ', end_time - start_time)
print('AI助手初始化完毕...')

@app.route('/V1/main_server',methods=['POST'])
def main_server():
    # 接受来自发送方的字段
    uid = request.form.get('uid','unknown')
    text = request.form.get('text','')

    # 调用机器人执行查询与回复
    answer = red_spider.chat_man(text)

    return answer


if __name__ == '__main__':
    # host='0.0.0.0' 表示监听所有网卡，这样 192.168... 才能访问
    # debug=True 方便你改代码后 Flask 自动重启
    app.run(host='0.0.0.0', port=19999, debug=False)