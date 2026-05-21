# 使用正则表达式进行ner任务，适合规范业务场景
import re
import time

# 构造一个soe标签字典
def get_re_label(sentence,start_index,end_index):
    re_label = []
    for i in sentence:
        if i in start_index:
            re_label.append('S')
        elif i in end_index:
            re_label.append('E')
        else:
            re_label.append('O')

    return ''.join(re_label)

# 使用正则表达式进行识别
def ner_re(sentence,start_index,end_index):
    ne_list = []
    label = get_re_label(sentence,start_index,end_index)
    pattern = re.compile('SO*E')
    ne_label = re.finditer(pattern,label)

    for i in ne_label:
        ne_list.append(sentence[int(i.start()):int(i.end())])

    return ne_list

if __name__ == '__main__':
    sentence = '也可接到本决定书之日起六十日内向中国国家市场监督管理总局或者北京市人民政府申请行政复议，杭州海康威视数字技术股份有限公司'
    start_list = ['中','北','杭']
    end_list = ['局','府','司']
    start_time = time.time()
    ne_list = ner_re(sentence,start_list,end_list)
    print(ne_list)
    print(f'用时: {(time.time() - start_time):.6f} 秒')