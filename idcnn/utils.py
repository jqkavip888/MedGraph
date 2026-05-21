# 数据预处理
from idcnn.config_idcnn import *
import os

abs_path = os.path.dirname(os.path.abspath(__file__))



# 数据集初始化
class InputFeatures(object):
    def __init__(self, text, label, input_id, label_id, input_mask):
        self.text = text
        self.label = label
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


# 读取字典，建立一个index表
def load_vocab(vocab_file):
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            # 跳过重复的
            if line in vocab:
                continue

            vocab[line] = index  # 字典添加kv对
            index += 1

    return vocab


# 将数据集text和label分开
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = []
        label = []
        texts = []
        labels = []
        # 以自然句为单位，将处理好的两个临时list放入两个最终list中
        for line in file.readlines():
            if line != '\n':
                line = line.strip().split(' ')
                text.append(line[0])
                label.append(line[-1])

            else:
                texts.append(text)
                labels.append(label)
                text = []
                label = []

    return texts, labels


# 封装可迭代对象
def load_data(file_path, max_length, label_dict, vocab):
    """
    制作一个封装好的数据迭代器
    :param file_path: 数据集
    :param max_length: 句子最大长度
    :param label_dict: 上面构建的标签字典
    :param vocab: token字典
    :return: 封装好的可迭代器
    """
    texts, labels = load_file(file_path)        # 使用把分开的text和label读取进来
    assert len(texts) == len(labels)            # 整体判断是否对齐
    result = []                                 # 设置一个空list用来存放迭代器
    for i in range(len(texts)):
        # 判断单字符是否对齐
        assert len(texts[i]) == len(labels[i])
        token = texts[i]
        label = labels[i]

        # 数据裁剪，提前去掉标识符位置
        if len(token) > max_length - 2:
            token = token[:(max_length - 2)]
            label = label[:(max_length - 2)]

        # 设置语义标识符
        token_f = ['[CLS]'] + token + ['[SEP]']
        label_f = ['<start>'] + label + ['<eos>']

        # token做张量化映射
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in token_f]     # 在字典则映射，不在则unk
        label_ids = [label_dict[i] for i in label_f]

        # 构造输入数据和掩码
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dict['<pad>'])     # label不用0，label用字典设定的pad来补位


        # 确认数据长度
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length

        # 封装
        feature = InputFeatures(text=token_f, label=label_f,input_id=input_ids,
                                input_mask=input_mask,label_id=label_ids)

        result.append(feature)

    return result


# 利用标签字典恢复真实的预测标签
def recover_label(pred_var, gold_var, l2i_dic, i2l_dic):
    assert len(pred_var) == len(gold_var)
    pred_variable = []
    gold_variable = []

    for i in range(len(gold_var)):
        start_index = gold_var[i].index(l2i_dic['<start>'])
        end_index = gold_var[i].index(l2i_dic['<eos>'])
        pred_variable.append(pred_var[i][start_index:end_index])
        gold_variable.append(gold_var[i][start_index:end_index])

    pred_label = []
    gold_label = []
    for j in range(len(gold_variable)):
        pred_label.append([i2l_dic[t] for t in pred_variable[j]])
        gold_label.append([i2l_dic[t] for t in gold_variable[j]])

    return pred_label, gold_label


# 计算NER的关键指标
def get_ner_fmeasure(golden_lists, predict_lists, label_type='BMES'):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0

    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]

        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)

        if label_type == 'BMES':
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)

    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num

    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    accuracy = (right_tag + 0.0) / all_tag
    print('gold_num = ', golden_num, ' pred_num = ', predict_num, ' right_num = ', right_num)

    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position: input_len] + input_string[0: target_position]
    return output_string


def get_ner_BMES(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []

    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, '', 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, '', 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, '', 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ''
            index_tag = ''
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)

    return stand_matrix


def save_model(path, model, epoch):
    pass


def load_model(path, model):
    return model


if __name__ == '__main__':
    pass

