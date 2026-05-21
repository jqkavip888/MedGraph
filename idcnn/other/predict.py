import torch
from idcnn.idcnn_crf import IDCNN_CRF  # 确保这里指向你定义的模型类
from idcnn.train import sentence
from idcnn.utils import load_vocab  # 确保指向你的工具函数
import os


abs_path = os.path.dirname(os.path.abspath(__file__))


# 1. 基础配置（必须与训练时一致）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): device = "mps"  # 支持你的M1

# vocab_file = "../data/vocab.txt"
# model_path = "../data/model/idcnn.pt"  # 指向你刚跑出的那个F1=1.0的模型
vocab_file = os.path.join(abs_path, "../data/vocab.txt")
model_path = os.path.join(abs_path, "../data/model/idcnn.pt")
tag_map = {"O": 0, "B-dis": 1, "I-dis": 2, "E-dis": 3, "B-sym": 4, "I-sym": 5, "E-sym": 6}  # 换成你实际的tag_map
id2tag = {v: k for k, v in tag_map.items()}
vocab = load_vocab(vocab_file)

# 2. 加载模型
# 假设你的IDCNN初始化需要这些参数，请根据实际情况修改
model = IDCNN_CRF(tagset_size=len(tag_map), vocab_size=int(len(vocab)),num_filters=64,
                  dropout=0.4,use_cuda=True,embedding_dim=sentence).to(device)
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # 切换到评估模式


def predict(text):
    # 预处理：转ID
    ids = [vocab.get(char, vocab.get("[UNK]")) for char in text]
    input_tensor = torch.LongTensor([ids]).to(device)
    mask = torch.ones_like(input_tensor).byte()  # 推理时全为1

    with torch.no_grad():
        # 模型推理
        logits = model(input_tensor)
        # CRF解码（得到最可能的路径）
        paths = model.crf.decode(logits, mask)

    # 将结果映射回标签
    res_tags = [id2tag[p] for p in paths[0]]

    # 漂亮地打印输出
    for char, tag in zip(text, res_tags):
        print(f"{char}[{tag}] ", end="")
    print("\n")


# 3. 灵魂拷问：输入一些训练集可能没见过的话
if __name__ == "__main__":
    print("--- 压力测试开始 ---")
    test_sentences = [
        # 1. 组合测试：训练集有“咳嗽”，有“急性上呼吸道感染”，试试组合
        "由于急性上呼吸道感染，我一直在咳嗽。",

        # 2. 干扰项测试：出现实体词，但语境不对（看它会不会误标）
        "我不喜欢吃那种叫感冒灵的糖果。",

        # 3. 未见词测试：用一个你数据里绝对没有的新流行病词（比如“幻阳症”，虽然不是病）
        "我怀疑自己得了幻阳症，感觉浑身酸痛。",

        # 4. 边界测试：看它能否把长词切对
        "双侧精索静脉曲张导致的腹胀极其难受。"
    ]

    for s in test_sentences:
        predict(s)