# AI Doctor is a robot dialog system what base on healthcare.
*ai医生是一个基于医疗数据集的机器人对话系统，这只是个LLM作业而已，由于数据集和模型文件较大，已经删除了duie数据集和模型*

it is just a LLM engnieer homework/personal project, i have deleted train/test data and model because what is size problem

Python · PyTorch · Neo4j · Llama(GPT2) · NER/RE · IDCNN/BiLstm/CRF/Multi-Head-Selection · ONNX · Flask

---

## project overview
项目概览


```
离线部分 offline part:

medical.json 结构化数据   construction of data
        ↓
build_medicalgraph.py
        ↓
graph_database(Neo4j)


开源数据集   duIE train data
    ↓
BERT/BiLSTM encode 
    ↓
CRF头 → BIEO标签（NER） CRF head → BIEO labels（NER)
    ↓
NER结果 embedding 后拼接回编码器输出    NER output into embedding layer and couple with encoder to next layer(MHS) out
    ↓
MHS头 → 关系三元组（RE）   MHS head → SPO relationship extraction(RE)


在线部分 online part:

用户输入   user input
    ↓
AC自动机（实体预检）   Actree
    ↓
BERT 意图分类（ONNX）   intent classification by Bert(ONNX)
    ↓
Cypher 查询 Neo4j   query Neo4j
    ↓
GPT2(Llama) 兜底     default conversation module(GPT2, it is place by llama next version which module is better than GPT2)
    ↓
Flask API 返回   
```

---

## technology stack

| catagory | technology |
|------|------|
| framework | Python · PyTorch · Transformers · Flask |
| NLP | IDCNN+CRF · BiLSTM · BERT · Multi-head Selection |
| inference acceleration | ONNX Runtime · mixed precision trainning |
| graph database | Neo4j · Cypher |
| training platform | AutoDL（RTX 4090 / RTX 5090） |

---

## module details

### ① NER — 命名实体识别 Named Entity Recognition（IDCNN + CRF）

抽取 `disease / symptom / drug / food` 四类医疗实体，采用 BIEO 四标签格式。
extract 4 kinds of label of `disease / symptom / drug / food` from data source, using BIEO 4-labels format

- **IDCNN**：通过膨胀卷积扩大感受野提取上下文特征，输出发射矩阵
- using dilated conv1d draw the features from context, outputing the emission metrix to CRF layer
- **CRF**：持有转移矩阵，联合发射分数经 Viterbi 解码输出全局最优标注序列
- hold on translation metrix, and combines with emission score(emission metrix) and Viterbi decode to output the best sequence

| dataset | size |
|--------|------|
| train | 93,000 |
| test | 28,000 |

---

### ② RE — 关系抽取（CRF + Multi-head Selection） relationship extraction

使用 **DuIE 开源数据集**训练，推理时迁移至医疗文本（`medical.json`，共 **8,808 条**疾病记录的 `desc / cause` 等自由文本字段），补充结构化字段未覆盖的隐含关系。
using open source DuIE dataset to trainning, hence convert into inference base on `medical.json` dataset of 8,808 diseases record in all

> 存在通用域 → 医疗域的**域迁移**问题，是图谱召回率存在上限的原因之一。
> generation data domain → health data domain, this is main problem of upper limit of graph database

| dataset | size |
|--------|------|
| train | 169,895 |
| test | 21,261 |

**模型对比结果：**
comparase result

| model | Triplets P | Triplets R | Triplets F | NER P | NER R | NER F | remark |
|------|-----------|-----------|-----------|-------|-------|-------|------|
| BiLSTM + CRF + MHS | 0.7522 | 0.5592 | 0.6415 | 0.8848 | 0.8910 | 0.8879 | epoch 21 best |
| **BERT + CRF + MHS（FP32）** | **0.7712** | **0.7658** | **0.7685** | **0.8968** | **0.9404** | **0.9181** | — |
| BERT + CRF + MHS（MPT） | 0.7268 | 0.7277 | 0.7272 | 0.8911 | 0.9416 | 0.9156 | — |

**训练成本：**
trainning cost

RE 输出为四维张量，M1 8GB 设备内存溢出，`batch_size=1` 后估时仍需 24h+，改为云 GPU 后显著缩短：
RE output 4-dimension tensor, run out so far M1 8gb device memory, modify `batch_size=1` need run 24 hour+
switch to lend GPU on cloud could speed up 5-3 hours(rtx 4090/5090) to trainning finish

| device | single trainning time | cash cost |
|------|------------|---------|
| M1 8GB（local） | 24h+（OOM） | 0 |
| AutoDL RTX 4090 | ≈ 5h | about 11 CNY（1.88 CNY/h × training + debugging） |
| AutoDL RTX 5090 | ≈ 3.5h | about 15 CNY（2.88 CNY/h × training + debugging） |

---

### ③ 在线问答系统 — Pipeline 三阶段 Q&A system - pipeline

抽取结果经 Cypher `MERGE` 写入 Neo4j，构建 **23,111 个节点、154,396 条关系**的医疗知识图谱。
construct 23,111 'symptom/disease/food/medical' nodes in neo4j by cypher, and 154,396 relationship edges base on that nodes

**规则路径（Aho-Corasick）**
rule path（Aho-Corasick）

以 `ahocorasick` 将 `disease / symptom / drug / food` 词典合并为多模式匹配树，实体命中速度 **~0.3 ms**，但对模糊语义无法覆盖。
integrate the `disease / symptom / drug / food` vocab into a whole vocab to mutil-catch tree, accelerate to catch to 0.3ms, but could not cover ambiguous semantic

**ONNX 推理加速**
inference acceleration by ONNX framework

将 BERT 转换为 .ONNX 后：
convert Bert into .ONNX:

| path | classification time cost |
|------|---------|
| cold path | 191–211 ms |
| hot path | 0.05–0.28 ms |

加速后瓶颈转移：
bottleneck shift after acceleration:

| module | time cost |
|------|------|
| Neo4j query | 363–423 ms |
| Llama generation | 3,005–3,071 ms |

在 2 核无超线程服务器上，QPS上限 **0.33**（Llama），整体 QPS 上限由 Llama 生成问题导致。
running on vmware server build on M1 8gb macbook, QPS upper to 0.33(llama generation as a result)

> **结论**：单点加速后瓶颈转移，推理优化需系统级整体考量。
> conclusion: inference acceleration optimization need thinking holistically system

**对话兜底模块**
default dialog module

| iter | model | result |
|------|------|------|
| v1 | GPT-2 | semantic chaos |
| v2 | Llama 3.2-1B（Q4） | nutural conversation feeling |

前期采用 GPT-2，效果不理想，不像个人； 换用 Llama 3.2-1B（Q4 量化）后回复质量明显改善，趋近正常对话语感。
V1 GPT-2 — semantic chaos; V2 Llama 3.2-1B (Q4 quantized) — natural conversation quality 
---

## 快速开始

```bash
# 自行安装依赖，目前没有导出
pip install -r requirements.txt

# 自行安装本地模型 Bert/GPT2/llama

# 启动 Flask API
python app.py
```

> 运行前请确保本地已启动 Neo4j 实例，并在配置文件中填写连接信息。
