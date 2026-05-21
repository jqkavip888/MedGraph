# 导入工具包
import os
import argparse
import torch
from tqdm import tqdm
from config.config import read_config
from torch.optim import Adam, SGD, AdamW
from preprocessings.duie_selection import DuIE_selection_preprocessing
from models.selection import MultiHeadSelection
from dataloaders.selection_loader import Selection_Dataset, Selection_loader
from prefetch_generator import BackgroundGenerator
from metrics.F1_score import F1_triplet, F1_ner
# from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, default='duie_selection_re', help='experiments/duie_selection_re.json')
parser.add_argument('--mode', '-m', type=str, default='train', help='preprocessing|train|evaluation')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.model_dir = './saved_model'
        self.config = read_config(os.path.join('experiments', self.exp_name + '.json'))
        self.gpu = self.config['gpu']
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None

    # 设置优化器的函数
    def _optimizer(self, name, model):
        # 分模块的优化器设置
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }]

        m = {
            # 优化器AdamW(AdamWeightDecayOptimizer), 微调BERT时可以加速收敛
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5),
            'adamw': AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
            }

        return m[name]

    # 初始化模型
    def _init_model(self):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 1. 定义设备并绑定到 self
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiHeadSelection(self.config).to(self.device)

        # 2. 判断是否可以使用 AMP 并绑定到 self
        self.use_amp = torch.cuda.is_available()

        # 3. 初始化 GradScaler 并绑定到 self（这一步极其关键，保存 AMP 的状态）
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # 4. 初始化模型并推到设备上
        self.model = MultiHeadSelection(self.config).to(self.device)

    # 预处理阶段需要执行的函数
    def preprocessing(self):
        if self.exp_name == 'duie_selection_re':
            self.preprocessor = DuIE_selection_preprocessing(self.config)

        # 依次生成关系元数据字典, 生成所有的训练集, 测试集样本数据, 生成字典
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # 生成BIO任务的字典
        self.preprocessor.gen_bio_vocab()

    # 运行函数
    def run(self, mode):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.config['optimizer'], self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.config['evaluation_epoch'])
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    # 加载已经训练好的模型
    def load_model(self, epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.exp_name + '_' + str(epoch))))

    # 保存模型
    def save_model(self, epoch):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    # 评估函数
    def evaluation(self):
        # 初始化数据集和数据迭代器
        dev_set = Selection_Dataset(self.config, self.config['dev'])
        loader = Selection_loader(dev_set, batch_size=self.config['eval_batch'], pin_memory=True)
        self.triplet_metrics.reset()
        # 将模型设置为评估模式
        self.model.eval()

        with torch.no_grad():
            for batch_ndx, sample in tqdm(enumerate(BackgroundGenerator(loader))):
            # 遍历验证集数据集
            # for batch_ndx, sample in enumerate(BackgroundGenerator(loader)):
                # 将测试集样本送入模型处理, 得到输出张量字典output
                output = self.model(sample, is_train=False)

                # 张量字典output, 先完成RE任务的数据抽取
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                # 张量字典output, 再完成NER任务的数据抽取
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            # 遍历所有测试集样本后, 先完成RE任务的评估
            triplet_result = self.triplet_metrics.get_metric()
            # 遍历所有测试集样本后, 再完成NER任务的评估
            ner_result = self.ner_metrics.get_metric()

            # 最后将评估结果直接进行格式化打印
            print('Triplets-> ' + ', '.join(["%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
                ]) + ' ||' + 'NER->' + ', '.join(["%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")]))


    # 训练函数
    def train(self):
        # 构造训练集数据和数据迭代器
        train_set = Selection_Dataset(self.config, self.config['train'])
        loader = Selection_loader(train_set, batch_size=self.config['train_batch'], pin_memory=True)

        # 经典双重for循环, 训练若干个epochs
        for epoch in range(self.config['epoch_num']):
            # 将模型设置为训练模式
            self.model.train()

            for batch_idx, sample in tqdm(enumerate(loader)):
            # for batch_idx, sample in enumerate(loader):    
                # 经典"老三样", 并通过模型得到训练损失值

                # 把数据推到设备上
                # inputs = batch_idx['input'].to(self.device)
                # targets = batch_idx['target'].to(self.device)

                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type,dtype=torch.float16, enabled=self.use_amp):
                    output = self.model(sample, is_train=True)
                    loss = output['loss']
                # loss.backward()
                # self.optimizer.step()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # 每一个epoch训练结束后, 保存一个模型
            self.save_model(epoch)

            # 每一个epoch训练结束后, 进行一个测试集上的评估
            if epoch >= 0:
                self.evaluation()


if __name__ == '__main__':
    # 构造运行类Runner的对象
    runner = Runner(exp_name=args.exp_name)
    # 调用训练类对象的运行函数run, 进行模型的训练或评估(第一次开始前有数据预处理的步骤)
    runner.run(mode=args.mode)

