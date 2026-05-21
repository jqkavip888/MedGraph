import os
import argparse
import torch
from tqdm import tqdm
from config.hyper import read_config
from torch.optim import Adam, SGD
from preprocessings.duie_selection import DuIE_selection_preprocessing
from models.selection import MultiHeadSelection
from dataloaders.selection_loader import Selection_Dataset,Selection_loader
from prefetch_generator import BackgroundGenerator
from metrics.F1_score import F1_triplet, F1_ner


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, default='duie_selection_re.json',
                    help='experiments/exp_name.json')
parser.add_argument('--mode', '-m', type=str, default='preprocessing',
                    help='preprocessing|train|evaluation')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'
        self.hyper = read_config(os.path.join('experiments', self.exp_name))
        self.gpu = self.hyper['gpu']
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None

    def _optimizer(self, name, model):
        m = {'adam': Adam(model.parameters()), 'sgd': SGD(model.parameters(), lr=0.5)}
        return m[name]

    def _init_model(self):
        # device = torch.device("cuda:" + str(self.gpu) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = MultiHeadSelection(self.hyper).to(self.device)


    def preprocessing(self):
        # if self.exp_name == 'duie_selection_re':
        #     self.preprocessor = DuIE_selection_preprocessing(self.hyper)
        # 打印出来看看，到底是多了空格、多了.json，还是大小写不对
        print(f"--- 调试信息：当前 exp_name 为 '{self.exp_name}' ---")

        # 建议使用 .startswith 或者直接强制赋值进行测试
        if self.exp_name.startswith('duie_selection_re'):
            self.preprocessor = DuIE_selection_preprocessing(self.hyper)

        # 防御性编程：如果还是 None，直接报错拦截，不要往下走
        if self.preprocessor is None:
            raise ValueError(f"错误：无法识别的 exp_name '{self.exp_name}'。 "
                             f"请检查 main.py 启动参数或配置文件名。")

        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode):
        if mode == 'preprocessing':
            # self.preprocessing()
            pass
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper['optimizer'], self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper['evaluation_epoch'])
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper['dev'])
        loader = Selection_loader(dev_set, batch_size=self.hyper['eval_batch'], pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        with torch.no_grad():
            # for batch_ndx, sample in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader)):
            for batch_ndx, sample in enumerate(BackgroundGenerator(loader)):    
                output = self.model(sample, is_train=False)
                # print('*************************')
                # print('selection_triplets[0]:', output['selection_triplets'][0])
                # print(len(output['selection_triplets']))
                # print('*************************')
                # print('spo_gold[0]:', output['spo_gold'][0])
                # print(len(output['spo_gold']))
                # print('####################')
                # print(HELLO_WORLD)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            triplet_result = self.triplet_metrics.get_metric(reset=False)
            ner_result = self.ner_metrics.get_metric(reset=False)
            
            print('Triplets-> ' + ', '.join(["%s: %.4f" % (name[0], value)
                   for name, value in triplet_result.items() if not name.startswith("_")
                   ]) + ' ||' + 'NER->' + ', '.join(["%s: %.4f" % (name[0], value)
                   for name, value in ner_result.items() if not name.startswith("_")]))

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper['train'])
        loader = Selection_loader(train_set, batch_size=self.hyper['train_batch'], pin_memory=True)

        for epoch in tqdm(range(self.hyper['epoch_num'])):
            self.model.train()
            
            # for batch_idx, sample in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader)):
            for batch_idx, sample in enumerate(BackgroundGenerator(loader)):
                self.optimizer.zero_grad()
                
                output = self.model(sample, is_train=True)
                loss = torch.mean(output['loss'])
                
                loss.backward()
                self.optimizer.step()

            # 每一个epoch保存一次当前模型
            self.save_model(epoch)

            # 从第5个epoch开始, 每个epoch进行一次模型验证集上的评估
            if epoch >= 5:
                self.evaluation()

if __name__ == "__main__":
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode='train')

