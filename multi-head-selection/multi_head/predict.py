import os
import argparse
import torch
import copy
from tqdm import tqdm
from config.hyper import read_config
from torch.optim import Adam, SGD
from preprocessings.duie_selection import DuIE_selection_preprocessing
from models.selection import MultiHeadSelection
from dataloaders.selection_loader import Selection_Dataset,Selection_loader
from prefetch_generator import BackgroundGenerator
from metrics.F1_score import F1_triplet, F1_ner
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, default='duie_selection_re.json',
                    help='experiments/exp_name.json')
parser.add_argument('--mode', '-m', type=str, default='preprocessing',
                    help='preprocessing|train|evaluation|predict')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'
        self.hyper = read_config(os.path.join('experiments', self.exp_name))
        self.gpu = self.hyper['gpu']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None
        # self.data_root = self.hyper['data_root']
        # self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        # self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))


    def _optimizer(self, name, model):
        m = {'adam': Adam(model.parameters()), 'sgd': SGD(model.parameters(), lr=0.5)}
        return m[name]

    def _init_model(self):
        # device = torch.device("cuda:" + str(self.gpu) if torch.cuda.is_available() else "cpu")
        self.model = MultiHeadSelection(self.hyper).to(self.device)

    def preprocessing(self):
        if self.exp_name == 'duie_selection_re':
            self.preprocessor = DuIE_selection_preprocessing(self.hyper)

        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper['optimizer'], self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper['evaluation_epoch'])
            self.evaluation()
        elif mode == 'predict':
            self._init_model()
            self.load_model(epoch=self.hyper['predict_epoch'])
            sentence = input('Please input a sentence:')
            res = self.predict(sentence)
            print('res = ', res)
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'duie_selection_re_' + str(epoch))))

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
            for batch_ndx, sample in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader)):
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            triplet_result = self.triplet_metrics.get_metric(reset=False)
            ner_result = self.ner_metrics.get_metric(reset=False)
            
            print('Triplets-> ' + ', '.join(["%s: %.4f" % (name[0], value)
                   for name, value in triplet_result.items() if not name.startswith("_")
                   ]) + ' ||' + 'NER->' + ', '.join(["%s: %.4f" % (name[0], value)
                   for name, value in ner_result.items() if not name.startswith("_")]))

    def predict(self, input_text):
        # input_text: 原始中文输入文本
        with torch.no_grad():
            oov = self.model.word_vocab['oov']
            tokens = list(map(lambda x: self.model.word_vocab.get(x, oov), input_text))
            combine = []
            combine.append(tokens)
            combine.append(tokens)
            tokens = torch.tensor(combine).to(self.device)
            lengths = (len(tokens[0]), len(tokens[1]))

            if self.hyper['cell_name'] in ('gru', 'lstm'):
                # [batch_size, seq_len]
                mask = tokens != self.model.word_vocab['<pad>']
                bio_mask = mask
            elif self.hyper.cell_name in ('bert'):
                pass
            else:
                raise ValueError('unexpected encoder name!')

            if self.hyper['cell_name'] in ('lstm', 'gru'):
                embedded = self.model.word_embeddings(tokens)
                # 原始训练代码中的sample.length取值就是原始文本的长度, 这里直接测量即可.
                pack_padded_embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
                o, h = self.model.encoder(pack_padded_embedded)
                o, _ = pad_packed_sequence(o, batch_first=True)
                o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper['hidden_size'], dim=2))
            elif self.hyper.cell_name == 'bert':
                pass
            else:
                raise ValueError('unexpected encoder name!')

            # 获取发射矩阵张量
            emi = self.model.emission(o)
            decoded_tag = self.model.tagger.decode(emissions=emi, mask=bio_mask)
            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.model.bio_vocab['<pad>']] * (tokens.size()[1] - len(line)))
            bio_gold = torch.tensor(temp_tag).to(self.device)

            tag_emb = self.model.bio_emb(bio_gold)
            o = torch.cat((o, tag_emb), dim=2)

            # forward multi head selection
            B, L, H = o.size()
            u = self.model.activation(self.model.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
            v = self.model.activation(self.model.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
            uv = self.model.activation(self.model.selection_uv(torch.cat((u, v), dim=-1)))

            selection_logits = torch.einsum('bijh,rh->birj', uv, self.model.relation_emb.weight)

            # 第二个参数即原始输入文本input_text
            predict_spo_selection = self.model.inference(mask, input_text, decoded_tag, selection_logits, is_predict=True)

            return predict_spo_selection

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper['train'])
        loader = Selection_loader(train_set, batch_size=self.hyper['train_batch'], pin_memory=True)

        for epoch in tqdm(range(self.hyper['epoch_num'])):
            self.model.train()
            
            for batch_idx, sample in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader)):
                self.optimizer.zero_grad()
                
                output = self.model(sample, is_train=True)
                loss = torch.mean(output['loss'])
                
                loss.backward()
                self.optimizer.step()

            # 每一个epoch保存一次当前模型
            self.save_model(epoch)

            # 从第6个epoch开始, 每隔一段训练周期, 进行一次模型验证集上的评估
            if epoch % self.hyper['print_epoch'] == 0 and epoch > 5:
                self.evaluation()

if __name__ == "__main__":
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode=args.mode)


# input sentence: 《愤怒的唐僧》由北京吴意波影视文化工作室与优酷电视剧频道联合制作，故事以喜剧元素为>主，讲述唐僧与佛祖打牌，得罪了佛祖，被踢下人间再渡九九八十一难的故事
