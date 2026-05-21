import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import copy
from typing import Dict, List, Tuple, Set, Optional
from functools import partial
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




class MultiHeadSelection(nn.Module):
    def __init__(self, hyper):
        super(MultiHeadSelection, self).__init__()
        self.hyper = hyper
        self.data_root = hyper['data_root']
        self.gpu = hyper['gpu']

        self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}

        self.word_embeddings = nn.Embedding(len(self.word_vocab), hyper['emb_size'])

        self.relation_emb = nn.Embedding(len(self.relation_vocab), hyper['rel_emb_size'])
        # bio + pad
        self.bio_emb = nn.Embedding(len(self.bio_vocab), hyper['bio_emb_size'])

        if hyper['cell_name'] == 'gru':
            self.encoder = nn.GRU(hyper['emb_size'],
                                  hyper['hidden_size'],
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper['cell_name'] == 'lstm':
            self.encoder = nn.LSTM(hyper['emb_size'],
                                   hyper['hidden_size'],
                                   bidirectional=True,
                                   batch_first=True)
        elif hyper['cell_name'] == 'bert':
            pass
        else:
            raise ValueError('cell name should be gru/lstm/bert!')

        if hyper['activation'].lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper['activation'].lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)

        self.selection_u = nn.Linear(hyper['hidden_size'] + hyper['bio_emb_size'], hyper['rel_emb_size'])
        self.selection_v = nn.Linear(hyper['hidden_size'] + hyper['bio_emb_size'], hyper['rel_emb_size'])
        self.selection_uv = nn.Linear(2 * hyper['rel_emb_size'], hyper['rel_emb_size'])
        self.emission = nn.Linear(hyper['hidden_size'], len(self.bio_vocab) - 1)

    def inference(self, mask, text_list, decoded_tag, selection_logits, is_predict=False):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2)
        # [batch, seq, rel, seq]
        selection_mask = selection_mask.expand(-1, -1, len(self.relation_vocab), -1)
        selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > self.hyper['threshold']
        selection_triplets = self.selection_decode(text_list, decoded_tag, selection_tags, is_predict)
        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2)
        # [batch, seq, rel, seq]
        selection_mask = selection_mask.expand(-1, -1, len(self.relation_vocab), -1)
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(output['loss'].item(),                                                                                 output['crf_loss'].item(),
                                                                           output['selection_loss'].item(),                                                                           epoch,
                                                                           epoch_num)

    def forward(self, sample, is_train, is_predict=False):
        # print("DEBUG: Start Forwarding")
        # device = torch.device("cuda:" + str(self.gpu) if torch.cuda.is_available() else "cpu")
        device = next(self.parameters()).device

        # print('**********************')
        # print('tokens_id:', sample.tokens_id)
        # print('selection_id:', sample.selection_id)
        # print('bio_id:', sample.bio_id)
        tokens = sample.tokens_id.to(device)
        selection_gold = sample.selection_id.to(device)
        bio_gold = sample.bio_id.to(device)

        # print('**********************')
        # print('text:', sample.text)
        # print('spo_gold:', sample.spo_gold)
        # print('bio:', sample.bio)
        # print('######################')
        # print(HELLO_WORLD)
        text_list = sample.text
        spo_gold = sample.spo_gold
        bio_text = sample.bio

        if self.hyper['cell_name'] in ('gru', 'lstm'):
            # [batch_size, seq_len]
            mask = tokens != self.word_vocab['<pad>']
            bio_mask = mask
        elif self.hyper.cell_name in ('bert'):
            pass
        else:
            raise ValueError('unexpected encoder name!')

        if self.hyper['cell_name'] in ('lstm', 'gru'):
            embedded = self.word_embeddings(tokens)
            pack_padded_embedded = pack_padded_sequence(embedded, sample.length, batch_first=True)
            o, h = self.encoder(pack_padded_embedded)
            o, _ = pad_packed_sequence(o, batch_first=True)
            o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper['hidden_size'], dim=2))
        elif self.hyper.cell_name == 'bert':
            pass
        else:
            raise ValueError('unexpected encoder name!')
        
        emi = self.emission(o)
        output = {}
        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold, mask=bio_mask)
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            # print('*******************')
            # print('decoded_tag:', decoded_tag)
            output['decoded_tag'] = [list(map(lambda x : self.id2bio[x], tags)) for tags in decoded_tag]
            # print('output["decoded_tag"]:', output['decoded_tag'])
            # print('*******************')
            output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] * (tokens.size()[1] - len(line)))
            bio_gold = torch.tensor(temp_tag).to(device)

        tag_emb = self.bio_emb(bio_gold)
        o = torch.cat((o, tag_emb), dim=2)

        # forward multi head selection
        B, L, H = o.size()
        u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # vf(uz_j+Wz_i)  vf(uv[uz_j;Wz_i])

        # correct one 32x86x86x100 50x100
        selection_logits = torch.einsum('bijh,rh->birj', uv, self.relation_emb.weight)

        # use loop instead of matrix
        # selection_logits_list = []
        # for i in range(self.hyper.max_text_len):
        #     uvi = uv[:, i, :, :]
        #     sigmoid_input = uvi
        #     selection_logits_i = torch.einsum('bjh,rh->brj', sigmoid_input,
        #                                         self.relation_emb.weight).unsqueeze(1)
        #     selection_logits_list.append(selection_logits_i)
        # selection_logits = torch.cat(selection_logits_list,dim=1)

        if not is_train:
            output['selection_triplets'] = self.inference(mask, text_list, decoded_tag, selection_logits, is_predict)
            output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits, selection_gold)

        loss = crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss
        output['description'] = partial(self.description, output=output)
        # print("DEBUG: End Forwarding")
        
        return output

    def selection_decode(self, text_list, sequence_tags, selection_tags, is_predict=False):
        reversed_relation_vocab = {v: k for k, v in self.relation_vocab.items()}
        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        if not is_predict:
            text_list = list(map(list, text_list))
        else:
            temp_list = []
            temp_list.append(text_list)
            temp_list.append(text_list)
            text_list = temp_list

        def find_entity(pos, text, sequence_tags):
            entity = []
            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                
                entity = list(reversed(temp_entity))
            
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu(), as_tuple=False)
       
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
          
            object1 = find_entity(o, text_list[b], tags)
            subject1 = find_entity(s, text_list[b], tags)

            assert object1 != '' and subject1 != ''

            triplet = {'object': object1, 'predicate': predicate, 'subject': subject1}
            result[b].append(triplet)
        return result

    def get_metrics(self, reset):
        pass

