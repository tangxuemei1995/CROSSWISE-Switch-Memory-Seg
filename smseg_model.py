from __future__ import absolute_import, division, print_function

import os

import math, copy

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)
from pytorch_pretrained_bert.tokenization import BertTokenizer
# imort transformer
import pytorch_pretrained_zen as zen

from torch.nn import CrossEntropyLoss
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pytorch_pretrained_bert.crf import CRF
import transformer
from torch.autograd import Variable

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'max_ngram_length': 5,
    'use_bert': False,
    'use_lstm': False,
    'use_trans': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_memory': False,
    'decoder': 'crf'
}


class PositionalEncoding(nn.Module):
    'for trans'
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class WordKVMN(nn.Module):
    def __init__(self, hidden_size, word_size):
        super(WordKVMN, self).__init__()
        self.temper = hidden_size ** 0.5
        self.word_embedding_a = nn.Embedding(word_size, hidden_size)  # ????????????????????????
        self.word_embedding_c = nn.Embedding(10, hidden_size)  # ???????????????????????????
# (word_seq, sequence_output, label_value_matrix, word_mask)
    def forward(self, word_seq, hidden_state, label_value_matrix, word_mask_metrix):
        embedding_a = self.word_embedding_a(word_seq)
        embedding_c = self.word_embedding_c(label_value_matrix)

        embedding_a = embedding_a.permute(0, 2, 1)  # ???tensor???????????????,?????????2???1??????
        u = torch.matmul(hidden_state, embedding_a) / self.temper
        # print(u.size())
	 

        tmp_word_mask_metrix = torch.clamp(word_mask_metrix, 0, 1)  # ??????input???????????????????????????????????? [min,max][min,max]
        # print(tmp_word_mask_metrix.size())
	 
        exp_u = torch.exp(u)
        # print(u.size())
 #        exit()
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)
        # exit()
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)
        
        embedding_c = embedding_c.permute(3, 0, 1, 2)
        o = torch.mul(p, embedding_c)

        o = o.permute(1, 2, 3, 0)
        o = torch.sum(o, 2)

        return o


class WMSeg(nn.Module):

    def __init__(self, word2id, big_word2id, dict2id, labelmap, dataset_map, hpara, args):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec.pop('args')

        self.word2id = word2id
        self.big_word2id = big_word2id
        self.dict2id = dict2id
        self.labelmap = labelmap
        self.train_batch_size = args.train_batch_size
        self.switch = args.switch
        self.attention_mode = args.attention_mode
        # self.classifier = args.classifier
        self.hpara = hpara
        self.dataset_map = dataset_map
        self.num_labels = len(self.labelmap) + 1
        self.num_labels_cls = len(self.dataset_map)
        self.max_seq_length = self.hpara['max_seq_length']
        self.max_ngram_size = self.hpara['max_ngram_size']
        self.max_ngram_length = self.hpara['max_ngram_length']

        self.bert_tokenizer = None
        self.bert = None

        self.zen_tokenizer = None
        self.zen = None
        self.zen_ngram_dict = None
        self.trans = None
        self.lstm = None

        # if self.hpara['use_trans']:
        #            print(11111)
       #  if self.hpara['use_lstm']:
#
#             self.embed = nn.Embedding(21128, 300)
#
#             self.lstm = nn.LSTM(input_size=300, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True)
#
#             self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model,
#                                                                 do_lower_case=self.hpara['do_lower_case'])
#             self.dropout = nn.Dropout(0.2)
#
#             hidden_size = 200
#
#
#             # if args.do_train:
# #                 self.batch_size = self.train_batch_size
# #             else:
# #                 self.batch_size = self.eval_batch_size
#
#         elif self.hpara['use_trans']:
#
#             c = copy.deepcopy
#             # encoder=TransformerEncoder(num_layers=N,model_size=d_model,inner_size=d_ff,key_size=d_model//h,value_size=d_model//h,num_head=h,dropout=dropout)
#
#             self.position = PositionalEncoding(d_model=256, dropout=0.2)
#
#             self.embed = Embedding(d_model=256, vocab=21128)
#             hidden_size = 256  # d_model?????????????????? d_ff
#             self.trans = transformer.make_encoder(N=6, d_model=256, h=4, dropout=0.2, d_ff=512)
#             self.dense_tr = nn.Linear(256, 256)
#             self.activation_tr = nn.Tanh()
#
#             for name, p in self.trans.named_parameters():
#                 if p.dim() > 1 and p.requires_grad == True:
#                     nn.init.xavier_uniform_(p)
#             # self.trans = transformer.make_encoder(N=6, d_model=256, h=4, dropout=0.2, d_ff=1024)
#             # trans?????????bert???tokenizer
#             self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model,
#                                                                 do_lower_case=self.hpara['do_lower_case'])
#             self.dropout = nn.Dropout(0.2)



        if self.hpara['use_bert']:
            if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                                                    do_lower_case=self.hpara['do_lower_case'])
                # print(args.bert_model)
                self.bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                # print(self.bert)
                self.hpara['bert_tokenizer'] = self.bert_tokenizer
                self.hpara['config'] = self.bert.config
            else:
                self.bert_tokenizer = self.hpara['bert_tokenizer']
                self.bert = BertModel(self.hpara['config'])
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
            
        elif self.hpara['use_zen']:
            if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(zen.PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.zen_tokenizer = zen.BertTokenizer.from_pretrained(args.bert_model,
                                                                       do_lower_case=self.hpara['do_lower_case'])
                self.zen_ngram_dict = zen.ZenNgramDict(args.bert_model, tokenizer=self.zen_tokenizer)
                self.zen = zen.modeling.ZenModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                self.hpara['zen_tokenizer'] = self.zen_tokenizer
                self.hpara['zen_ngram_dict'] = self.zen_ngram_dict
                self.hpara['config'] = self.zen.config
            else:
                self.zen_tokenizer = self.hpara['zen_tokenizer']
                self.zen_ngram_dict = self.hpara['zen_ngram_dict']
                self.zen = zen.modeling.ZenModel(self.hpara['config'])
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.hpara['use_memory']:
            self.kv_memory = WordKVMN(hidden_size, len(big_word2id))
        else:
            self.kv_memory = None

        if self.attention_mode == 'cat':

            self.classifier = nn.Linear(hidden_size * 2, self.num_labels, bias=False)
        
        else:
            self.classifier = nn.Linear(hidden_size, self.num_labels, bias=False)

        if args.classifier:
	     
            self.classifier_cls = nn.Linear(hidden_size, self.num_labels_cls, bias=True)

        else:
            self.classifier_cls = None

        if self.hpara['decoder'] == 'crf':
            self.crf = CRF(tagset_size=self.num_labels - 3, gpu=False)
        else:
            self.crf = None

        if args.do_train:
            self.spec['hpara'] = self.hpara
#seg_model(input_ids, segment_ids, input_mask, label_ids, label_cls_id, valid_ids, l_mask, word_ids,
                                    # matching_matrix, word_mask, ngram_ids, ngram_positions, device)
					 
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, labels_cls=None, valid_ids=None,
                attention_mask_label=None, word_seq=None, label_value_matrix=None, word_mask=None,
                input_ngram_ids=None, ngram_position_matrix=None, device=None):

        if self.lstm is not None:
            masks = attention_mask
            word_emb = self.embed(input_ids)
            # print('input_id', input_ids.size())
            seq_length = masks.sum(1)
            # print('seq_length',seq_length.data())
            sorted_seq_length, perm_idx = seq_length.sort(descending=True)
            word_emb = word_emb[perm_idx, :]
            # print('word_emb', word_emb.size())

            pack_sequence = pack_padded_sequence(word_emb, lengths=sorted_seq_length, batch_first=True)
            # print('pack_sequence',pack_sequence.size())
            sequence_output, (h_n, c_n) = self.lstm(pack_sequence)
            # print('sentence_output',sequence_output.size())
            sequence_output, out_len = pad_packed_sequence(sequence_output, total_length=input_ids.size(1),batch_first=True)
            _, unperm_idx = perm_idx.sort()
            # print(sequence_output.size())
#             print(out_len.size())
            sequence_output = sequence_output[unperm_idx, :]

            # batch_size = word_emb.size(0)

        # (sequence_output, (h_n, c_n)) = self.lstm(word_emb)
            pooled_output = torch.cat([h_n[-1], h_n[-2]], -1).to(device)
            # print(pooled_output.size())
#             print(sequence_output.size())
#             exit()

       
	
	
	
        elif self.trans is not None:

            mask = attention_mask
            out = self.embed(input_ids)
            out = self.position(out)
            sequence_output = self.trans(out, mask.float())
            # print(sequence_output.size())
            pool_output = torch.sum(sequence_output,1, out=None)
            # print(pool_output.size())
            # exit()
            # first_token_tensor = sequence_output[:, 0]
            pooled_output = self.dense_tr(pool_output)
            pooled_output = self.activation_tr(pool_output)

            # print(sequence_output.size())


        elif self.bert is not None:
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                       output_all_encoded_layers=False)
        elif self.zen is not None:
            sequence_output, pooled_output = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                                      ngram_position_matrix=ngram_position_matrix,
                                                      token_type_ids=token_type_ids, attention_mask=attention_mask,
                                                      output_all_encoded_layers=False)
        else:
            raise ValueError()

        if self.classifier_cls is not None:
            cls_logits = self.classifier_cls(pooled_output)
            probility = F.softmax(cls_logits, -1)
            pre_cls = torch.argmax(F.log_softmax(cls_logits, -1), -1)
            loss_ = CrossEntropyLoss(ignore_index=0)
            cls_loss = loss_(cls_logits.view(-1, self.num_labels_cls), labels_cls.view(-1))
            # print(probility.cpu().detach().numpy().tolist())
            #            exit()

            index_ = pre_cls.cpu().numpy().tolist()
            # print('now classifier!!')
        # if self.switch_attention and not self.switch:

        if self.switch == 'hard_switch':

            # print('now switch!!')

            new_label_value_matrix = np.zeros(
                (label_value_matrix.size(0), label_value_matrix.size(2), label_value_matrix.size(3)), dtype=np.int)
            new_word_seq = np.zeros((word_seq.size(0), word_seq.size(-1)), dtype=np.int)
            new_word_mask = np.zeros((word_mask.size(0), word_mask.size(2), word_mask.size(3)), dtype=np.float)
            label_value_matrix = label_value_matrix.cpu().numpy()
            word_seq = word_seq.cpu().numpy()
            word_mask = word_mask.cpu().numpy()

            for i in range(len(index_)):
                '''
                ??????????????????????????????????????????
                '''
                new_label_value_matrix[i] = label_value_matrix[i][index_[i]]
                new_word_seq[i] = word_seq[i][index_[i]]
                new_word_mask[i] = word_mask[i][index_[i]]

            label_value_matrix = torch.tensor(new_label_value_matrix, dtype=torch.long)
            word_seq = torch.tensor(new_word_seq, dtype=torch.long)
            word_mask = torch.tensor(new_word_mask, dtype=torch.float)
            # ????????????????????????label_value_matrix???word_seq????????????tensor

            label_value_matrix = label_value_matrix.to(device)
            word_seq = word_seq.to(device)
            word_mask = word_mask.to(device)

            if self.kv_memory is not None:
                # print('now memory!!')
                o = self.kv_memory(word_seq, sequence_output, label_value_matrix, word_mask)

            if self.attention_mode == 'cat':
                sequence_output = torch.cat([sequence_output, o], -1)
            else:
                sequence_output = torch.add(o, sequence_output)



        elif self.switch == 'soft_switch':
            # print(word_seq.size(),word_mask.size(),label_value_matrix.size())

            new_label_value_matrix = torch.chunk(label_value_matrix, 4, dim=1)
            new_word_seq = torch.chunk(word_seq, 4, dim=1)
            new_word_mask = torch.chunk(word_mask, 4, dim=1)
            new_probility = torch.chunk(probility, 4, dim=1)
            o = np.zeros((sequence_output.size(0), sequence_output.size(1), sequence_output.size(2)), dtype=np.float)
            o = torch.tensor(o, dtype=torch.float32)
            o = o.to(device)
            # print('o',o.size())
            for i in range(len(self.dataset_map)):
                if self.kv_memory is not None:
                    # print('now memory!!')

                    word_seq = new_word_seq[i].squeeze(dim=1)
                    label_value_matrix = new_label_value_matrix[i].squeeze(dim=1)
                    word_mask = new_word_mask[i].squeeze(dim=1)

                    # print(word_seq.size(),word_mask.size(),label_value_matrix.size())
                    o1 = self.kv_memory(word_seq, sequence_output, label_value_matrix, word_mask)
                    # print('o1',o1.size())

                    probility = new_probility[i].unsqueeze(dim=-1)  # ????????????

                    probility = probility.expand_as(o1)
                    # print('probility',probility)
                    # exit()
                    o1 = probility * o1

                    o = torch.add(o, o1)

                    # print('o',o.size())

            # exit()
            if self.attention_mode == 'cat':
                sequence_output = torch.cat([sequence_output, o], -1)
            else:
                sequence_output = torch.add(o, sequence_output)
        else:
            sequence_output = sequence_output

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        if self.crf is not None:
            # crf = CRF(tagset_size=number_of_labels+1, gpu=True)
            # print('now crf!!')
            total_loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            scores, tag_seq = self.crf._viterbi_decode(logits, attention_mask)
            # Only keep active parts of the loss
        else:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

        if self.classifier_cls is not None:

            total_loss = total_loss * 0.7 + cls_loss * 0.3

            return total_loss, tag_seq, index_

        else:
            # if args.do_train:

            index_ = [0] * self.train_batch_size

            return total_loss, tag_seq, index_

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['max_ngram_length'] = args.max_ngram_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_lstm'] = args.use_lstm
        hyper_parameters['use_trans'] = args.use_trans
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_memory'] = args.use_memory
        hyper_parameters['decoder'] = args.decoder
        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model, args):
        spec = spec.copy()
        res = cls(args=args, **spec)
        res.load_state_dict(model)
        return res

    def load_data(self, data_path, do_predict=False):

        if not do_predict:
            flag = data_path[data_path.rfind('/') + 1: data_path.rfind('.')]
            lines = readfile(data_path, self.dataset_map)
        else:
            flag = 'predict'
            lines = readsentence(data_path)

        data = []
        for sentence, label, label_cls in lines:
            if self.kv_memory is not None:

                word_list = {}  # ?????????ngram?????????ngram,????????????ngram??????
                matching_position = {}

                for key in self.dict2id.keys():
                    word_list[key], matching_position[key] = [], []
                    for i in range(len(sentence)):
                        for j in range(self.max_ngram_length):
                            if i + j > len(sentence):
                                break
                            word = ''.join(sentence[i: i + j + 1])
                            if word in self.dict2id[key]:
                                try:
                                    index = word_list[key].index(word)
                                except ValueError:
                                    word_list[key].append(word)
                                    index = len(word_list[key]) - 1
                                word_len = len(word)
                                for k in range(
                                        j + 1):  # ??????ngram,i+k????????????????????????????????????index?????????word_list????????????ID???l?????????n-gram ????????????ngram????????????????????????????????????
                                    if word_len == 1:
                                        l = 'S'
                                    elif k == 0:
                                        l = 'B'
                                    elif k == j:
                                        l = 'E'
                                    else:
                                        l = 'I'
                                    matching_position[key].append(
                                        (i + k, index, l))  # ((0,index,S),(0,index,B),(1,index,E))
            else:
                word_list = None
                matching_position = None
            # print(sentence,word_list['MSR'],matching_position['MSR'])
            #           print(sentence,word_list['SGW'],matching_position['SGW'])
            #           print(len(word_list),len(matching_position))
            #           exit()
            data.append((sentence, label, label_cls, word_list, matching_position))

        examples = []
        for i, (sentence, label, label_cls, word_list, matching_position) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = {}
            if word_list is not None:
                for key in word_list.keys():
                    word[key] = ' '.join(word_list[key])  # ?????????????????????????????????????????????
            else:
                word = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                                         label=label, label_cls=label_cls, word=word, matrix=matching_position))
        return examples

    def convert_examples_to_features(self, examples, ):

        max_seq_length = min(int(max([len(e.text_a.split(' ')) for e in examples]) * 1.1 + 2), self.max_seq_length)

        if self.kv_memory is not None:
            max_word_size = 0
            for key in self.dataset_map.keys():
                max_word_size_ = max(min(max([len(e.word[key].split(' ')) for e in examples]), self.max_ngram_size), 1)
                if max_word_size_ > max_word_size:
                    max_word_size = max_word_size_

        features = []

        tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            label_cls_id = example.label_cls
            tokens = []
            labels = []
            valid = []  ####
            label_mask = []
            if len(textlist) != len(labellist):
                print(textlist, labellist)
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):  # ???????????????????????????valid???1???????????????0?????????tokenize????????????????????????????????????????????????
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)  # ???????????????????????????????????????1
            label_mask.insert(0, 1)  # ?????????????????????????????????1
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.labelmap[labels[i]])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length
            # ???????????????????????????????????????
            # print(self.dataset_map)
            if self.kv_memory is not None:
                # print(len(self.dataset_map))
                word_ids = [[]] * len(self.dataset_map)
                matching_matrix = [[]] * len(self.dataset_map)
                # print(len(self.dataset_map),word_ids,matching_matrix )
                for key in self.dataset_map.keys():
                    word_ids_ = []
                    matching_matrix_ = np.zeros((max_seq_length, max_word_size), dtype=np.int)

                    # print(key, max_word_size, self.dataset_map[key], word_ids[self.dataset_map[key]])

                    # print(len(word_ids_))
                    wordlist = example.word[key]
                    wordlist = wordlist.split(' ') if len(wordlist) > 0 else []
                    matching_position = example.matrix[key]

                    if len(wordlist) > max_word_size:
                        wordlist = wordlist[:max_word_size]
                    for word in wordlist:
                        try:
                            word_ids_.append(self.big_word2id[
                                                 word])  # ??????????????????????????????????????????????????????????????????????????????????????????????????????word_id????????????,???????????????????????????????????????????????????????????????<PAD>
                        except KeyError:
                            print(word)
                            print(wordlist)
                            print(textlist)
                            raise KeyError()
                    # print(word_ids_)
                    while len(word_ids_) < max_word_size:
                        word_ids_.append(0)

                    # print('*******')
                    for position in matching_position:
                        char_p = position[0] + 1  # ??????????????????
                        word_p = position[1]  # ??????word_list ????????????
                        if char_p > max_seq_length - 2 or word_p > max_word_size - 1:
                            continue
                        else:
                            boundrymap = {'B':0,'I':1,'E':2,'S':3}
                            matching_matrix_[char_p][word_p] = boundrymap[position[2]]  # ????????????
                    matching_matrix[self.dataset_map[key]] = matching_matrix_
                    word_ids[self.dataset_map[key]] = word_ids_
                    # print(matching_matrix)
                    # if len(word_ids_) != max_word_size:
                    #                       print(len(word_ids_),max_word_size)
                    #                       print(word_ids_)
                    #                       exit()
                    assert len(word_ids_) == max_word_size

            else:
                word_ids = None
                matching_matrix = None

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])

                # random.shuffle(ngram_matches)
                ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

                max_ngram_in_seq_proportion = math.ceil(
                    (len(tokens) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                  dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              label_cls_id=label_cls_id,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              matching_matrix=matching_matrix,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_label_cls_id = torch.tensor([f.label_cls_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        label_cls_id = all_label_cls_id.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        if self.hpara['use_memory']:
            all_word_ids = torch.tensor([f.word_ids for f in feature], dtype=torch.long)
            all_matching_matrix = torch.tensor([f.matching_matrix for f in feature],
                                               dtype=torch.long)  # torch.long???,?????????tensor??????index,tensor??????????????????????????????????????????tensor???????????????
            all_word_mask = torch.tensor([f.matching_matrix for f in feature], dtype=torch.float)

            word_ids = all_word_ids.to(device)
            matching_matrix = all_matching_matrix.to(device)
            word_mask = all_word_mask.to(device)
        else:
            word_ids = None
            matching_matrix = None
            word_mask = None
        if self.hpara['use_zen']:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return input_ids, input_mask, l_mask, label_ids, label_cls_id, matching_matrix, ngram_ids, ngram_positions, segment_ids, valid_ids, word_ids, word_mask


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, label_cls=None, word=None, matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.label_cls = label_cls
        self.word = word
        self.matrix = matrix


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_cls_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_cls_id = label_cls_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readfile(filename, dataset_map):
    f = open(filename)
    data = []
    sentence = []
    label = []
    label_cls = 0

    text = f.read().split('\n\n')
    for sen in text:
        sen = sen.strip().split('\n')
        firstline = True
        for i in range(len(sen)):
            line = sen[i]
            if line.strip() == '':
                continue
            splits = line.split('\t')
            if i == 0:
                if splits[0] in dataset_map.keys():

                    label_cls = dataset_map[splits[0]]
                    continue
                else:
                    print('please check this sentence:', sen)
                    continue
            if i == len(sen) - 1:
                continue

            splits = line.split('\t')

            char = splits[0].strip()
            l = splits[1].strip()
            sentence.append(char)
            label.append(l)
        if len(sentence) == 0 or len(label) == 0:
            continue
        # print(len(sentence),len(label))
        assert len(sentence) == len(label)

        data.append((sentence, label, label_cls))
        sentence = []
        label = []
        sentence = []
        label_cls = 0

    return data


def readsentence(filename):
    data = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list = ['S' for _ in range(len(line))]
            data.append((line, label_list))
    return data

