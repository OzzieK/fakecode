# -*- coding: utf-8 -*-
# ---------------------------------
#                      -----.
#                   _.'__    `.
#               .--(#)(##)---/#\
#             .' @          /###\
#             :         ,   #####
#              `-..__.-' _.-\###/
#                    `;_:    `"'
#                  .'"""""`.
#                 /,  神就 ,\
#                #    是我!  \\
#                `-._______.-'
#                ___`. | .'___
#               (______|______)
# ----------------------------------

import json, sys, os
from copy import copy, deepcopy
from utils.alphabet import Alphabet
from utils.functions import *
import pickle as pkl


class Data(object):
    def __init__(self):
        self.data_dir = './../data/gene_term_format_by_sentence.json'
        self.data_ratio = (0.9, 0.05, 0.05)  # total 2000
        self.save_dir = './data/'

        self.pos_as_feature = True
        self.use_char = True

        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.ptag_alphabet = Alphabet('tag')
        self.label_alphabet = Alphabet('label', label=True)
        self.seqlabel_alphabet = Alphabet('span_label', label=True)

        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.ptag_alphabet_size = 0
        self.label_alphabet_size = 0
        self.seqlabel_alphabet_size = 0

        self.max_sentence_length = 500

        self.term_truples = []

        self.sent_texts = []
        self.chars = []
        self.lengths = []
        self.ptags = []
        self.seq_labels = []

        self.word_ids_sent = []
        self.char_id_sent = []
        self.tag_ids_sent = []
        self.label_ids_sent = []
        self.seq_labels_ids = []

        self.longSpan = True
        self.shortSpan = True
        self.termratio = 0.3

        self.word_feature_extractor = "LSTM"  ## "LSTM"/"CNN"/"GRU"/
        self.char_feature_extractor = "CNN"  ## "LSTM"/"CNN"/"GRU"/None

        # training
        self.optimizer = 'Adam'  # "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.training = True
        self.average_batch_loss = True
        self.evaluate_every = 10 # evaluate every n batches

        # Embeddings
        self.word_emb_dir = None
        self.char_emb_dir = None
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.pos_emb_dim = 20
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        # HP
        self.term_span = 5
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_cnn_layer = 2
        self.HP_batch_size = 100
        self.HP_epoch = 100
        self.HP_lr = 0.01
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_l2 = 1e-8
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 2
        self.HP_bilstm = True
        self.HP_gpu = False
        self.HP_term_span = 6
        self.HP_momentum = 0

        # init data
        self.build_vocabs()
        self.all_instances = self.load_data()
        self.load_pretrain_emb()

    def build_vocabs(self):
        ''''''
        with open(self.data_dir, 'r') as filin:
            filelines = filin.readlines()

        for lin_id, lin_cnt in enumerate(filelines):
            lin_cnt = lin_cnt.strip()
            line = json.loads(lin_cnt)
            words = line['words']
            tags = line['tags']
            terms = line['terms']
            for word in words:
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)
            for tag in tags:
                self.ptag_alphabet.add(tag)
            self.sent_texts.append(words)
            self.ptags.append(tags)
            assert len(words) == len(tags)
            self.lengths.append(len(words))
            seq_label, termple = self.reformat_label(words, terms)
            self.seq_labels.append(seq_label)

            if len(terms) > 0:
                tmp_terms = []
                for itm in termple:
                    tmp_terms.append([itm[0], itm[1], itm[2]])
                    self.label_alphabet.add(itm[2])
                self.term_truples.append(tmp_terms)
            else:
                self.term_truples.append([[-1, -1, 'None']])
                self.label_alphabet.add('None')

        for ter in self.seq_labels:
            for ater in ter:
                for ate in ater:
                    self.seqlabel_alphabet.add(ate)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.ptag_alphabet_size = self.ptag_alphabet.size()
        self.seqlabel_alphabet_size = self.seqlabel_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        self.close_alphabet()

    def load_pretrain_emb(self):
        ''''''
        if self.word_emb_dir:
            print('Loading pretrained Word Embedding from %s'.format(self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)
        if self.char_emb_dir:
            print('Loading pretrained Char Embedding from %s'.format(self.char_emb_dir))
            self.pretrain_word_embedding, self.char_emb_dim = build_pretrain_embedding(
                (self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb))

    def load_data(self):
        ''''''
        all_instances = []
        assert len(self.sent_texts) == len(self.term_truples) == len(self.ptags)
        for sent_text, ptag, term_truple, seqlabel in zip(self.sent_texts, self.ptags, self.term_truples, self.seq_labels):
            self.word_ids_sent.append(self.word_alphabet.get_indexs(sent_text))
            sent_char = []
            sent_char_ids = []
            for word in sent_text:
                char_list = list(word)
                sent_char.append(char_list)
                char_ids = self.char_alphabet.get_indexs(char_list)
                sent_char_ids.append(char_ids)
            seqLabel_ids = [self.seqlabel_alphabet.get_indexs(seqlab) for seqlab in seqlabel]
            self.seq_labels_ids.append(seqLabel_ids)
            self.chars.append(sent_char)
            self.char_id_sent.append(sent_char_ids)
            self.tag_ids_sent.append(self.ptag_alphabet.get_indexs(ptag))
            term_truple = [[term[0], term[1], self.label_alphabet.get_index(term[2])] for term in term_truple]
            self.label_ids_sent.append(term_truple)
            all_instances.append([self.word_ids_sent[-1], sent_char_ids, self.tag_ids_sent[-1], [term_truple, seqLabel_ids]])
        return all_instances

    def reformat_label(self, words, terms):
        label = [[] for i in range(len(words))]
        termtruple = []
        if len(terms) > 0:
            for term in terms:
                beg = term[0]
                end = term[1]
                lab_ = term[2]
                termtruple.append((beg, end, lab_))
                if beg == end:
                    label[beg].append('S')
                    continue
                label[beg].append('B')
                label[end].append('E')
                if end - beg > 1:
                    for itm in range(beg+1, end):
                        label[itm].append('I')
        for slab in label:
            if slab == []:
                slab.append('O')
        label = [list(set(lab)) for lab in label]
        return label, termtruple

    def restore(self, data_file):
        print('Loading data from %s'%data_file)
        with open(data_file, 'rb') as filin:
            obj_dict = pkl.load(filin)
            self.__dict__.update(obj_dict)

    def save(self, save_file):
        print('Saving data to %s'%save_file)
        with open(save_file, 'wb') as filout:
            pkl.dump(self.__dict__, filout, 2)

    def close_alphabet(self):
        self.word_alphabet.close()
        self.ptag_alphabet.close()
        self.label_alphabet.close()
        self.seqlabel_alphabet.close()
        self.char_alphabet.close()
        return





if __name__ == '__main__':

    data = Data()
    data.load_data()

    print('data_ratio' in data.__dict__)
    print(data.__str__())
    print(data.ptag_alphabet.__dict__)
    print(data.word_alphabet.__dict__)
    print(data.char_alphabet.__dict__)
    print(data.label_ids_sent[0])
    print(data.word_ids_sent[0])
    print(data.tag_ids_sent[0])
    print(data.ptag_alphabet.get_instances(data.tag_ids_sent[0]))

