# -*- coding: utf-8 -*-
# @Author: Yuze Gao
# @Date:   2019-04-10 18:17:45
# -----------------------------------------------------------------
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
# -----------------------------------------------------------------


from __future__ import print_function
from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.wordfeat.WordSeq import WordSequence, reformat_input_data
from utils.data import Data
from copy import copy
import torch
import torch.optim as optim
from torch.autograd import Variable
from random import shuffle
from utils.functions import masked_log_softmax, MaskedQueryAttention

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.attSeq = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1))

    def forward(self, hidden_states):
        # (B, L, H) -> (B , L, 1)
        energy = self.attSeq(hidden_states)
        weights = F.softmax(energy.squeeze(-1), dim=0)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (hidden_states * weights.unsqueeze(-1)).sum(dim=0)
        return outputs, weights


class TermAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TermAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.termWeight = nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)))
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, hidden_states):
        # after this, we have (batch, dim1) with a diff weight per each cell
        if hidden_states.size(0) == 0:
            return torch.zeros(self.hiddenDim)
        attention_hidden = hidden_states * self.termWeight
        attention_score = self.attSeq(attention_hidden)
        attention_score = F.softmax(attention_score, dim=0).view(hidden_states.size(0), 1)
        scored_x = hidden_states * attention_score
        condensed_x = torch.sum(scored_x, dim=0)
        return condensed_x #, attention_score

class SeqAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SeqAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.SeqWeight = nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)))
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, hidden_states, mask):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_hidden = hidden_states * self.SeqWeight
        attention_score = self.attSeq(attention_hidden)
        mask = mask.unsqueeze(-1).float()
        attention_score = attention_score * mask
        attention_score = masked_log_softmax(attention_score, mask, dim=1).view(hidden_states.size(0), hidden_states.size(1), 1)
        scored_x = hidden_states * attention_score
        condensed_x = torch.tanh(scored_x)
        return condensed_x #, attention_score

class SpanClassfy(nn.Module):
    def __init__(self, data):
        super(SpanClassfy, self).__init__()
        print('build span ranking model ...')
        print('use_char: ', data.use_char)
        if data.use_char:
            print('char feature extractor: ', data.char_feature_extractor)
        print('word feature extractor: ', data.word_feature_extractor)
        self.gpu = data.HP_gpu
        self.longSpan = data.longSpan
        self.shortSpan = data.shortSpan
        self.hiddenDim = data.HP_hidden_dim
        self.average_batch = data.average_batch_loss
        self.word_hidden_features = WordSequence(data)
        self.classNum = data.label_alphabet_size
        self.max_span = data.term_span
        self.termRatio = data.termratio
        self.termAttention = TermAttention(data.HP_hidden_dim)
        self.SeqAttention = SeqAttention(data.HP_hidden_dim)
        self.asaquery = nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)))
        self.asa = MaskedQueryAttention(data.HP_hidden_dim)
        self.spanEmb2Score = nn.Linear(data.HP_hidden_dim, 2)
        self.loss_fun = nn.CrossEntropyLoss()
        self.pos_as_feature = data.pos_as_feature
        if self.pos_as_feature:
            self.pos_embedding_dim = data.pos_emb_dim
            self.pos_embedding = nn.Embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)

    def get_candidate_span_pairs(self, seq_lengths):
        ''''''
        sents_lengths = [np.arange(seq_len) for seq_len in seq_lengths]
        candidate_starts = [sent.reshape(-1, 1).repeat(self.max_span, 1) for sent in sents_lengths]
        span_lengths = np.arange(self.max_span)
        candidate_ends = [copy(sent_itm + span_lengths) for sent_itm in candidate_starts]
        candidate_ends = [np.array(np.minimum(canEnd, sentLen-1)) for sentLen, canEnd in zip(seq_lengths, candidate_ends)]
        spanPairs = []
        for canStarts, canEnds in zip(candidate_starts, candidate_ends):
            sentSpanPairs = []
            for wordStart, wordEnds in zip(canStarts, canEnds):
                tmp_spans = [(start, end+1) for start, end in zip(wordStart, wordEnds)]
                tmp_spans = list(set(tmp_spans))
                sentSpanPairs.extend(tmp_spans)
            spanPairs.append(sentSpanPairs)
        return spanPairs

    def get_span_hiddens(self, hidden_states, span_pairs, seq_lengths):
        assert len(hidden_states) == len(span_pairs)
        spanHiddens = []
        for seHiddens, sentPair, seqLen in zip(hidden_states, span_pairs, seq_lengths):
            sentUnit = []
            sentHid = seHiddens[:seqLen]
            for pairs in sentPair:
                sentUnit.append(sentHid[pairs[0]: pairs[1]])
            spanHiddens.append(sentUnit)
        return spanHiddens

    def get_span_emb(self, spanHiddens):
        ''''''
        SpamEmb = []
        for sentence in spanHiddens:
            sentSpanEmb = [self.termAttention(wordSpan) for wordSpan in sentence]
            SpamEmb.append(sentSpanEmb)
        return SpamEmb

    def get_term_score(self, span_embs):
        ''''''
        termScore = []
        for sentence in span_embs:
            sentTerms = [self.spanEmb2Score(termemb) for termemb in sentence]
            termScore.append(sentTerms)
        return termScore

    def reformat_labels(self, golden_labels):
        ''''''
        golden_spans = []
        golden_class = []
        golden_term_num = []
        for sentSpan in golden_labels:
            golden_spans.append([(itm[0], itm[1]+1) for itm in sentSpan])
            golden_class.append([itm[2] for itm in sentSpan])
            sent_term_num = len(sentSpan) if sentSpan[0][1] != -1 else 0
            golden_term_num.append(sent_term_num)
        return golden_spans, golden_class, golden_term_num

    def flatted_list(self, batched_):
        ''''''
        tmp = [itm.unsqueeze(0) for sent in batched_ for itm in sent]
        return torch.cat(tmp, dim=0)


    def forward(self, word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, training=True):
        hidden_features, word_rep = self.word_hidden_features(word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        hidden_features = self.asa(self.asaquery, hidden_features, mask)
        # hidden_features = self.SeqAttention(hidden_features, mask)
        features = []
        features.append(hidden_features)
        if self.pos_as_feature:
            features.append(self.pos_embedding(pos_inputs))

        golden_labels = [bat_[0] for bat_ in batch_label]
        spanPairs = self.get_candidate_span_pairs(seq_lengths=word_seq_lengths)
        golden_spans, golden_class, golden_term_num = self.reformat_labels(golden_labels)
        gold_span_hidden = self.get_span_hiddens(hidden_features, golden_spans, word_seq_lengths)
        gold_span_emb = self.get_span_emb(gold_span_hidden)
        gold_span_score = self.get_term_score(gold_span_emb)
        flatted_gold = self.flatted_list(gold_span_score)
        gold_predicted = torch.argmax(flatted_gold, dim=1)
        gold_right = gold_predicted.eq(1).sum().float()  # tp
        false_nega = flatted_gold.size()[0] - gold_right # false negative
        flatted_gold_soft = F.softmax(flatted_gold, dim=1)
        gold_target = torch.ones(flatted_gold_soft.size()[0], dtype=torch.long)
        gold_loss = self.loss_fun(flatted_gold, gold_target)
        neg_samples = []
        for AAA, BBB, CCC in zip(spanPairs, golden_spans, golden_term_num):
            neg_samples.append(list(set(AAA) - set(BBB)))
        neg_span_hidden = self.get_span_hiddens(hidden_features, neg_samples, word_seq_lengths)
        neg_span_emb = self.get_span_emb(neg_span_hidden)
        neg_span_score = self.get_term_score(neg_span_emb)
        flatted_nega = self.flatted_list(neg_span_score)
        nega_predicted = torch.argmax(flatted_nega, dim=1)
        nega_right = nega_predicted.eq(0).sum().float() # true negative
        false_gold = flatted_nega.size()[0] - nega_right
        nega_target = torch.zeros(flatted_nega.size()[0], dtype=torch.long)
        nega_loss = self.loss_fun(flatted_nega, nega_target)
        flatted_nega_soft = F.softmax(flatted_nega, dim=1)
        accuracy = (gold_right+nega_right)/(gold_predicted.size(0)+nega_predicted.size()[0])
        # filtering


        print(nega_loss+gold_loss, gold_right/gold_predicted.size(0), nega_right/nega_predicted.size()[0], accuracy, gold_predicted.size()[0], nega_predicted.size()[0]-nega_right)

        return nega_loss+gold_loss

if __name__ == '__main__':
    
    data = Data()
    instances = data.all_instances
    batched = instances[:100]
    span_seq = SpanClassfy(data)
    optimizer = optim.Adam(span_seq.parameters(), lr=0.01, weight_decay=data.HP_l2)
    for epo in range(100):
        for idx in range(len(instances)//100):
            batched = instances[idx*100:(idx+1)*100]
            span_seq.train()
            word_seq_tensor, pos_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask = reformat_input_data(
                batched, use_gpu=True)
            reps = span_seq(word_seq_tensor, pos_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths,
                            char_seq_recover, labels, mask)
            reps.backward()
            optimizer.step()
            span_seq.zero_grad()
        epo += 1