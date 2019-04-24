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

class TermNumAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TermNumAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.termNumWeight = nn.Parameter(torch.Tensor(np.random.randn(hidden_dim)))
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, hidden_states):
        # after this, we have (batch, dim1) with a diff weight per each cell
        if hidden_states.size(0) == 0:
            return torch.zeros(self.hiddenDim)
        attention_hidden = hidden_states * self.termNumWeight
        attention_score = self.attSeq(attention_hidden)
        attention_score = F.softmax(attention_score, dim=1) # sigmoid(attention_score)
        scored_x = hidden_states * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x

class SpanSequnce(nn.Module):
    def __init__(self, data):
        super(SpanSequnce, self).__init__()
        print('build span ranking model ...')
        print('use_char: ', data.use_char)
        if data.use_char:
            print('char feature extractor: ', data.char_feature_extractor)
        print('word feature extractor: ', data.word_feature_extractor)
        self.gpu = data.HP_gpu
        self.longSpan = data.longSpan
        self.shortSpan = data.shortSpan
        self.average_batch = data.average_batch_loss
        self.word_hidden_features = WordSequence(data)
        self.classNum = data.label_alphabet_size
        self.max_span = data.term_span
        self.termRatio = data.termratio
        self.termAttention = TermAttention(data.HP_hidden_dim)
        # self.termNumAtten = TermNumAttention(data.HP_hidden_dim)
        #Variable(torch.randn(data.HP_hidden_dim), requires_grad=True) #(data.HP_hidden_dim)
        self.spanEmb2Score = nn.Linear(data.HP_hidden_dim, 1)
        # self.seqTermNum = nn.Linear(data.HP_hidden_dim, 1)
        

    def get_pairs(self, candidate_starts, candidate_ends):
        ''''''
        spanPairs = []
        for canStarts, canEnds in zip(candidate_starts, candidate_ends):
            sentSpanPairs = []
            for wordStart, wordEnds in zip(canStarts, canEnds):
                tmp_spans = [(start, end+1) for start, end in zip(wordStart, wordEnds)]
                tmp_spans = list(set(tmp_spans))
                sentSpanPairs.extend(tmp_spans)
            spanPairs.append(sentSpanPairs)
        return spanPairs

    def get_candidate_span_pairs(self, seq_lengths):
        ''''''
        sents_lengths = [np.arange(seq_len) for seq_len in seq_lengths]
        candidate_starts = [sent.reshape(-1, 1).repeat(self.max_span, 1) for sent in sents_lengths]
        span_lengths = np.arange(self.max_span)
        candidate_ends = [copy(sent_itm + span_lengths) for sent_itm in candidate_starts]
        candidate_ends = [np.array(np.minimum(canEnd, sentLen-1)) for sentLen, canEnd in zip(seq_lengths, candidate_ends)]
        spanPairs = self.get_pairs(candidate_starts, candidate_ends)
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

    def get_term_class(self, BtermVecs):
        ''''''
        BTermClasses = []
        for senTerms in BtermVecs:
            tmpSTClass = [self.spanEmb2class(vecItem) for vecItem in senTerms]
            BTermClasses.append(tmpSTClass)
        return BTermClasses

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

    def getKtopSpans(self, scores, spans, seqLens):
        ''''''
        sortedSpans = [torch.Tensor(itm).sort() for itm in scores]
        kNum = (seqLens.float() * self.termRatio).floor().int().numpy()
        topSpans = []
        for (Tscores, Tindex), span, k in zip(sortedSpans, spans, kNum):
            topSpans.append([span[idx] for idx in Tindex[:k]])
        return topSpans

    def getAccPrec(self, topspans, goldens):
        ''''''
        rightSpan = 0
        wrongSpan = 0
        totalGold = 0
        totalPred = 0

        for pSpan, gSpan in zip(topspans, goldens):
            totalGold += len(gSpan)
            totalPred += len(pSpan)
            for itm in pSpan:
                if itm in gSpan:
                    rightSpan += 1
                else:
                    wrongSpan += 1
        recall = float(rightSpan) / totalGold
        precision = float(rightSpan) / totalPred

        return recall, precision

    def forward(self, word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, training=True):
        hidden_features = self.word_hidden_features(word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        golden_labels = [bat_[0] for bat_ in batch_label]

        spanPairs = self.get_candidate_span_pairs(seq_lengths=word_seq_lengths)
        spanHiddens = self.get_span_hiddens(hidden_features, spanPairs, word_seq_lengths)
        spanEmb = self.get_span_emb(spanHiddens)
        termScores = self.get_term_score(spanEmb)
        topSpans = self.getKtopSpans(termScores, spanPairs, word_seq_lengths)
        assert len(word_inputs) == len(golden_labels)
        golden_spans, golden_class, golden_term_num = self.reformat_labels(golden_labels)
        recall, precision = self.getAccPrec(topSpans, golden_spans)

        if training:
            gold_span_hidden = self.get_span_hiddens(hidden_features, golden_spans, word_seq_lengths)
            gold_span_emb = self.get_span_emb(gold_span_hidden)
            gold_span_score = self.get_term_score(gold_span_emb)
            neg_samples = []
            for AAA, BBB, CCC in zip(spanPairs, golden_spans, golden_term_num):
                if CCC !=0:
                    neg_samples.append(list(set(AAA) - set(BBB))[:CCC])
                else:
                    neg_samples.append(list(set(AAA) - set(BBB))[:CCC+1])
            neg_span_hidden = self.get_span_hiddens(hidden_features, neg_samples, word_seq_lengths)
            neg_span_emb = self.get_span_emb(neg_span_hidden)
            neg_span_score = self.get_term_score(neg_span_emb)
            ### 1
            golden_avg = []
            for itm in gold_span_score:
                tmplist = 1.0 - torch.Tensor(itm).mean()
                tmplist.requires_grad = True
                golden_avg.append(tmplist)
            golden_avg = torch.Tensor(golden_avg).mean()
            golden_avg.requires_grad = True

            negative_avg = []
            for itm in neg_span_score:
                tmplist = torch.Tensor(itm).mean()
                tmplist.requires_grad = True
                negative_avg.append(tmplist)
            negative_avg = torch.Tensor(negative_avg).mean()
            negative_avg.requires_grad = True

            ec_loss = golden_avg + negative_avg
        final_loss = 100 * (1-precision) * ec_loss if training else 1-precision
        print(precision, ec_loss)

        return final_loss

if __name__ == '__main__':
    
    data = Data()
    instances = data.all_instances
    batched = instances[:100]
    span_seq = SpanSequnce(data)
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