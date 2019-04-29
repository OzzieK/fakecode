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


class TermAttention(nn.Module): # Attention on span
    def __init__(self, hidden_dim):
        super(TermAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, query, hidden_states):
        # after this, we have (batch, dim1) with a diff weight per each cell
        if hidden_states.size(0) == 0:
            return torch.zeros(self.hiddenDim)
        query = self.linear_in(query)
        attention_score = torch.matmul(hidden_states, query)
        attention_score = F.softmax(attention_score, dim=0).view(hidden_states.size(0), 1)
        scored_x = hidden_states * attention_score
        condensed_x = torch.sum(scored_x, dim=0)
        # condensed_x = self.tanh(condensed_x)
        return condensed_x #, attention_score


class SpanRanking(nn.Module):
    def __init__(self, data):
        super(SpanRanking, self).__init__()
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
        self.termWeight = nn.Parameter(torch.Tensor(np.random.randn(data.HP_hidden_dim)))
        self.termAttention = TermAttention(data.HP_hidden_dim)
        if self.pos_as_feature:
            self.pos_embedding_dim = data.pos_emb_dim
            self.pos_embedding = nn.Embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)
            self.pos_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)))

        self.spanEmb2Score = nn.Linear(data.HP_hidden_dim, 1)
        self.loss_fun = nn.SoftMarginLoss()


    def get_candidate_span_pairs(self, seq_lengths):
        '''
        :param seq_lengths: the sequence length of sentences
        :return: the candidate span pairs [max span length: self.max_span] of each sentences
        sentence length: 4; self.max_span = 2
        spanPairs: [0,1], [0,2], [1,2], [1,3], [2,3], [2,4], [3,4]
        '''
        sents_lengths = [np.arange(seq_len) for seq_len in seq_lengths]
        candidate_starts = [sent.reshape(-1, 1).repeat(self.max_span, 1) for sent in sents_lengths]
        span_lengths = np.arange(self.max_span)
        candidate_ends = [copy(sent_itm + span_lengths) for sent_itm in candidate_starts]
        candidate_ends = [np.array(np.minimum(canEnd, sentLen-1)) for sentLen, canEnd in zip(seq_lengths, candidate_ends)]
        spanPairs = []
        for canStarts, canEnds in zip(candidate_starts, candidate_ends):
            sentSpanPairs = []
            for wordStarts, wordEnds in zip(canStarts, canEnds):
                tmp_spans = [(start, end+1) for start, end in zip(wordStarts, wordEnds)]
                tmp_spans = list(set(tmp_spans))
                sentSpanPairs.extend(tmp_spans)
            spanPairs.append(sentSpanPairs)
        return spanPairs

    def get_span_hiddens(self, wordReps, hidden_states, postags, span_pairs, seq_lengths):
        '''
        <pydoc>
        :param hidden_states: sequence token hidden states
        :param span_pairs: the candidate span pairs of each sentence
        :param seq_lengths: the length of each sentence
        :return: flat_Hiddens: the hidden slice of each span pair, [span_num, n*hidden_dim]\
                 flat_spanRep: the vector representation of each span, [span_num, hidden_dim]\
                 flat_spanSErep: the vector that contains just the begin and end word of a span, [span_num, 2*hidden_dim]\
                 flat_spanPairs: the flattened span pairs, [span_num]\
                 flat_spanLens: the length of each span pair, [span_num]\
                 flat_sentIds: corresponsing sentence IDs of the span pair, [span_num]\
                 sent_num: total sentence number, int [or batch_size]
        '''
        assert len(hidden_states) == len(span_pairs)
        flat_Hiddens = []
        flat_wordreps = []
        flat_posembs = []
        flat_spanPairs = []
        flat_sentIds = []
        flat_spanSErep = []
        flat_spanLens = []
        sent_num = len(span_pairs)
        sentence_slice = [[] for _ in range(sent_num)]
        spanPairID = 0

        for sent_id, (wordrep, seHiddens, postag, sentPair, seqLen) in enumerate(zip(wordReps, hidden_states, postags, span_pairs, seq_lengths)):
            sentHid = seHiddens[:seqLen]
            wordrep_ = wordrep[:seqLen]
            pos_embs = postag[:seqLen]
            for pairs in sentPair:
                flat_Hiddens.append(sentHid[pairs[0]: pairs[1]])
                flat_wordreps.append(wordrep_[pairs[0]:pairs[1]])
                flat_posembs.append(pos_embs[pairs[0]: pairs[1]])
                flat_spanSErep.append(torch.cat((sentHid[pairs[0]], sentHid[pairs[1]-1]), dim=0))
                flat_sentIds.append(sent_id)
                flat_spanPairs.append(pairs)
                flat_spanLens.append(pairs[1]-pairs[0])
                sentence_slice[sent_id].append(spanPairID)
                spanPairID += 1
        # get span representation
        flat_spanRep = [self.termAttention(self.termWeight, wordSpan).unsqueeze(0) for wordSpan in flat_Hiddens]
        return flat_Hiddens, flat_spanRep, flat_spanSErep, flat_posembs, flat_spanPairs, flat_spanLens, flat_sentIds, sentence_slice, sent_num

    def get_sentSliceResult(self, sentence_slice, results):
        sentSliceRes = []
        for itm in sentence_slice:
            start = itm[0]
            end = itm[-1]
            sentSliceRes.append(results[start:end+1])
        return sentSliceRes

    def getGoldIndex(self, golden_spans, sentSpanCandi, sentSlice):
        ''''''
        golden_IDs = []
        goldenSentIDs = [[] for _ in range(len(sentSpanCandi))]
        OOVSpan = 0
        for sentID, (gold_, candi_, canIDs) in enumerate(zip(golden_spans, sentSpanCandi, sentSlice)):
            for gold in gold_:
                try:
                    tmp_ID = canIDs[candi_.index(gold)]
                except ValueError:
                    OOVSpan += 1
                    continue
                golden_IDs.append(tmp_ID)
                goldenSentIDs[sentID].append(tmp_ID)
        return golden_IDs, goldenSentIDs, OOVSpan

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

    def forward(self, word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, training=True):

        hidden_features, word_rep = self.word_hidden_features(word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        pos_embs = self.pos_embedding(pos_inputs)
        golden_labels = [bat_[0] for bat_ in batch_label]
        assert len(word_inputs) == len(golden_labels)
        spanPairs = self.get_candidate_span_pairs(seq_lengths=word_seq_lengths)
        golden_spans, golden_class, golden_term_num = self.reformat_labels(golden_labels)
        flat_Hiddens, flat_spanRep, flat_spanSErep, flat_posembs, flat_spanPairs, flat_spanLens, flat_sentIds, sentence_slice, sent_num \
            = self.get_span_hiddens(word_rep, hidden_features, pos_embs, spanPairs, word_seq_lengths)
        termScores = self.spanEmb2Score(torch.cat(flat_spanRep, dim=0))

        sentence_span_score = self.get_sentSliceResult(sentence_slice, termScores)
        sentence_span_candidate = self.get_sentSliceResult(sentence_slice, flat_spanPairs)
        flat_golden_indexes, sent_gold_indexes, oovNum = self.getGoldIndex(golden_spans, sentence_span_candidate, sentence_slice)
        flat_wrong_indexes = list(set(range(len(flat_spanPairs))) - set(flat_golden_indexes))

        sorted_score, reindex = termScores.sort(0, descending=True)
        total_words = word_seq_lengths.sum().float()
        K = (total_words * 0.3).floor().int()
        predicted_index = reindex[:K]
        predicted_right = 0
        for pix in predicted_index:
            if pix in flat_golden_indexes:
                predicted_right += 1
        precision = float(predicted_right) / float(K)

        gold_score = termScores[flat_golden_indexes]#.mean()
        gold_target = torch.ones(len(flat_golden_indexes), dtype=torch.float)
        wrong_score = termScores[flat_wrong_indexes]#.mean()
        wrong_target = torch.zeros(len(flat_wrong_indexes), dtype=torch.float)
        loss = self.loss_fun(torch.cat((gold_score, wrong_score), dim=0).view(-1), torch.cat((gold_target, wrong_target), dim=0).view(-1))
        print(loss, precision)
        return loss, precision

if __name__ == '__main__':
    
    data = Data()
    instances = data.all_instances
    batched = instances[:100]
    span_seq = SpanRanking(data)
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