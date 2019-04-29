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
from utils.functions import masked_log_softmax, MaskedQueryAttention, getElmo, random_embedding


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.attSeq = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1))

    def forward(self, hidden_states):
        energy = self.attSeq(hidden_states)
        weights = F.softmax(energy.squeeze(-1), dim=0)
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


class SpanClassfy(nn.Module):
    def __init__(self, data):
        super(SpanClassfy, self).__init__()
        print('build span classifying model ...')
        print('use_char: ', data.use_char)
        if data.use_char:
            print('char feature extractor: ', data.char_feature_extractor)
        print('word feature extractor: ', data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.hiddenDim = data.HP_hidden_dim
        self.average_batch = data.average_batch_loss
        # self.classNum = data.label_alphabet_size
        self.max_span = data.term_span
        self.termRatio = data.termratio
        self.pos_embedding_dim = data.pos_emb_dim
        self.useSpanLen = data.useSpanLen
        self.useElmo = data.use_elmo
        self.pos_as_feature = data.pos_as_feature
        # self.SeqAttention = SeqAttention(data.HP_hidden_dim)

        self.word_hidden_features = WordSequence(data)

        self.asaquery = nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)))
        self.queryAtten = MaskedQueryAttention(data.HP_hidden_dim)

        self.termAttention = TermAttention(data.HP_hidden_dim)
        self.posAttention = TermAttention(data.pos_emb_dim)

        self.loss_fun = nn.CrossEntropyLoss()
        self.elmoEmb = getElmo(layer=2, dropout=data.HP_dropout, out_dim=data.HP_hidden_dim)



        self.pos_embedding = nn.Embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)
        self.pos_embedding.weight.data.copy_(
            torch.from_numpy(random_embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)))
        self.spanLenemb = nn.Embedding(self.max_span + 1, data.spamEm_dim)
        self.spanLenemb.weight.data.copy_(torch.from_numpy(random_embedding(self.max_span + 1, data.spamEm_dim)))

        self.posSeq2Vec = nn.Linear(data.pos_emb_dim * self.max_span, data.pos_emb_dim)
        self.hidden2Vec = nn.Linear(data.HP_hidden_dim * self.max_span, data.HP_hidden_dim)
        self.elmo2Vec = nn.Linear(data.HP_hidden_dim * self.max_span, data.HP_hidden_dim)

        self.feature_dim = data.HP_hidden_dim * 4

        if self.pos_as_feature:
            self.feature_dim += data.pos_emb_dim * 4

        if self.useElmo:
            self.feature_dim += data.HP_hidden_dim * 4

        if self.useSpanLen:
            self.feature_dim += data.spamEm_dim
        self.spanEmb2Score = nn.Sequential(nn.Linear(self.feature_dim, self.hiddenDim), nn.ReLU(True), nn.Linear(self.hiddenDim, 2))

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

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
            for wordStarts, wordEnds in zip(canStarts, canEnds):
                tmp_spans = [(start, end+1) for start, end in zip(wordStarts, wordEnds)]
                tmp_spans = list(set(tmp_spans))
                sentSpanPairs.extend(tmp_spans)
            spanPairs.append(sentSpanPairs)
        return spanPairs

    def get_span_features(self, hidden_states, postags, elmof, span_pairs, seq_lengths):
        '''
        <pydoc>
        :param hidden_states: sequence token hidden states
        :param postags: the pos-tag info
        :param elmof: the elmo vector
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
        assert hidden_states.size(0) == len(span_pairs)

        # from hidden
        flat_HiddenSeq = []
        flat_Hiddennode = []
        flat_spanSErep = []

        # from pos-tags
        flat_posatt = []
        flat_posembs = []
        flat_posSERep = []

        # from elmo
        flat_elmoSeq = []
        flat_elmonode = []
        flat_elmospanSE = []

        flat_spanPairs = []
        flat_spanLens = []
        flat_sentIds = []

        sent_num = len(span_pairs)
        sentence_slice = [[] for _ in range(sent_num)]
        spanPairID = 0
        pad_pos = torch.zeros(self.pos_embedding_dim)
        pad_hidden = torch.zeros(self.hiddenDim)
        for sent_id, (elmoSent, seHiddens, postag, sentPair, seqLen) in enumerate(zip(elmof, hidden_states, postags, span_pairs, seq_lengths)):
            sentHid = seHiddens[:seqLen]
            elmoS = elmoSent[:seqLen]
            pos_embs = postag[:seqLen]
            for pairs in sentPair:
                flat_HiddenSeq.append(sentHid[pairs[0]: pairs[1]])
                flat_posatt.append(pos_embs[pairs[0]: pairs[1]])
                flat_elmoSeq.append(elmoS[pairs[0]: pairs[1]])
                if pairs[1] - pairs[0] == self.max_span:
                    flat_posembs.append(pos_embs[pairs[0]: pairs[1]].view(-1))
                    flat_elmonode.append(elmoS[pairs[0]: pairs[1]].view(-1))
                    flat_Hiddennode.append(sentHid[pairs[0]: pairs[1]].view(-1))
                else:
                    flat_posembs.append(torch.cat((pos_embs[pairs[0]: pairs[1]].view(-1), pad_pos.repeat(self.max_span + pairs[0] - pairs[1])), dim=-1))
                    flat_elmonode.append(torch.cat((elmoS[pairs[0]: pairs[1]].view(-1), pad_hidden.repeat(self.max_span + pairs[0] - pairs[1])), dim=-1))
                    flat_Hiddennode.append(torch.cat((sentHid[pairs[0]: pairs[1]].view(-1), pad_hidden.repeat(self.max_span + pairs[0] - pairs[1])), dim=-1))

                flat_posSERep.append(torch.cat((pos_embs[pairs[0]], pos_embs[pairs[1]-1]), dim=0).unsqueeze(0))
                flat_spanSErep.append(torch.cat((sentHid[pairs[0]], sentHid[pairs[1]-1]), dim=0).unsqueeze(0))
                flat_elmospanSE.append(torch.cat((elmoS[pairs[0]], elmoS[pairs[1]-1]), dim=0).unsqueeze(0))

                flat_sentIds.append(sent_id)
                flat_spanPairs.append(pairs)
                flat_spanLens.append(pairs[1]-pairs[0])
                sentence_slice[sent_id].append(spanPairID)

                spanPairID += 1

        # get span representation
        flat_spanRep = [self.termAttention(wordSpan).unsqueeze(0) for wordSpan in flat_HiddenSeq] # attention node
        flat_spanRep = torch.cat(flat_spanRep, dim=0)
        flat_elmospanAT = [self.termAttention(wordSpan).unsqueeze(0) for wordSpan in flat_elmoSeq] # attention node
        flat_elmospanAT = torch.cat(flat_elmospanAT, dim=0)
        flat_posatt = [self.posAttention(pospan).unsqueeze(0) for pospan in flat_posatt]
        flat_posatt = torch.cat(flat_posatt, dim=0)

        # function condense
        flat_Hiddennode = [self.hidden2Vec(hidden_seq).unsqueeze(0) for hidden_seq in flat_Hiddennode]
        flat_Hiddennode = torch.cat(flat_Hiddennode, dim=0)
        flat_posembs = [self.posSeq2Vec(pos_seq).unsqueeze(0) for pos_seq in flat_posembs]
        flat_posembs = torch.cat(flat_posembs, dim=0)
        flat_elmonode = [self.elmo2Vec(elmo_seq).unsqueeze(0) for elmo_seq in flat_elmonode]
        flat_elmonode = torch.cat(flat_elmonode, dim=0)

        # start and end word representation
        flat_spanSErep = torch.cat(flat_spanSErep, dim=0)
        flat_posSERep = torch.cat(flat_posSERep, dim=0)
        flat_elmospanSE = torch.cat(flat_elmospanSE, dim=0)

        return flat_spanRep, flat_Hiddennode, flat_spanSErep, flat_elmospanAT, flat_elmonode, flat_elmospanSE, \
               flat_posatt, flat_posembs, flat_posSERep, flat_spanPairs, flat_spanLens, flat_sentIds, sentence_slice, sent_num

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


    def forward(self, word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, sent_texts, training=True):
        hidden_features, word_rep = self.word_hidden_features(word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        hidden_features = self.queryAtten(self.asaquery, hidden_features, mask)
        elmo_features, elmo_mask = self.elmoEmb(sent_texts)
        # hidden_features = self.SeqAttention(hidden_features, mask)

        pos_embs = self.pos_embedding(pos_inputs)

        golden_labels = [bat_[0] for bat_ in batch_label]
        # get candidate start id and end id pair
        spanPairs = self.get_candidate_span_pairs(seq_lengths=word_seq_lengths)

        # get golden span info
        golden_spans, golden_class, golden_term_num = self.reformat_labels(golden_labels)

        flat_spanRep, flat_Hiddennode, flat_spanSErep, flat_elmospanAT, flat_elmonode, flat_elmospanSE, \
        flat_posatt, flat_posembs, flat_posSERep, flat_spanPairs, flat_spanLens, flat_sentIds, sentence_slice, sent_num \
            = self.get_span_features(hidden_features, pos_embs, elmo_features, spanPairs, word_seq_lengths)

        sentence_span_candidate = self.get_sentSliceResult(sentence_slice, flat_spanPairs)
        flat_golden_indexes, sent_gold_indexes, oovNum = self.getGoldIndex(golden_spans, sentence_span_candidate,
                                                                           sentence_slice)

        spanEmbs = [flat_spanRep, flat_Hiddennode, flat_spanSErep]

        if self.useElmo:
            spanEmbs.append(flat_elmospanAT)
            spanEmbs.append(flat_elmonode)
            spanEmbs.append(flat_elmospanSE)

        if self.pos_as_feature:
            spanEmbs.append(flat_posatt)
            spanEmbs.append(flat_posembs)
            spanEmbs.append(flat_posSERep)

        lenEmbeddings = self.spanLenemb(torch.Tensor(flat_spanLens).long())
        if self.useSpanLen:
            spanEmbs.append(lenEmbeddings)
        # print(flat_spanRep.size())
        # print(flat_Hiddennode.size())
        # print(flat_spanRep.size())
        # exit(0)
        spanEmbs = torch.cat(spanEmbs, dim=-1)

        flat_neg_indexes = list(set(range(len(flat_spanPairs))) - set(flat_golden_indexes))

        spanClasses = self.spanEmb2Score(spanEmbs)

        gold_span_classes = spanClasses[flat_golden_indexes]
        nega_span_classes = spanClasses[flat_neg_indexes]

        gold_targets = torch.ones((gold_span_classes.size(0)), dtype=torch.long)
        nega_targets = torch.zeros((nega_span_classes.size(0)), dtype=torch.long)

        # the negative samples are too much times of the golden, so we'd better calculate them separately
        gold_loss = self.loss_fun(gold_span_classes, gold_targets)
        nega_loss = self.loss_fun(nega_span_classes, nega_targets)

        loss = gold_loss + nega_loss

        PredResults = torch.argmax(spanClasses, dim=-1)
        postivePred = PredResults.eq(1).sum().float()
        wrghtPred = PredResults[flat_golden_indexes].sum().float()
        precision = wrghtPred / postivePred
        recall = wrghtPred / float(len(flat_golden_indexes))
        print(loss, postivePred, wrghtPred, precision, recall)

        # filtering

        return loss, precision

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
            word_seq_tensor, pos_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask, sentTexts = reformat_input_data(
                batched, use_gpu=True)
            reps = span_seq(word_seq_tensor, pos_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths,
                            char_seq_recover, labels, mask, sent_texts=sentTexts)
            reps.backward()
            optimizer.step()
            span_seq.zero_grad()
        epo += 1