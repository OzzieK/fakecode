# -*- coding: utf-8 -*-
# @Author: Yuze
# @Date:   2019-04-15 11:47:52
# @Last Modified by:   Yuze,     Contact: yuze.gao@outlook.com
# @Last Modified time: 2019-04-15 11:48:32
# ------------------------------------------------------------------------------------------------
#  ┌───┐   ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┐
#  │Esc│   │ F1│ F2│ F3│ F4│ │ F5│ F6│ F7│ F8│ │ F9│F10│F11│F12│ │P/S│S L│P/B│  ┌┐    ┌┐    ┌┐
#  └───┘   └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┘  └┘    └┘    └┘
#  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───────┐ ┌───┬───┬───┐ ┌───┬───┬───┬───┐
#  │~ `│! 1│@ 2│# 3│$ 4│% 5│^ 6│& 7│* 8│( 9│) 0│_ -│+ =│ BacSp │ │Ins│Hom│PUp│ │N L│ / │ * │ - │
#  ├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─────┤ ├───┼───┼───┤ ├───┼───┼───┼───┤
#  │ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{ [│} ]│ | \ │ │Del│End│PDn│ │ 7 │ 8 │ 9 │   │
#  ├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤ └───┴───┴───┘ ├───┼───┼───┤ + │
#  │ Caps │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  │               │ 4 │ 5 │ 6 │   │
#  ├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────────┤     ┌───┐     ├───┼───┼───┼───┤
#  │ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│  Shift   │     │ ↑ │     │ 1 │ 2 │ 3 │   │
#  ├─────┬──┴─┬─┴──┬┴───┴───┴───┴───┴───┴──┬┴───┼───┴┬────┬────┤ ┌───┼───┼───┐ ├───┴───┼───┤ E││
#  │ Ctrl│    │Alt │         Space         │ Alt│    │    │Ctrl│ │ ← │ ↓ │ → │ │   0   │ . │←─┘│
#  └─────┴────┴────┴───────────────────────┴────┴────┴────┴────┘ └───┴───┴───┘ └───────┴───┴───┘
#  -----------------------------------------------------------------------------------------------


from models.MultiLabelSeq import MultiLabelSeq
from models.SpanClassify import SpanClassfy
from models.SpanRanking import  SpanRanking
from utils.data import Data
from utils.functions import reformat_input_data
from copy import copy, deepcopy
from tqdm import tqdm

import torch
import gc
import torch.optim as optim
import random
import argparse
import sys, time
import pickle

import numpy as np


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    # print(" Learning rate is set as:\r", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate(data, instances, model, name='eval'):
    ''''''
    right_token = 0
    whole_token = 0
    pred_results = []
    pre_scores = []
    oold_result = []
    model.eval()
    total_instances = len(instances)
    batch_size = data.HP_batch_size
    start_time = time.time()
    total_batches = total_instances // batch_size + 1

    if name == 'eval':
        print('Evaluate the dev set,  total %d batches'%(total_batches))
    elif name == 'test':
        print('Evaluate the test set, total %d batches'%(total_batches))
    total_loss = 0
    total_accuracy = 0
    for batch_id in range(total_batches):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        instance = instances[start: end]
        if not instance:
            continue
        batch_words, batch_pos, batch_sent_lens, batch_sents_recover, batch_char_seqs, batch_char_lens,\
        batch_char_recover, labels, mask, sent_texts = reformat_input_data(instance, use_gpu=False, if_train=False)
        loss, accuracy = model.forward(batch_words, batch_pos, batch_sent_lens, batch_char_seqs, batch_char_lens, batch_char_recover, labels, mask, sent_texts)
        total_loss += loss.item()
        total_accuracy += accuracy.item()
    return total_loss/total_batches, total_accuracy/total_batches

def train(data, model, tratioNum, dratioNum):
    print("Training Model ... ")
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)

    best_dev = -1
    all_instances = data.all_instances
    train_instances = all_instances[:tratioNum+1]
    for idx in range(data.HP_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print('Start Epoch %s/%s'%(idx+1, data.HP_epoch))
        if data.optimizer.lower() == 'sgd':
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        taccuracy = 0

        random.shuffle(train_instances)
        model.zero_grad()
        batch_size = data.HP_batch_size
        train_batchs = len(train_instances) // batch_size + 1
        print('total batchs', train_batchs)
        for batch_id in range(train_batchs):

            model.train()
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size

            end = end if end < len(train_instances) else len(train_instances)
            batch_instance = train_instances[start: end]
            if not batch_instance:
                continue
            batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_seq_recover, batch_char_seq, \
            batch_char_seq_lengths, batch_char_seq_recover, batch_labels, \
            batch_mask, batch_texts = reformat_input_data(batch_instance, data.HP_gpu, True)
            instance_count += 1
            loss, accuracy = model.forward(batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_char_seq, batch_char_seq_lengths, batch_char_seq_recover, batch_labels, batch_mask, batch_texts)

            sample_loss += loss.item()
            total_loss += loss.item()
            taccuracy += accuracy.item()
            if end % 500 == 0:
                sample_loss = 0

            if batch_id % data.evaluate_every == 0:
                print(taccuracy, batch_id, taccuracy / (batch_id + 1))
                devloss, devaccuracy = evaluate(data, all_instances[tratioNum+1: dratioNum+1], model)
                print("Evaluate ... \n", 'loss: ', devloss, 'accuracy: ', devaccuracy, '\n')
            loss.backward()
            optimizer.step()
            model.zero_grad()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Term Extraction')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config', help='Configuration File', default='None')
    parser.add_argument('--wordemb', help='Embedding for words', default='None')
    parser.add_argument('--charemb', help='Embedding for chars', default='None')
    parser.add_argument('--model', help='Choose a model', choices=['spanc', 'multi'], default='spanc')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="save/model/saved_model.lstm.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    args = parser.parse_args()

    data = Data()
    data.HP_gpu = False #torch.cuda.is_available()
    instances = data.all_instances
    total_num = len(instances)
    ratio = data.data_ratio
    train_ratio = round(ratio[0]*total_num)
    dev_ratio = round((ratio[0]+ratio[1])*total_num)
    # train_instances = instances[:train_ratio+1]
    # dev_instances = instances[train_ratio+1:dev_ratio+1]
    # test_instances = instances[dev_ratio+1:]
    model = SpanRanking(data)
    if args.model == 'multi':
        model = MultiLabelSeq(data)
    if args.model == 'spanc':
        model = SpanClassfy(data)
    train(data, model, train_ratio, dev_ratio)
