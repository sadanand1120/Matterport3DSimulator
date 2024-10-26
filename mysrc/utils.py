import json
import os
import sys
from collections import defaultdict
import numpy as np
np.int = np.int32
import networkx as nx
import pprint
import ipdb
import torch
pp = pprint.PrettyPrinter(indent=4)

from tasks.R2R.env import R2RBatch
from tasks.R2R.utils import load_datasets, load_nav_graphs
from tasks.R2R.agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent, Seq2SeqAgent
from tasks.R2R.utils import read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, timeSince
from tasks.R2R.model import EncoderLSTM, AttnDecoderLSTM
from tasks.R2R.eval import Evaluation
from simple_colors import red, green

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'mysrc/snapshots/'
IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80


def save_r2r_seq2seq_json_encodings(split):
    ''' Extract the instruction encodings for a split. '''
    print('Saving instruction encodings for %s' % split)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    new_data = []
    for item in load_datasets([split]):
        new_item = dict(item)
        new_item['instr_encodings'] = tokenizer(item['instructions']).input_ids
        new_data.append(new_item)
    with open('tasks/R2R/data/R2R_%s_enc.json' % split, 'w') as f:
        json.dump(new_data, f, indent=4)


def eval_from_json(split, json_filepath):
    ''' Evaluate a json file of agent trajectories. '''
    ev = Evaluation([split])
    score_summary, scores = ev.score(json_filepath)
    # ipdb.set_trace()
    print(green('\n%s' % json_filepath, 'bold'))
    pp.pprint(score_summary)


def eval_seq2seq_loaded(split):
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    enc_hidden_size = 512 // 2 if False else 512
    encoder = EncoderLSTM(len(vocab), 256, enc_hidden_size, padding_idx,
                          0.5, bidirectional=False)
    encoder.load_state_dict(torch.load(os.path.join(SNAPSHOT_DIR, 'seq2seq_sample_imagenet_train_enc_iter_20000'), map_location='cuda'))
    encoder = encoder.cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              32, 512, 0.5)
    decoder.load_state_dict(torch.load(os.path.join(SNAPSHOT_DIR, 'seq2seq_sample_imagenet_train_dec_iter_20000'), map_location='cuda'))
    decoder = decoder.cuda()

    env = R2RBatch(IMAGENET_FEATURES, batch_size=40, splits=[split], tokenizer=tok)
    ev = Evaluation([split])
    os.makedirs(RESULT_DIR, exist_ok=True)
    outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, 'seq2seq_final'.lower())
    agent = Seq2SeqAgent(env, outfile, encoder, decoder, 20)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()
    score_summary, _ = ev.score(outfile)
    print(green('\n%s' % f'seq2seq_final {split}', 'bold'))
    pp.pprint(score_summary)


if __name__ == '__main__':
    # eval_seq2seq_loaded('val_unseen')
    # eval_from_json('val_unseen', 'tasks/R2R/results/val_unseen_seq2seq_final_agent.json')
    # save_r2r_seq2seq_json_encodings('val_seen')
    # eval_from_json('my_val_seen', 'third_party/NaviLLM/build/eval/R2R_my_val_seen.json')
    # eval_from_json('my_val_seen_modified', 'third_party/NaviLLM/build/eval/R2R_my_val_seen_modified.json')
    # eval_from_json('my_val_seen', '/root/mount/Matterport3DSimulator/tasks/R2R/results/my_val_seen_seq2seq_final_agent.json')
    # eval_from_json('my_val_seen_modified', '/root/mount/Matterport3DSimulator/tasks/R2R/results/my_val_seen_modified_seq2seq_final_agent.json')
    pass
