"""
For a given method, it runs the model on the input json file data:
Input json: List of following dicts:
    {
    "distance": float,            e.g. 10.86
    "scan": str,                  e.g. 8194nk5LbLH
    "path_id": int,               e.g. 4332
    "path": list of gt viewpts,   e.g. ["c9e8dc09263e4d0da77d16de0ecddd39", ...]
    "heading": float,             e.g. 4.055
    "instructions": [str x 3],    e.g. ["Walk to the other end of the lobby and wait near the exit.", ...]
    }
and produces one file:
score.json: dict of overall and trajectory wise scores and metrics
preds.json: list of predicted trajectories with format [{'instr_id': string pathid_instidx, 'trajectory':[[viewpoint_id, heading_rads, elevation_rads], ...]}, ...]
"""

import json
import os
import shutil
import sys
from collections import defaultdict
import numpy as np
np.int = np.int32
import networkx as nx
from copy import deepcopy
import pprint
import ipdb
import torch
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent, Seq2SeqAgent
from utils import read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, timeSince
from model import EncoderLSTM, AttnDecoderLSTM
from simple_colors import red, green
from eval import Evaluation

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'
IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80
_OUTPUT_DIR = "tasks/R2R/infer/"
_TMP_DIR = "tasks/R2R/tmp/"


def r2r_seq2seq(input_jsonpath, split):
    data = []
    with open(input_jsonpath) as f:
        data += json.load(f)
    
    # generating preds
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    enc_hidden_size = 512 // 2 if False else 512
    encoder = EncoderLSTM(len(vocab), 256, enc_hidden_size, padding_idx,
                          0.5, bidirectional=False)
    encoder.load_state_dict(torch.load('tasks/R2R/snapshots/seq2seq_sample_imagenet_train_enc_iter_20000', map_location='cuda'))
    encoder = encoder.cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              32, 512, 0.5)
    decoder.load_state_dict(torch.load('tasks/R2R/snapshots/seq2seq_sample_imagenet_train_dec_iter_20000', map_location='cuda'))
    decoder = decoder.cuda()
    env = R2RBatch(IMAGENET_FEATURES, batch_size=40, splits=[split], tokenizer=tok)
    tmpfile = os.path.join(_TMP_DIR, f"r2r_seq2seq_{split}.json")
    agent = Seq2SeqAgent(env, tmpfile, encoder, decoder, 20)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()

    # generating scores
    ev = Evaluation([split])
    score_summary, _ = ev.score(tmpfile)
    print(green(f'r2r_seq2seq_{split}', 'bold'))
    pp.pprint(score_summary)

    # reading preds
    preds = []
    with open(tmpfile) as f:
        preds += json.load(f)
    new_preds = dict()
    for pred in preds:
        instr_id = pred['instr_id']
        traj = pred['trajectory']
        new_preds[instr_id] = traj
    data_and_preds = []
    compact_data_and_preds = []
    for d in data:
        ins = d['instructions']
        for i, inst in enumerate(ins):
            instr_id = f"{d['path_id']}_{i}"
            if instr_id in new_preds.keys():
                data_and_preds.append({
                    "distance": d["distance"],
                    "scan": d["scan"],
                    "path_id": d["path_id"],
                    "gt_path": d["path"],
                    "pred_path": [viewpoint for viewpoint, _, _ in new_preds[instr_id]],
                    "trajectory": new_preds[instr_id],
                    "heading": d["heading"],
                    "instruction": inst,
                    "metrics": ev.scores_dict[instr_id],
                })
                compact_data_and_preds.append({
                    "instruction": inst,
                    "success": ev.scores_dict[instr_id]['success'],
                    "spl": ev.scores_dict[instr_id]['spl'],
                })
    outfile = os.path.join(_OUTPUT_DIR, f"r2r_seq2seq_{split}.json")
    compact_outfile = os.path.join(_OUTPUT_DIR, "compact", f"r2r_seq2seq_{split}_compact.json")
    with open(outfile, 'w') as f:
        json.dump(data_and_preds, f, indent=4)
    with open(compact_outfile, 'w') as f:
        json.dump(compact_data_and_preds, f, indent=4)


if __name__ == "__main__":
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_DIR, "compact"), exist_ok=True)
    os.makedirs(_TMP_DIR, exist_ok=True)

    input_jsonpath = "tasks/R2R/data/R2R_val_unseen.json"
    r2r_seq2seq(input_jsonpath, split='val_unseen')
    
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
