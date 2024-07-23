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
and produces two files with the predicted trajectories of format [{'instr_id': string pathid_instidx, 'trajectory':[[viewpoint_id, heading_rads, elevation_rads], ...]}, ...] and scores.
"""

import json
import os
import sys
import subprocess
import shutil
import numpy as np
np.int = np.int32
import pprint
import torch
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from agent import Seq2SeqAgent
from utils import read_vocab, Tokenizer, padding_idx
from model import EncoderLSTM, AttnDecoderLSTM
from simple_colors import green
from eval import Evaluation
from copy import deepcopy

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80
_DATA_PATH = "tasks/R2R/data/"
_OUTPUT_DIR = "tasks/R2R/infer/"
_TMP_DIR = "tasks/R2R/tmp/"

def r2r_seq2seq(split):
    input_jsonpath = os.path.join(_DATA_PATH, f"R2R_{split}.json")
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
    env = R2RBatch(IMAGENET_FEATURES, batch_size=40 if len(data) > 40 else len(data), splits=[split], tokenizer=tok)
    tmpfile = os.path.join(_TMP_DIR, f"r2r_seq2seq_{split}.json")
    agent = Seq2SeqAgent(env, tmpfile, encoder, decoder, 20)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()

    # generating scores
    ev = Evaluation([split])
    score_summary, _ = ev.score(tmpfile)
    score_summary['total_num_instr'] = len(ev.scores_dict)
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
    compact_data_and_preds.append(score_summary)
    for d in data:
        ins = d['instructions']
        for i, inst in enumerate(ins):
            instr_id = f"{d['path_id']}_{i}"
            if instr_id in new_preds.keys():
                data_and_preds.append({
                    "distance": d["distance"],
                    "scan": d["scan"],
                    "path_id": d["path_id"],
                    "instr_id": instr_id,
                    "gt_path": d["path"],
                    "pred_path": [viewpoint for viewpoint, _, _ in new_preds[instr_id]],
                    "trajectory": new_preds[instr_id],
                    "heading": d["heading"],
                    "instruction": inst,
                    "metrics": ev.scores_dict[instr_id],
                })
                compact_data_and_preds.append({
                    "instruction": inst,
                    "instr_id": instr_id,
                    "success": ev.scores_dict[instr_id]['success'],
                    "oracle_success": ev.scores_dict[instr_id]['oracle_success'],
                    "spl": ev.scores_dict[instr_id]['spl'],
                    "gt_path": d["path"],
                    "pred_path": [viewpoint for viewpoint, _, _ in new_preds[instr_id]],
                })
    outfile = os.path.join(_OUTPUT_DIR, f"r2r_seq2seq_{split}.json")
    compact_outfile = os.path.join(_OUTPUT_DIR, "compact", f"r2r_seq2seq_{split}_compact.json")
    with open(outfile, 'w') as f:
        json.dump(data_and_preds, f, indent=4)
    with open(compact_outfile, 'w') as f:
        json.dump(compact_data_and_preds, f, indent=4)

def navillm(split, use_buildpreds=False):
    input_jsonpath = os.path.join(_DATA_PATH, f"R2R_{split}.json")
    data = []
    with open(input_jsonpath) as f:
        data += json.load(f)

    if not use_buildpreds:
        # generating preds
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        data_enc = []
        for item in data:
            new_item = dict(item)
            new_item['instr_encodings'] = tokenizer(item['instructions']).input_ids
            data_enc.append(new_item)
        tmpfile_enc = os.path.join(_TMP_DIR, f"navillm_{split}_enc.json")
        with open(tmpfile_enc, 'w') as f:
            json.dump(data_enc, f, indent=4)
        os.chdir("thirdparty/NaviLLM")
        try:
            command = [
                'torchrun', 
                '--nnodes=1', 
                '--nproc_per_node=1', 
                '--master_port', '41000', 
                'train.py',
                '--stage', 'multi', 
                '--mode', 'test', 
                '--data_dir', 'data', 
                '--cfg_file', 'configs/multi.yaml',
                '--pretrained_model_name_or_path', 'data/models/Vicuna-7B', 
                '--precision', 'amp_bf16',
                '--resume_from_checkpoint', 'data/model_with_pretrain.pt',
                '--test_datasets', 'R2R',
                '--jsonpath', os.path.join("../..", tmpfile_enc),
                '--batch_size', '12',
                '--output_dir', os.path.join("../..", _TMP_DIR, f"navillm_{split}_eval"),
                '--validation_split', split, 
                '--save_pred_results'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            # Print output as it is received
            with process.stdout, process.stderr:
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                for line in iter(process.stderr.readline, ''):
                    sys.stderr.write(line)
            # Wait for the process to complete
            process.wait()
            if process.returncode != 0:
                print(f"An error occurred with return code {process.returncode}")
        except Exception as e:
            print(f"An error occurred: {e}")
        os.chdir("../..")

    # generating scores
    if not use_buildpreds:
        tmpfile = os.path.join(_TMP_DIR, f"navillm_{split}_eval", f"R2R_{split}.json")
    else:
        tmpfile = os.path.join("thirdparty/NaviLLM/build/eval", f"R2R_{split}.json")
    ev = Evaluation([split])
    score_summary, _ = ev.score(tmpfile)
    score_summary['total_num_instr'] = len(ev.scores_dict)
    print(green(f'navillm_{split}', 'bold'))
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
    compact_data_and_preds.append(score_summary)
    for d in data:
        ins = d['instructions']
        for i, inst in enumerate(ins):
            instr_id = f"{d['path_id']}_{i}"
            if instr_id in new_preds.keys():
                data_and_preds.append({
                    "distance": d["distance"],
                    "scan": d["scan"],
                    "path_id": d["path_id"],
                    "instr_id": instr_id,
                    "gt_path": d["path"],
                    "pred_path": [viewpoint for viewpoint, _, _ in new_preds[instr_id]],
                    "trajectory": new_preds[instr_id],
                    "heading": d["heading"],
                    "instruction": inst,
                    "metrics": ev.scores_dict[instr_id],
                })
                compact_data_and_preds.append({
                    "instruction": inst,
                    "instr_id": instr_id,
                    "success": ev.scores_dict[instr_id]['success'],
                    "oracle_success": ev.scores_dict[instr_id]['oracle_success'],
                    "spl": ev.scores_dict[instr_id]['spl'],
                    "gt_path": d["path"],
                    "pred_path": [viewpoint for viewpoint, _, _ in new_preds[instr_id]],
                })
    outfile = os.path.join(_OUTPUT_DIR, f"navillm_{split}.json")
    compact_outfile = os.path.join(_OUTPUT_DIR, "compact", f"navillm_{split}_compact.json")
    with open(outfile, 'w') as f:
        json.dump(data_and_preds, f, indent=4)
    with open(compact_outfile, 'w') as f:
        json.dump(compact_data_and_preds, f, indent=4)


if __name__ == "__main__":
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_DIR, "compact"), exist_ok=True)
    os.makedirs(_TMP_DIR, exist_ok=True)

    for split in ['train', 'val_seen', 'val_unseen']:
        # r2r_seq2seq(split)
        navillm(split, use_buildpreds=True)
    
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
