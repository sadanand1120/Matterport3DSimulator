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
from easydict import EasyDict as edict
pp = pprint.PrettyPrinter(indent=4)

from transformers import AutoTokenizer
from tasks.R2R.env import R2RBatch
from tasks.R2R.agent import Seq2SeqAgent
from tasks.R2R.utils import read_vocab, Tokenizer, padding_idx, load_datasets
from tasks.R2R.model import EncoderLSTM, AttnDecoderLSTM
from tasks.R2R.eval import Evaluation
from copy import deepcopy
from simple_colors import green
from tqdm import tqdm

from thirdparty.NaviLLM.train import minimal_eval

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
SNAPSHOTS_DIR_R2R = 'mysrc/snapshots'
MAX_INPUT_LENGTH = 80
_DATA_PATH = "tasks/R2R/data/"
_OUTPUT_DIR = "mysrc/infer/"
_TMP_DIR = "mysrc/tmp/"


def save_r2r_seq2seq_json_encodings(split):
    ''' Extract the instruction encodings for a split. '''
    print('Saving instruction encodings for %s' % split)
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
    score_summary['total_num_instr'] = len(ev.scores_dict)
    print(green(os.path.basename(json_filepath), 'bold'))
    pp.pprint(score_summary)
    return ev, score_summary


def save_results_scores(split, tmpfile, method_name, data):
    ''' Save the results and scores to disk. '''
    # generating scores
    ev, score_summary = eval_from_json(split, tmpfile)

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
    data_and_preds.append(score_summary)
    for d in data:
        ins = d['instructions']
        for i, inst in enumerate(ins):
            instr_id = f"{d['path_id']}_{i}"
            if instr_id in new_preds.keys():
                data_and_preds.append({
                    "instr_id": instr_id,
                    "scan": d["scan"],
                    "instruction": inst,
                    "path_id": d["path_id"],
                    "gt_path": d["path"],
                    "pred_path": [viewpoint for viewpoint, _, _ in new_preds[instr_id]],
                    "distance": d["distance"],
                    "heading": d["heading"],
                    "metrics": ev.scores_dict[instr_id],
                    "trajectory": new_preds[instr_id],
                })
    outfile = os.path.join(_OUTPUT_DIR, f"{method_name}_{split}.json")
    with open(outfile, 'w') as f:
        json.dump(data_and_preds, f, indent=4)


def r2r_seq2seq(split):
    input_jsonpath = os.path.join(_DATA_PATH, f"R2R_{split}.json")
    data = []
    with open(input_jsonpath) as f:
        data += json.load(f)

    # generating preds
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    enc_hidden_size = 512 // 2 if False else 512
    encoder = EncoderLSTM(len(vocab), 256, enc_hidden_size, padding_idx, 0.5, bidirectional=False)
    encoder.load_state_dict(torch.load(os.path.join(SNAPSHOTS_DIR_R2R, 'seq2seq_sample_imagenet_train_enc_iter_20000'), map_location='cuda'))
    encoder = encoder.cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(), 32, 512, 0.5)
    decoder.load_state_dict(torch.load(os.path.join(SNAPSHOTS_DIR_R2R, 'seq2seq_sample_imagenet_train_dec_iter_20000'), map_location='cuda'))
    decoder = decoder.cuda()
    env = R2RBatch(IMAGENET_FEATURES, batch_size=40 if len(data) > 40 else len(data), splits=[split], tokenizer=tok)
    tmpfile = os.path.join(_TMP_DIR, f"r2r_seq2seq_{split}.json")
    agent = Seq2SeqAgent(env, tmpfile, encoder, decoder, 20)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()

    save_results_scores(split, tmpfile, "r2r_seq2seq", data)


def navillm_legacy(split, use_buildpreds=False):
    # DO NOT RUN on multiple nodes in parallel, only use nnodes=1 and nproc_per_node=1
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
                '--seed', '0',
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

    if not use_buildpreds:
        tmpfile = os.path.join(_TMP_DIR, f"navillm_{split}_eval", f"R2R_{split}.json")
    else:
        tmpfile = os.path.join("thirdparty/NaviLLM/build/eval", f"R2R_{split}.json")
    save_results_scores(split, tmpfile, "navillm", data)


def navillm(split, do_set_individ_seeds=False, val_batch_size=1):
    input_jsonpath = os.path.join(_DATA_PATH, f"R2R_{split}.json")
    data = []
    with open(input_jsonpath) as f:
        data += json.load(f)

    # generating preds
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_enc = []
    for item in data:
        new_item = dict(item)
        new_item['instr_encodings'] = tokenizer(item['instructions']).input_ids
        data_enc.append(new_item)
    tmpfile_enc = os.path.join(_TMP_DIR, f"navillm_{split}_enc.json")
    with open(tmpfile_enc, 'w') as f:
        json.dump(data_enc, f, indent=4)

    navillm_args = edict({
        'stage': 'multi',
        'seed': 0,
        'mode': 'test',
        'data_dir': 'data',
        'cfg_file': 'configs/multi.yaml',
        'pretrained_model_name_or_path': 'data/models/Vicuna-7B',
        'precision': 'amp_bf16',
        'resume_from_checkpoint': 'data/model_with_pretrain.pt',
        'test_datasets': ['R2R'],
        'jsonpath': os.path.join("../..", tmpfile_enc),
        'batch_size': 12,
        'output_dir': os.path.join("../..", _TMP_DIR, f"navillm_{split}_eval"),
        'validation_split': split,
        'save_pred_results': True,
        'val_batch_size': 1,
        'do_set_individ_seeds': do_set_individ_seeds,
    })

    os.chdir("thirdparty/NaviLLM")
    minimal_eval(**navillm_args)
    os.chdir("../..")

    tmpfile = os.path.join(_TMP_DIR, f"navillm_{split}_eval", f"R2R_{split}.json")
    save_results_scores(split, tmpfile, "navillm", data)


if __name__ == "__main__":
    # eval_from_json('val_unseen', 'tasks/R2R/results/val_unseen_seq2seq_final_agent.json')
    # save_r2r_seq2seq_json_encodings('val_seen')
    # eval_from_json('my_val_seen', 'third_party/NaviLLM/build/eval/R2R_my_val_seen.json')
    # eval_from_json('my_val_seen', '/root/mount/Matterport3DSimulator/tasks/R2R/results/my_val_seen_seq2seq_final_agent.json')

    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    os.makedirs(_TMP_DIR, exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    for split in tqdm(['val_unseen']):
        # r2r_seq2seq(split)
        navillm(split, do_set_individ_seeds=False, val_batch_size=2)

    print("Current directory:", os.getcwd(), "Trying to remove:", _TMP_DIR)
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
