''' Evaluation of agent trajectories '''

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

from env import R2RBatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent, Seq2SeqAgent
from utils import read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, timeSince
from model import EncoderLSTM, AttnDecoderLSTM
from simple_colors import red, green

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'mysrc/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'
IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in load_datasets(splits):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        self.scores_dict = dict()

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        # import ipdb; ipdb.set_trace()
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError as err:
                    print('Error: The provided trajectory moves from %s to %s but the navigation graph contains no '
                          'edge between these viewpoints. Please ensure the provided navigation trajectories '
                          'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0]))
                    raise
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])
        self.scores_dict[instr_id] = {
            'nav_error': self.distances[gt['scan']][final_position][goal],
            'oracle_error': self.distances[gt['scan']][nearest_position][goal],
            'trajectory_length': distance,
            'shortest_path_length': self.distances[gt['scan']][start][goal]
        }

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Trajectories not provided for %d instruction ids: %s' % (len(instr_ids), instr_ids)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])

        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])

        spls = []
        for err, length, sp in zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                spls.append(sp / max(length, sp))
            else:
                spls.append(0)

        score_summary = {
            'length': np.average(self.scores['trajectory_lengths']),
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle success_rate': float(oracle_successes) / float(len(self.scores['oracle_errors'])),
            'success_rate': float(num_successes) / float(len(self.scores['nav_errors'])),
            'spl': np.average(spls)
        }

        for idx in self.scores_dict.keys():
            suc = 1 if self.scores_dict[idx]['nav_error'] < self.error_margin else 0
            oracle_suc = 1 if self.scores_dict[idx]['oracle_error'] < self.error_margin else 0
            if suc == 1:
                spl = self.scores_dict[idx]['shortest_path_length'] / max(self.scores_dict[idx]['trajectory_length'], self.scores_dict[idx]['shortest_path_length'])
            else:
                spl = 0
            self.scores_dict[idx]['success'] = suc
            self.scores_dict[idx]['oracle_success'] = oracle_suc
            self.scores_dict[idx]['spl'] = spl

        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores


def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen']:
        # for split in ['val_seen', 'my_val_seen']:
        env = R2RBatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    # eval_simple_agents()
    # eval_seq2seq()
