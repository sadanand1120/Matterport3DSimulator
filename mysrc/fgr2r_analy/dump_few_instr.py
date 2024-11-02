import os
import sys
import json
import ast
from easydict import EasyDict as edict
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint
import math
import numpy as np
import cv2
import random


def get_all_instr_strs(jsonpath, save_path, N):
    with open(jsonpath, 'r') as f:
        data = json.load(f)

    instrs = [edict(item).instructions[0] for item in data]

    # randomly sample N instructions
    instrs = random.sample(instrs, N)

    save_dict = edict()
    for instr in instrs:
        save_dict[instr] = " "

    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    N = 200
    get_all_instr_strs('tasks/R2R/data/R2R_fgr2r_sub_val_seen.json', f'mysrc/fgr2r_analy/val_seen_instrs{N}.json', N=N)
