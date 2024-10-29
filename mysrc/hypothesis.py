"""
Trying to test the hypothesis that maybe NaviLLM's performance dropped significantly on going from R2R to R2R_fgr2r due to the fact that the latter had samples without full stop and sentence capitalization.
TAKEAWAY: no it doesn't! Negligible difference in overall performance, though per sample, it can change.
"""
import os
import json
from easydict import EasyDict as edict
from tqdm import tqdm


def to_r2r_format(jsonpath):
    """
    capitalize sentence case and add punctuations to the instructions in the json file
    """
    with open(jsonpath, 'r') as f:
        data = json.load(f)

    data = [edict(item) for item in data]
    new_data = []

    for item in data:
        for idx, instr in enumerate(item.instructions):
            instr = instr.strip()
            instr = instr.capitalize()
            if not instr.endswith('.'):
                instr += '.'
            instr += " "
            item.instructions[idx] = instr
        new_data.append(item)

    with open(jsonpath, 'w') as f:
        json.dump(new_data, f, indent=2)


def to_fgr2r_format(jsonpath):
    """
    all lowercase and no punctuations to the instructions in the json file
    """
    with open(jsonpath, 'r') as f:
        data = json.load(f)

    data = [edict(item) for item in data]
    new_data = []

    for item in data:
        for idx, instr in enumerate(item.instructions):
            instr = instr.strip()
            instr = instr.lower()
            if instr.endswith('.'):
                instr = instr[:-1]
            item.instructions[idx] = instr
        new_data.append(item)

    with open(jsonpath, 'w') as f:
        json.dump(new_data, f, indent=2)


if __name__ == "__main__":
    rootdir = 'tasks/R2R/data'

    for split in tqdm(['train', 'val_seen', 'val_unseen']):
        # jsonpath = os.path.join(rootdir, f'R2R_{split}.json')
        # to_fgr2r_format(jsonpath)
        jsonpath = os.path.join(rootdir, f'R2R_fgr2r_sub_{split}.json')
        to_r2r_format(jsonpath)
