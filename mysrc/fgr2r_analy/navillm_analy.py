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
random.seed(0)
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from mysrc.fgr2r import R2R_TEMPLATE
import math
from collections import defaultdict
from mysrc.fgr2r import visualize_traj_from_path


def to_r2r_sample(pred_sample):
    r2r_sample = deepcopy(R2R_TEMPLATE)
    r2r_sample.scan = pred_sample.scan
    r2r_sample.path_id = pred_sample.path_id
    r2r_sample.heading = pred_sample.heading
    r2r_sample.instructions = [pred_sample.instruction]
    r2r_sample.path = pred_sample.pred_path
    return r2r_sample


if __name__ == "__main__":
    video_name = "video"
    pred_sample = edict({
        "instr_id": "5803.1.0_0",
        "scan": "8WUmhLawc2A",
        "instruction": "Go straight down hallway turn right just after the kitchen table. ",
        "path_id": "5803.1.0",
        "gt_path": [
            "b64a2ef467bb455aaee16a04bd9d7812",
            "6a0eb3cedc6847efa4044c75ad7649a8"
        ],
        "pred_path": [
            "b64a2ef467bb455aaee16a04bd9d7812",
            "6a0eb3cedc6847efa4044c75ad7649a8",
            "67ff5dc94eeb4675ba3a2074ee5ca22c",
            "ecf4e1fb7421476e9988c8823a311171",
            "6ee4b38dbc684eb3a7ccb7a4b2f36221"
        ],
        "heading": 2.371,
        "metrics": {
            "nav_error": 6.623774251179944,
            "oracle_error": 0,
            "trajectory_length": 9.069426584885965,
            "shortest_path_length": 1.9746174455068504,
            "success": 0,
            "oracle_success": 1,
            "spl": 0
        },
        "trajectory": [
            [
                "b64a2ef467bb455aaee16a04bd9d7812",
                0,
                0
            ],
            [
                "6a0eb3cedc6847efa4044c75ad7649a8",
                0,
                0
            ],
            [
                "67ff5dc94eeb4675ba3a2074ee5ca22c",
                0,
                0
            ],
            [
                "ecf4e1fb7421476e9988c8823a311171",
                0,
                0
            ],
            [
                "6ee4b38dbc684eb3a7ccb7a4b2f36221",
                0,
                0
            ]
        ]
    })

    r2r_sample = to_r2r_sample(pred_sample)

    visualize_traj_from_path(r2r_sample=r2r_sample,
                             RECORD_VIDEO=True,
                             video_path=f'mysrc/fgr2r_analy/{video_name}.mp4',
                             manual=False)
