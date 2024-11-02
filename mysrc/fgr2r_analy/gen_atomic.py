import os
import sys
import json
import ast
from easydict import EasyDict as edict
from copy import deepcopy
from mysrc.frontend import gpt4, llama3, llama3_batch
from tqdm import tqdm
from pprint import pprint
import math
import numpy as np
import cv2

if __name__ == "__main__":
    preprompt_path = "mysrc/fgr2r_analy/preprompt.txt"
    with open(preprompt_path, 'r') as f:
        preprompt = f.read().strip()

    instr = "stop in the doorway on the right"
    prompt = f"\n\nInstruction: {instr}\nDecomposition:\n"

    # print("Testing GPT-4o mini")
    # text, reason = gpt4(context=preprompt,
    #                     prompt=prompt,
    #                     model='gpt-4o-mini',
    #                     temperature=0.0,
    #                     stop="END",
    #                     seed=0)
    # print(text)
    # print(reason)
    # print("*" * 50)

    print("Testing Llama-3.1 8B Instruct")
    text, reason = llama3_batch(context=preprompt,
                                prompts=[prompt] * 40,
                                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                                temperature=0.0,
                                stop="END",
                                seed=0,
                                num_gpus=2)
    print(text)
    print(reason)
    print("*" * 50)
