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
from mysrc.frontend import gpt4, llama3, llama3_batch
from mysrc.fgr2r import R2R_TEMPLATE
import math
from collections import defaultdict

APPEND_ACTIONS_HEADING = defaultdict(
    lambda: '',
    {
        math.radians(0.0): '',
        math.radians(90.0): 'Turn right. ',
        math.radians(180.0): 'Turn around. ',
        math.radians(270.0): 'Turn left. '
    }
)


def _get_heading_action(final_heading_rad, initial_heading_rad):
    # Normalize the difference within [0, 2π)
    angle_diff = (final_heading_rad - initial_heading_rad) % (2 * math.pi)
    if angle_diff < 0:
        angle_diff += 2 * math.pi

    # Find the closest predefined angle
    closest_angle = min(APPEND_ACTIONS_HEADING.keys(), key=lambda k: abs(k - angle_diff))
    return APPEND_ACTIONS_HEADING[closest_angle]


def visualize_graph(G):
    """
    Visualizes a MultiDiGraph with labeled edges.

    Parameters:
    - G (networkx.MultiDiGraph): The graph to visualize.

    Returns:
    - None
    """
    raise NotImplementedError("Needs modifications.")
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Prepare styles for edges
    connectionstyles = ['arc3,rad=0.2', 'arc3,rad=0.15', 'arc3,rad=-0.15', 'arc3,rad=-0.2']
    edge_colors = ['gray', 'blue', 'green', 'red']
    edge_key_styles = {}
    unique_keys = list(set(k for _, _, k in G.edges(keys=True)))
    for i, key in enumerate(unique_keys):
        edge_key_styles[key] = {
            'connectionstyle': connectionstyles[i % len(connectionstyles)],
            'color': edge_colors[i % len(edge_colors)]
        }

    # Draw edges and edge labels
    for u, v, k, data in G.edges(keys=True, data=True):
        # Get the style for this edge key
        style = edge_key_styles.get(k, {'connectionstyle': 'arc3,rad=0.0', 'color': 'gray'})
        # Draw the edge
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            connectionstyle=style['connectionstyle'],
            arrowstyle='-|>',
            arrowsize=10,
            edge_color=style['color'],
            width=1.5,
            ax=ax
        )
        # Prepare the edge label
        if data.get('instructions'):
            instruction = data['instructions'][0][:20] + '...'  # Truncate for readability
        else:
            instruction = 'No instruction'
        label = f"{k}: {instruction}"

        # Calculate the midpoint for label
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rad = float(style['connectionstyle'].split('=')[1])
        # Calculate the position along the edge for the label
        label_pos = (
            (x0 + x1) / 2 + rad * (y1 - y0),
            (y0 + y1) / 2 + rad * (x0 - x1)
        )
        # Draw the edge label
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=8,
            color='darkred',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def add_to_graph(G: nx.MultiDiGraph, jsonpath: str, dataset: str, split: str):
    if type(jsonpath) == str:
        with open(jsonpath, 'r') as f:
            data = json.load(f)
    elif type(jsonpath) == list:
        data = jsonpath
    else:
        raise ValueError("jsonpath must be a .json file or a list of dicts")

    for item in data:
        distance = item.get('distance', None)
        scan = item.get('scan', None)
        path_id = item.get('path_id', None)
        path = item.get('path', None)
        heading = item.get('heading', None)
        instructions = item.get('instructions', None)
        end_heading = item.get('end_heading', None)
        start_viewpoint = path[0]
        end_viewpoint = path[-1]

        # nodes: accessed as G.nodes[start_viewpoint] -> node attrs dict
        if not G.has_node(start_viewpoint):
            G.add_node(start_viewpoint)
        else:
            pass  # add attrs
        if not G.has_node(end_viewpoint):
            G.add_node(end_viewpoint)
        else:
            pass  # add attrs

        # edges: accessed as G.edges[start_viewpoint, end_viewpoint, key] -> edge attrs dict
        key = f'{dataset}::{split}::{path_id}'
        G.add_edge(start_viewpoint, end_viewpoint, key=key,
                   distance=distance, scan=scan, path_id=path_id, path=path,
                   heading=heading, instructions=instructions, end_heading=end_heading)

    return G


def get_subgraph_from_keys(G, keys, mode="full"):
    if keys is None:
        return G
    keys = set(keys)
    if mode == "full":
        H = nx.subgraph_view(G, filter_edge=lambda u, v, k: k in keys)
    elif mode == "starts":
        prefixes = tuple(keys)
        H = nx.subgraph_view(G, filter_edge=lambda u, v, k: k.startswith(prefixes))
    return H


def sample_graph_paths(G, source, target, key_constraints, key_mode="full"):
    """
    Returns a dictionary where:
    - Keys are tuples of viewpoint paths (node sequences from source to target)
    - Values are lists of edge key sequences corresponding to all possible edges along the path

    Parameters:
    - G: networkx.MultiDiGraph
    - source: starting node
    - target: ending node
    - key_constraints: list of key prefixes to filter edges
    - key_mode: 'full' for exact key match, 'starts' for prefix match

    Returns:
    - edge_paths_dict: dict of {viewpoint_path: [edge_keys_path, ...]}
    """
    H = get_subgraph_from_keys(G, key_constraints, mode=key_mode)
    if nx.has_path(H, source, target):
        viewpoint_graph_paths = list(nx.all_shortest_paths(H, source, target))
    else:
        viewpoint_graph_paths = []
    viewpoint_graph_paths = list(set(tuple(path) for path in viewpoint_graph_paths))
    edge_paths_dict = {}
    for vpath in viewpoint_graph_paths:
        edge_key_options = []
        for i, node in enumerate(vpath[:-1]):
            edge_keys = list(H[node][vpath[i + 1]].keys())
            edge_key_options.append(edge_keys)
        edge_key_sequences = list(product(*edge_key_options))
        edge_paths_dict[tuple(vpath)] = edge_key_sequences
    return edge_paths_dict


def _distinct_edge_keys(epath):
    distinct_path_ids = set()
    for i, edge_key in enumerate(epath):
        dataset, split, path_id = edge_key.split('::')
        if dataset == 'r2r':
            return set()
        # else continue as its fgr2r
        r2r_path_id = path_id.split('.')[0]
        distinct_path_ids.add(r2r_path_id)
    return distinct_path_ids


def mix_n_match(G, source, target, k1, k2, key_constraints, key_mode="full"):
    parent_r2r_path_ids = set()
    parent_r2r_path_ids.add(str(k1.split('::')[-1]))
    parent_r2r_path_ids.add(str(k2.split('::')[-1]))
    edge_paths_dict = sample_graph_paths(G, source, target, key_constraints, key_mode)
    edge_paths_wanted = []  # tuples of vpath, epath
    for vpath in edge_paths_dict.keys():
        for epath in edge_paths_dict[vpath]:
            if _distinct_edge_keys(epath) == parent_r2r_path_ids:
                edge_paths_wanted.append((vpath, epath))
    return edge_paths_wanted


def wrapper_mix_n_match(G, split='val_seen'):
    H = get_subgraph_from_keys(G, [f"r2r::{split}"], mode="starts")
    viewpoint_tuples = find_edge_tuples(H)
    print(f"Found {len(viewpoint_tuples)} edge tuples.")
    final_edge_paths = []   # list of tuples of vpath, epath
    tqdm_bar = tqdm(viewpoint_tuples)
    for i, (a, b, k1, c, d, k2) in enumerate(tqdm_bar):
        tqdm_bar.set_postfix({"found": f"{len(final_edge_paths)/1000:.3f}k"})
        # if len(final_edge_paths) >= 800:
        #     break
        edge_paths = mix_n_match(G, a, d, k1, k2, key_constraints=None)
        final_edge_paths.extend(edge_paths)
        edge_paths = mix_n_match(G, c, b, k1, k2, key_constraints=None)
        final_edge_paths.extend(edge_paths)
    return final_edge_paths


def find_edge_tuples(G, internal_common=(1, 2)):
    result = []
    edge_set = set((u, v) for u, v, k in G.edges(keys=True))
    edges_list = list(G.edges(keys=True))
    for (a, b, k1) in edges_list:
        if len(result) >= 10000:
            break
        for (c, d, k2) in edges_list:
            if len(result) >= 10000:
                break
            if a == c or a == d or b == c or b == d:
                continue
            if (a, d) in edge_set:
                continue  # Edge from a to d exists
            if (c, b) in edge_set:
                continue  # Edge from c to b exists
            path1 = G.edges[a, b, k1].get('path', [])
            path2 = G.edges[c, d, k2].get('path', [])
            internal_nodes1 = set(path1[1:-1])
            internal_nodes2 = set(path2[1:-1])
            common_nodes = internal_nodes1.intersection(internal_nodes2)
            if internal_common[0] <= len(common_nodes) <= internal_common[1]:
                result.append((a, b, k1, c, d, k2))
    return result


def _prompt_generator(G: nx.MultiDiGraph, vpath, epath, PARENT_R2R_SAMPLES: edict, split='val_seen'):
    parents_instrs = {1: set(), 2: set()}
    chunks_instrs = {1: [], 2: []}
    transition_prompt = None
    TRANSITION = [None, False]

    for iedge, edge_key in enumerate(epath):
        u = vpath[iedge]
        v = vpath[iedge + 1]
        edge_data = edict(G.edges[u, v, edge_key])
        kd, ks, kpid = edge_key.split('::')
        r2r_pid, r2r_instr_idx, chunk_idx = kpid.split('.')
        r2r_pid = str(r2r_pid)
        r2r_instr_idx = int(r2r_instr_idx)
        if iedge == 0:
            TRANSITION[0] = r2r_pid
        if r2r_pid != TRANSITION[0]:
            if not TRANSITION[1]:
                # 1st transition
                TRANSITION = (r2r_pid, True)
                prev_edge_data = edict(G.edges[vpath[iedge - 1], vpath[iedge], epath[iedge - 1]])
                _init_heading = prev_edge_data.end_heading
                _final_heading = edge_data.heading
                transition_prompt = _get_heading_action(_final_heading, _init_heading)
            else:
                # skipping such instances
                return None, None

        if not TRANSITION[1]:
            # 1
            p_instr = PARENT_R2R_SAMPLES[str(r2r_pid)].instructions[r2r_instr_idx]
            parents_instrs[1].add(p_instr)
            c_instr = edge_data.instructions[0]
            chunks_instrs[1].append(c_instr)
        else:
            # 2
            p_instr = PARENT_R2R_SAMPLES[str(r2r_pid)].instructions[r2r_instr_idx]
            parents_instrs[2].add(p_instr)
            c_instr = edge_data.instructions[0]
            chunks_instrs[2].append(c_instr)

    full_prompts = []
    for i in range(1, 3):
        parent_instrs = list(parents_instrs[i])
        chunk_instrs = chunks_instrs[i]
        parent_instrs = '\n'.join(parent_instrs)
        chunk_instrs = '\n# '.join(chunk_instrs)
        # full_prompt = f"Parent:\n{parent_instrs}\nChunks:\n# {chunk_instrs}\nCombined:\n"
        full_prompt = f"Chunks:\n# {chunk_instrs}\nCombined:\n"
        full_prompts.append(full_prompt)
    return full_prompts, transition_prompt


def gen_mix_n_match_r2r(G: nx.MultiDiGraph, vpath, epath, _PATH_ID_COUNTER, split='val_seen'):
    r2r_jsonpath = f'tasks/R2R/data/R2R_{split}.json'
    fgr2r_jsonpath = f'tasks/R2R/data/R2R_fgr2r_sub_{split}.json'
    with open(r2r_jsonpath, 'r') as f:
        _r2r_data = json.load(f)
    with open(fgr2r_jsonpath, 'r') as f:
        _fgr2r_data = json.load(f)

    parent_r2r_path_ids = list(_distinct_edge_keys(epath))
    parent_r2r_path_ids = [str(p) for p in parent_r2r_path_ids]
    PARENT_R2R_SAMPLES = edict()
    for item in _r2r_data:
        if str(item['path_id']) in parent_r2r_path_ids:
            PARENT_R2R_SAMPLES[str(item['path_id'])] = edict(item)
    assert len(PARENT_R2R_SAMPLES) == 2
    assert PARENT_R2R_SAMPLES[parent_r2r_path_ids[0]].scan == PARENT_R2R_SAMPLES[parent_r2r_path_ids[1]].scan

    ego_r2r_sample = deepcopy(R2R_TEMPLATE)
    ego_r2r_sample.scan = PARENT_R2R_SAMPLES[parent_r2r_path_ids[0]].scan
    ego_r2r_sample.path_id = f"{parent_r2r_path_ids[0]}^{parent_r2r_path_ids[1]}::{_PATH_ID_COUNTER}"
    ego_r2r_sample.path = []
    ego_r2r_sample.heading = None
    ego_r2r_sample.end_heading = None
    ego_r2r_sample.instructions = None
    ego_r2r_sample.llama_prompts = None
    ego_r2r_sample.transition_prompt = None
    ego_r2r_sample.PARENT_R2R_SAMPLES = list(PARENT_R2R_SAMPLES.values())

    for iedge, edge_key in enumerate(epath):
        u = vpath[iedge]
        v = vpath[iedge + 1]
        edge_data = edict(G.edges[u, v, edge_key])
        if iedge == 0:
            ego_r2r_sample.heading = edge_data.heading
            ego_r2r_sample.path.extend(edge_data.path)
        if iedge == len(epath) - 1:
            ego_r2r_sample.end_heading = edge_data.end_heading
        if iedge > 0:
            ego_r2r_sample.path.extend(edge_data.path[1:])

    full_prompts, transition_prompt = _prompt_generator(G, vpath, epath, PARENT_R2R_SAMPLES, split=split)
    if full_prompts is None:
        return None
    ego_r2r_sample.llama_prompts = full_prompts
    ego_r2r_sample.transition_prompt = transition_prompt
    return ego_r2r_sample


def _save_batch_mix_n_match_r2r_samples(batch_samples, batch_llama_prompts1, batch_llama_prompts2, save_path, split='val_seen'):
    preprompt_path = "mysrc/fgr2r_analy/merge.txt"
    with open(preprompt_path, 'r') as f:
        preprompt = f.read().strip()
    batch_texts1, batch_reasons1 = llama3_batch(context=preprompt,
                                                prompts=batch_llama_prompts1,
                                                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                temperature=0.0,
                                                stop="END",
                                                seed=0,
                                                num_gpus=2)
    batch_texts2, batch_reasons2 = llama3_batch(context=preprompt,
                                                prompts=batch_llama_prompts2,
                                                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                temperature=0.0,
                                                stop="END",
                                                seed=0,
                                                num_gpus=2)
    for i in range(len(batch_samples)):
        text1 = batch_texts1[i].strip()
        text2 = batch_texts2[i].strip()
        transition_text = batch_samples[i].transition_prompt
        new_instr = f"{text1} {transition_text}{text2}"
        batch_samples[i].instructions = [new_instr]
        batch_samples[i].text1 = text1
        batch_samples[i].text2 = text2

    # save
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            cur_data_samples = json.load(f)
        cur_data_samples = [edict(item) for item in cur_data_samples]
        cur_data_samples.extend(batch_samples)
    else:
        cur_data_samples = batch_samples
    with open(save_path, 'w') as f:
        json.dump(cur_data_samples, f, indent=2)


def save_mix_n_match_r2r_samples(G, final_edge_paths, split='val_seen', save_num=10e3):
    save_num = min(save_num, len(final_edge_paths))
    save_path = f'mysrc/fgr2r_analy/R2R_mix_n_match_{split}.json'
    if os.path.exists(save_path):
        print(f"Removing existing file: {save_path}")
        os.remove(save_path)
    final_edge_paths = random.sample(final_edge_paths, int(save_num))

    generated_r2r_samples = []
    llama_prompts1 = []
    llama_prompts2 = []

    _PATH_ID_COUNTER = 0

    for vpath, epath in tqdm(final_edge_paths, desc="Generating R2R samples"):
        ego_r2r_sample = gen_mix_n_match_r2r(G, vpath, epath, _PATH_ID_COUNTER, split=split)
        if ego_r2r_sample is not None:
            generated_r2r_samples.append(ego_r2r_sample)
            llama_prompts1.append(ego_r2r_sample.llama_prompts[0])
            llama_prompts2.append(ego_r2r_sample.llama_prompts[1])
        _PATH_ID_COUNTER += 1

    print(f"Generated {len(generated_r2r_samples)} R2R samples.")

    print("Generating instructions ...")
    _LLAMA_BATCH_SIZE = 128
    for i in tqdm(range(0, len(generated_r2r_samples), _LLAMA_BATCH_SIZE), desc="Batching ..."):
        batched_generated_r2r_samples = generated_r2r_samples[i:i + _LLAMA_BATCH_SIZE]
        batched_llama_prompts1 = llama_prompts1[i:i + _LLAMA_BATCH_SIZE]
        batched_llama_prompts2 = llama_prompts2[i:i + _LLAMA_BATCH_SIZE]
        _save_batch_mix_n_match_r2r_samples(batched_generated_r2r_samples,
                                            batched_llama_prompts1,
                                            batched_llama_prompts2,
                                            save_path,
                                            split=split)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val_seen")
    parser.add_argument("--save_num", type=float, default=12.5e3)
    args = parser.parse_args()

    G = nx.MultiDiGraph()
    split = args.split
    print("Adding R2R to graph ...")
    G = add_to_graph(G, f'tasks/R2R/data/R2R_{split}.json', dataset='r2r', split=split)
    print("Adding FGR2R to graph ...")
    G = add_to_graph(G, f'tasks/R2R/data/R2R_fgr2r_sub_{split}.json', dataset='fgr2r', split=split)
    print("Wrapping mix-n-match ...")
    final_edge_paths = wrapper_mix_n_match(G, split=split)
    print(f"Found {len(final_edge_paths)} edge paths.")
    save_mix_n_match_r2r_samples(G, final_edge_paths, split=split, save_num=args.save_num)
    print("Done!")
