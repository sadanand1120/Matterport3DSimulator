import os
import sys
import json
import ast
from easydict import EasyDict as edict
from copy import deepcopy
from mysrc.driver import get_sim_instance
from tqdm import tqdm
from pprint import pprint
import math
import numpy as np
import cv2

R2R_TEMPLATE = edict({
    # 'distance': 10.86,   # not used anywhere, rather directly calculated from connectivity graph
    'scan': '8194nk5LbLH',
    'path_id': 4332,
    'path': [
        'c9e8dc09263e4d0da77d16de0ecddd39',
        'f33c718aaf2c41469389a87944442c62',
        'ae91518ed77047b3bdeeca864cd04029',
        '6776097c17ed4b93aee61704eb32f06c'
    ],
    'heading': 4.055,
    'instructions': [
        'Walk to the other end of the lobby and wait near the exit. ',
        'Walk straight toward the bar with the chairs/stool. Turn left and go straight until you get to three tables with chairs. Turn left and wait near the couch. ',
        'Go forward toward the windows. Go toward the the far couch, Stop next to the couch, in front of the windows. '
    ]
})


def simulate_traj_from_path(r2r_sample, WIDTH=800, HEIGHT=600):
    sim = get_sim_instance(WIDTH=WIDTH, HEIGHT=HEIGHT, scan_id=r2r_sample.scan, viewpoint_id=r2r_sample.path[0], heading_rad=r2r_sample.heading, elevation_rad=0.0)
    trajectory = []   # list of [viewpoint_id, global_heading, global_elevation]
    make_actions = []  # list of [next_navigable_viewpoint_idx, rel_heading, rel_elevation]
    for idx in range(1, len(r2r_sample.path)):
        state = sim.getState()[0]
        trajectory.append([state.location.viewpointId, state.heading, state.elevation])
        action_ = None
        for nidx, navloc_state in enumerate(state.navigableLocations):
            if navloc_state.viewpointId == r2r_sample.path[idx]:
                action_ = [nidx, navloc_state.rel_heading, navloc_state.rel_elevation]
                break
        if action_ is None:
            raise ValueError(f"Next viewpoint {r2r_sample.path[idx]} not found in navigable locations at viewpoint {state.location.viewpointId}")
        make_actions.append(action_)
        sim.makeAction([action_[0]], [action_[1]], [action_[2]])
    state = sim.getState()[0]
    trajectory.append([state.location.viewpointId, state.heading, state.elevation])
    sim.close()
    return trajectory, make_actions


def visualize_traj_from_path(r2r_sample, WIDTH=800, HEIGHT=600, vfovdeg=60):
    VFOV = math.radians(vfovdeg)
    HFOV = 2 * math.atan(math.tan(VFOV / 2) * WIDTH / HEIGHT)
    TEXT_COLOR = [40, 40, 230]
    ANGLEDELTA = 5 * math.pi / 180
    traj, make_actions = simulate_traj_from_path(r2r_sample, WIDTH=WIDTH, HEIGHT=HEIGHT)
    sim = get_sim_instance(WIDTH=WIDTH, HEIGHT=HEIGHT, scan_id=r2r_sample.scan, viewpoint_id=r2r_sample.path[0], heading_rad=r2r_sample.heading, elevation_rad=0.0)
    action_idx = 0
    print("Instructions:")
    for instr in r2r_sample.instructions:
        print(instr)

    cv2.namedWindow('Python RGB', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Python RGB', WIDTH, HEIGHT)

    while action_idx < len(make_actions):
        state = sim.getState()[0]
        rgb = np.array(state.rgb, copy=False)
        for idx, loc in enumerate(state.navigableLocations[1:]):
            # Draw actions on the screen
            fontScale = 1.3 / loc.rel_distance
            x = int(WIDTH / 2 + loc.rel_heading / HFOV * WIDTH)
            y = int(HEIGHT / 2 - loc.rel_elevation / VFOV * HEIGHT)
            cv2.putText(rgb, f"{idx+1}:{loc.viewpointId[:4]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, TEXT_COLOR, thickness=2)
        cv2.imshow('Python RGB', rgb)
        k = cv2.waitKey(0)
        k = (k & 255)
        if k == ord('q'):
            break

        cur_action = make_actions[action_idx]
        to_take_action = None
        # heading
        if cur_action[1] >= ANGLEDELTA:
            to_take_action = [0, ANGLEDELTA, 0]
            make_actions[action_idx][1] -= ANGLEDELTA
        elif cur_action[1] <= -ANGLEDELTA:
            to_take_action = [0, -ANGLEDELTA, 0]
            make_actions[action_idx][1] += ANGLEDELTA
        elif abs(cur_action[1]) > 0:
            to_take_action = [0, cur_action[1], 0]
            make_actions[action_idx][1] = 0
        # elevation
        elif cur_action[2] >= ANGLEDELTA:
            to_take_action = [0, 0, ANGLEDELTA]
            make_actions[action_idx][2] -= ANGLEDELTA
        elif cur_action[2] <= -ANGLEDELTA:
            to_take_action = [0, 0, -ANGLEDELTA]
            make_actions[action_idx][2] += ANGLEDELTA
        elif abs(cur_action[2]) > 0:
            to_take_action = [0, 0, cur_action[2]]
            make_actions[action_idx][2] = 0
        # viewpoint idx
        elif cur_action[0] >= 0:
            next_viewpoint_id = traj[action_idx + 1][0]
            next_viewpoint_idx = None
            for nidx, navloc_state in enumerate(state.navigableLocations):
                if navloc_state.viewpointId == next_viewpoint_id:
                    next_viewpoint_idx = nidx
                    break
            assert next_viewpoint_idx == 1, "Next viewpoint is not the closest to center"
            to_take_action = [next_viewpoint_idx, 0, 0]
            make_actions[action_idx][0] = -1
        else:
            action_idx += 1

        if to_take_action is not None:
            sim.makeAction([to_take_action[0]], [to_take_action[1]], [to_take_action[2]])

    print("Goal reached!")
    sim.close()
    cv2.destroyAllWindows()


def load_fg_r2r_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = [edict(item) for item in data]
    for item in data:
        item['new_instructions'] = ast.literal_eval(item['new_instructions'])
        new_instructions_list = []
        split_paths_list = []
        for new_instr, split_path in zip(item['new_instructions'], item['chunk_view']):
            new_instructions_list.append([' '.join(sub_instr_list) for sub_instr_list in new_instr])
            split_paths_list.append([tuple(path_endpts) for path_endpts in split_path])
        item['new_instructions'] = new_instructions_list
        item['chunk_view'] = split_paths_list
    return data


def gen_r2r_samples(fgr2r_loaded_sample):
    """
    given a fgr2r sample, generate multiple action-level R2R samples
    """
    generated_r2r_samples = []
    sim_traj, sim_make_actions = simulate_traj_from_path(fgr2r_loaded_sample)
    for idx, full_instr in enumerate(tqdm(fgr2r_loaded_sample['instructions'], leave=False)):
        if idx >= len(fgr2r_loaded_sample['new_instructions']):
            continue
        sub_instrs = fgr2r_loaded_sample['new_instructions'][idx]  # list of sub-instructions strings
        sub_paths = fgr2r_loaded_sample['chunk_view'][idx]  # list of sub-path end point tuples
        for sub_idx, sub_instr, sub_path in zip(range(len(sub_instrs)), sub_instrs, sub_paths):
            gen_r2r_sample = deepcopy(R2R_TEMPLATE)
            gen_r2r_sample.scan = fgr2r_loaded_sample.scan
            gen_r2r_sample.path_id = f"{fgr2r_loaded_sample.path_id}_{idx}.{sub_idx}"
            gen_r2r_sample.path = fgr2r_loaded_sample.path[sub_path[0] - 1:sub_path[1]]   # {sub_path[0]-1, sub_path[1]-1} subpath, both inclusive
            gen_r2r_sample.instructions = [sub_instr]
            gen_r2r_sample.heading = sim_traj[sub_path[0] - 1][1]
            generated_r2r_samples.append(gen_r2r_sample)
    return generated_r2r_samples


if __name__ == "__main__":
    # for split in tqdm(['train', 'val_seen', 'val_unseen']):
    #     fgr2r_path = f'tasks/R2R/data/FGR2R_{split}.json'
    #     fgr2r_data = load_fg_r2r_data(fgr2r_path)
    #     generated_r2r_samples = []
    #     for sample in tqdm(fgr2r_data, leave=False):
    #         gen_r2rs = gen_r2r_samples(sample)
    #         generated_r2r_samples.extend(gen_r2rs)
    #     with open(f'tasks/R2R/data/R2R_fgr2r_sub_{split}.json', 'w') as f:
    #         json.dump(generated_r2r_samples, f, indent=2)

    r2r_temp = edict({
        "scan": "QUCTc6BB5sX",
        "path_id": "2365_1.0",
        "path": [
            "75ff3e14cc414e0e80e81f036520aedf",
            "f1073371a7f44247a97f8c811c231676"
        ],
        "heading": 4.613,
        "instructions": [
            "go down hallway towards the two white vase"
        ]
    })

    visualize_traj_from_path(r2r_temp)
