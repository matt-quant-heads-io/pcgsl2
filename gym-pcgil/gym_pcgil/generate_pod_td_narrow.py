import os

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

from helper import TILES_MAP, str_arr_from_int_arr
from PIL import Image

import numpy as np
import random
import csv
import time
from helper import to_2d_array_level, int_arr_from_str_arr
import pprint

pp = pprint.PrettyPrinter(indent=4)

#This is for creating the directories
path_dir = 'exp_trajectories/narrow/init_maps_lvl{}'
# for idx in range(5, 50):
#     os.makedirs(path_dir.format(idx))

################################################


# This code is for generating the maps
def render_map(map, prob, rep, filename='', ret_image=False):
    # format image of map for rendering
    if not filename:
        img = prob.render(map)
    else:
        img = to_2d_array_level(filename)
    img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
    img = np.array(img)
    if ret_image:
        return img
    else:
        ren = rendering.SimpleImageViewer()
        ren.imshow(img)
        input(f'')
        time.sleep(0.3)
        ren.close()

def generate_play_trace_narrow(map, prob, rep, actions_list, render=False):
    """
        The "no-change" action  is 1 greater than the number of tile types (value is 8)


    """

    play_trace = []
    # loop through from 0 to 13 (for 14 tile change actions)
    old_map = map.copy()
    actions_list = [i for i in range(8)]

    # Insert the goal state into the play trace
    play_trace.insert(0, [old_map, None, None])

    count = 0
    current_loc = [random.randint(0, len(map) - 1), random.randint(0, len(map[0]) - 1)] # [0, 0]
    tile_changes = 0
    tile_visits = 0

    rep._old_map = np.array([np.array(l) for l in map])# np.ndarray(map.copy(), shape=(len(map), len(map[0])), ndim=2).astype(np.uint8)
    rep.reset(11, 7, {0: 0.58, 1: 0.3, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})
    rep._x = current_loc[0] # 0
    rep._y = current_loc[1] # 0

    # Render starting map
    if render:
        map_img = render_map(str_arr_from_int_arr(old_map), prob, rep, ret_image=True)
        ren = rendering.SimpleImageViewer()
        ren.imshow(map_img)
        # input(f'')
        time.sleep(0.3)
        ren.close()

    # Initialize start at 0,0

    # rep._x = 0
    # rep._y = 0

    tile_visits = 0
    row_idx, col_idx = 0 , 0

    while tile_changes < 60:
        new_map = old_map.copy()
        transition_info_at_step = [None, None, None]  # [current map, destructive_action, expert_action]
        # row_idx, col_idx = random.randint(0, len(map) - 1), random.randint(0, len(map[0]) - 1) # current_loc[1], current_loc[0]
        rep._x = col_idx
        rep._y = row_idx
        # print(f"position ({current_loc[0], current_loc[1]})")
        new_map[row_idx] = old_map[row_idx].copy()
        old_tile_type = new_map[row_idx][col_idx]
        # print(f"old_tile_type is {old_tile_type}")
        next_actions = [j for j in actions_list if j != old_tile_type] + ["No-change"]*27
        new_tile_type = random.choice(next_actions)
        if new_tile_type == "No-change":
            new_tile_type = old_tile_type
        else:
            tile_changes += 1

        destructive_action = [row_idx, col_idx, new_tile_type]
        expert_action = [row_idx, col_idx, old_tile_type]
        transition_info_at_step[1] = destructive_action.copy()
        transition_info_at_step[2] = expert_action.copy()
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map.copy()
        play_trace.insert(0, transition_info_at_step.copy())

        tile_visits += 1


        # Update position
        # current_loc[0] += 1
        # rep._x += 1
        # if current_loc[0] >= rep._map.shape[1]:
        #     current_loc[0] = 0
        #     current_loc[1] += 1
        #     rep._x = 0
        #     rep._y += 1
        #     if current_loc[1] >= rep._map.shape[0]:
        #         current_loc[1] = 0
        #         rep._y = 0

        old_map = new_map

        # Render
        if render:
            map_img = render_map(str_arr_from_int_arr(new_map), prob, rep, ret_image=True)
            ren = rendering.SimpleImageViewer()
            ren.imshow(map_img)
            # input(f'')
            time.sleep(0.3)
            ren.close()

        col_idx += 1
        if col_idx >= 11:
            col_idx = 0
            row_idx += 1
            if row_idx >= 7:
                row_idx = 0

    return play_trace


# # TODO: Need to change this for Turtle and Narrow Reps
actions_list = [act for act in list(TILES_MAP.values())]
prob = ZeldaProblem()
rep = NarrowRepresentation()

# Reverse the k,v in TILES MAP for persisting back as char map .txt format
REV_TILES_MAP = { "door": "g",
                  "key": "+",
                  "player": "A",
                  "bat": "1",
                  "spider": "2",
                  "scorpion": "3",
                  "solid": "w",
                  "empty": "."}


def to_char_level(map, dir=''):
    level = []

    for row in map:
        new_row = []
        for col in row:
            new_row.append(REV_TILES_MAP[col])
        # add side borders
        new_row.insert(0, 'w')
        new_row.append('w')
        level.append(new_row)
    top_bottom_border = ['w'] * len(level[0])
    level.insert(0, top_bottom_border)
    level.append(top_bottom_border)

    level_as_str = []
    for row in level:
        level_as_str.append(''.join(row) + '\n')

    with open(dir, 'w') as f:
        for row in level_as_str:
            f.write(row)


def act_seq_to_disk(act_seq, path):
    with open(path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(act_seq)


def act_seq_from_disk(path):
    act_seqs = []
    with open(path, "r") as f:
        data = f.readlines()
        for row in data:
            act_seq = [int(n) for n in row.split('\n')[0].split(',')]
            act_seqs.append(act_seq)
    return act_seqs




# Test reading in act_seq
# print(act_seq_from_disk('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/narrow/init_maps_lvl0/repair_sequence_0.csv'))


filepath = 'playable_maps/zelda_lvl{}.txt'
act_seq_filepath = 'exp_trajectories/narrow/init_maps_lvl{}/repair_sequence_{}.csv'
for idx in range(50):
    # get the good level map
    map = int_arr_from_str_arr(to_2d_array_level(filepath.format(idx)))
    for j_idx in range(15, 25):
        temp_map = map.copy()
        play_trace = generate_play_trace_narrow(temp_map, prob, rep, actions_list, render=False)
        repair_action_seq = [a[-1] for a in play_trace[:-1]]

        # print(f"repair_action_seq is {repair_action_seq}")


        # Write final destroyed map to .txt
        to_char_level(str_arr_from_int_arr(play_trace[0][0]), dir=f"exp_trajectories/narrow/init_maps_lvl{idx}/init_map_{j_idx}.txt")
        # Write action seq to .csv
        act_seq_to_disk(repair_action_seq, f"exp_trajectories/narrow/init_maps_lvl{idx}/repair_sequence_{j_idx}.csv")


        # print(f"repair_action_seq is {repair_action_seq} len is {len(repair_action_seq)}")


        # Test reading the written destroyed level
        # destroyed_map = to_2d_array_level(f'exp_trajectories/narrow/init_maps_lvl{idx}/init_map_{j_idx}.txt')
        # # render_map(destroyed_map, prob, rep, filename="", ret_image=False)
        # #
        # #
        # #
        # # # Testing repair from destroyed map to goal map
        # #
        # # print("Rendering repair map from destroyed state")
        # init_map = int_arr_from_str_arr(destroyed_map) # play_trace[0][0]
        # print(f"destroyed_map is {destroyed_map}")
        # repair_map = init_map.copy()
        # count = 0
        # for act_seq in repair_action_seq:
        #     repair_map[act_seq[0]][act_seq[1]] = act_seq[2]
        #     count += 1
        #     print(f"repair act count : {count}")
        #     map_img = render_map(str_arr_from_int_arr(repair_map), prob, rep, ret_image=True)
        #     ren = rendering.SimpleImageViewer()
        #     ren.imshow(map_img)
        #     # input(f'')
        #     time.sleep(0.3)
        #     ren.close()
