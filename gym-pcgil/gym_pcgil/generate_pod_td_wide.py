import os

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

from helper import TILES_MAP, str_arr_from_int_arr
from PIL import Image

import numpy as np
import random
import csv

from helper import to_2d_array_level, int_arr_from_str_arr
import pprint

pp = pprint.PrettyPrinter(indent=4)

#This is for creating the directories
path_dir = 'exp_trajectories/wide/init_maps_lvl{}'
# for idx in range(50):
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
        ren.close()

def generate_play_trace_wide(map, prob, rep, actions_list, render=False):
    play_trace = []
    # loop through from 0 to 13 (for 14 tile change actions)
    old_map = map.copy()
    actions_list = [i for i in range(8)]

    # Insert the goal state into the play trace
    play_trace.insert(0, [old_map, None, None])

    # Render starting map
    if render:
        map_img = render_map(str_arr_from_int_arr(old_map), prob, rep, ret_image=True)
        ren = rendering.SimpleImageViewer()
        ren.imshow(map_img)
        input(f'')
        ren.close()

    count = 0
    for num_tc_action in range(62):
        new_map = old_map.copy()
        transition_info_at_step = [None, None, None]  # [current map, destructive_action, expert_action]
        row_idx, col_idx = random.randint(0, len(map) - 1), random.randint(0, len(map[0]) - 1)
        new_map[row_idx] = old_map[row_idx].copy()
        old_tile_type = new_map[row_idx][col_idx]
        next_actions = [j for j in actions_list if j != old_tile_type]
        new_tile_type = random.choice(next_actions)
        destructive_action = [row_idx, col_idx, new_tile_type]
        expert_action = [row_idx, col_idx, old_tile_type]
        transition_info_at_step[1] = destructive_action.copy()
        transition_info_at_step[2] = expert_action.copy()
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map.copy()
        play_trace.insert(0, transition_info_at_step.copy())
        count += 1
        print(f"POD act count : {count}")

        old_map = new_map

        # Render
        if render:
            map_img = render_map(str_arr_from_int_arr(new_map), prob, rep, ret_image=True)
            ren = rendering.SimpleImageViewer()
            ren.imshow(map_img)
            input(f'')
            ren.close()

    return play_trace


# # TODO: Need to change this for Turtle and Narrow Reps
actions_list = [act for act in list(TILES_MAP.values())]
prob = ZeldaProblem()
rep = WideRepresentation()

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
# print(act_seq_from_disk('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/wide/init_maps_lvl0/repair_sequence_0.csv'))


filepath = 'playable_maps/zelda_lvl{}.txt'
act_seq_filepath = 'exp_trajectories/wide/init_maps_lvl{}/repair_sequence_{}.csv'
for idx in range(50):
    # get the good level map
    map = int_arr_from_str_arr(to_2d_array_level(filepath.format(idx)))
    for j_idx in range(50):
        temp_map = map.copy()
        play_trace = generate_play_trace_wide(temp_map, prob, rep, actions_list, render=False)
        repair_action_seq = [a[-1] for a in play_trace[:-1]]

        # Write final destroyed map to .txt
        to_char_level(str_arr_from_int_arr(play_trace[0][0]), dir=f"exp_trajectories/wide/init_maps_lvl{idx}/init_map_{j_idx}.txt")
        # Write action seq to .csv
        act_seq_to_disk(repair_action_seq, f"exp_trajectories/wide/init_maps_lvl{idx}/repair_sequence_{j_idx}.csv")

        # print(f"repair_action_seq is {repair_action_seq} len is {len(repair_action_seq)}")


        # Test reading the written destroyed level
        # destroyed_map = to_2d_array_level(f'exp_trajectories/wide/init_maps_lvl{idx}/init_map_{j_idx}.txt')
        # # render_map(destroyed_map, prob, rep, filename="", ret_image=False)
        #
        #
        #
        # # Testing repair from destroyed map to goal map
        #
        # print("Rendering repair map from destroyed state")
        # init_map = int_arr_from_str_arr(play_trace[0][0])
        # repair_map = init_map.copy()
        # count = 0
        # for act_seq in repair_action_seq:
        #     repair_map[act_seq[0]][act_seq[1]] = act_seq[2]
        #     count += 1
        #     print(f"repair act count : {count}")
        #     map_img = render_map(str_arr_from_int_arr(repair_map), prob, rep, ret_image=True)
        #     ren = rendering.SimpleImageViewer()
        #     ren.imshow(map_img)
        #     input(f'')
        #     ren.close()

# TODO: implmenet narrow as a supervised learning soln
# Start with narrow for supervised ** (wide in paralell then RL)


# Take broken map in and x y and type out
# Two ways of doing this: RL way and the supervised way
# RL way is: -1 if agent predicts wrong action if correct +1
# when the agent produces and an output, compare it with the target level
# if the output is different from current level AND the same as target leve, then reward
# supervised way is: produce a large number of training tuples, where we have
# features = broken level, target = x, y and tile for _some_ repair action
# for every broken level there are 14 different correct answers
# simply train on this dataset, accuracy will never be higher than 1/14th, but it _should_ improve
# ****should be work on both narrow and wide?****
# in neither of these settings do we even need to record the sequence of destruction

# a different thing: take broken levels in and produce correct levels - predict _all_tiles at the same time


# TODO: Now that we've verifiied the trainable espisodes are proper we need to perform the following 2 steps prior to training
#   1) Verificaiton of correctness of numpy archive & read in the file pairs (i.e. repair episodes) from step 1 to create the numpy archive file.
#   2) Create github for repo and push repo to GIL machine (with numpy archive file can put file pairs in .ignore)
#   3) delete all existing conda envs on GIL machine and all pcgrl repos and run the install proceess again (follow pcgrl github) so we have
#       clean packages & environemnt. Create new conda environment after deleting all exsiting envs on GIL.
#   4) start training wide model using GIL machine