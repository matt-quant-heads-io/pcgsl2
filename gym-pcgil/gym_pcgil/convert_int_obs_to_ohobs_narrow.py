import numpy as np
import pickle
from helper import TILES_MAP

"""
input: int[][]
x: x (row) position of player
y: y (column) position of player

returns: one hot observation where player position is center of padded map
"""

def transform_narrow(obs, x, y, return_onehot=True, transform=True):
    pad = 11
    pad_value = 1
    size = 22
    map = obs # obs is int
    # View Centering
    padded = np.pad(map, pad, constant_values=pad_value)
    cropped = padded[y:y + size, x:x + size]
    obs = cropped

    if return_onehot:
        obs = np.eye(8)[obs]
        if transform:
            new_obs = []
            for i in range(22):
                for j in range(22):
                    for z in range(8):
                        new_obs.append(obs[i][j][z])
            return new_obs
    return obs



"""
Steps:
------
1) transform pt + action_sequence into training data (oh_matrix --> action mapping), save this to csv (last column is action)
2) read in csv & use this to train/test MLP classifier (can do this in a notebook) save trained model
3) load in saved model and perform inference in a notebook


"""

from helper import to_2d_array_level, int_arr_from_str_arr, int_map_to_onehot
from generate_pod_td_narrow import act_seq_from_disk
import pandas as pd


rep_str = 'narrow'
oh_obs = []
# Onehot
# destroyed_map = int_map_to_onehot(int_arr_from_str_arr(to_2d_array_level(f'exp_trajectories/{rep_str}/init_maps_lvl0/init_map_0.txt')))
destroyed_map = int_arr_from_str_arr(to_2d_array_level(f'exp_trajectories/{rep_str}/init_maps_lvl0/init_map_0.txt'))
destroyed_map_flatten = [item for sublist in destroyed_map for item in sublist]

# Onehot
# for new_row in destroyed_map_flatten:
#     oh_obs.extend(new_row)

# Onehot
# data_for_df = {f"col_{i}": [] for i in range(len(oh_obs))}
data_for_df = {f"col_{i}": [] for i in range(3872)}
data_for_df['target'] = []
# data_for_df = {f"obs": [], "target": []}


for i in range(50):
    for j in range(25):
        print(f"{i},{j}")
        # Onehot
        # next_destroyed_map = int_map_to_onehot(int_arr_from_str_arr(to_2d_array_level(f'exp_trajectories/{rep_str}/init_maps_lvl{i}/init_map_{j}.txt')))


        next_destroyed_map = int_arr_from_str_arr(to_2d_array_level(f'exp_trajectories/{rep_str}/init_maps_lvl{i}/init_map_{j}.txt'))
        repair_sequence = act_seq_from_disk(f'exp_trajectories/{rep_str}/init_maps_lvl{i}/repair_sequence_{j}.csv')
        repair_map = next_destroyed_map.copy()

        for rep_action in repair_sequence:
            new_oh_obs = []
            # Flatten the map
            new_map = repair_map.copy()
            # Onehot
            # for new_row in new_map_flatten:
            #     # print(f"new_row is {new_row}")
            #     new_oh_obs.extend(new_row)
            # print(new_map_flatten)
            # data_for_df[f"obs"].append(transform_narrow(new_map, rep_action[1], rep_action[0]))

            # train using CNN do not flatten the obs: aka transform_narrow(new_map, rep_action[1], rep_action[0], flat=False), and comment out for-loop for z, tile in enumerate(flat_obs)
            flat_obs = transform_narrow(new_map, rep_action[1], rep_action[0])
            for z, tile in enumerate(flat_obs):
                data_for_df[f"col_{z}"].append(tile)

            data_for_df['target'].append(rep_action[2])

            new_map[rep_action[0]][rep_action[1]] = rep_action[2]
            repair_map = new_map

print(f"Creating DataFrame")
df = pd.DataFrame(data=data_for_df)


print(f"Writing DataFrame to csv")
df.to_csv("narrow_td_onehot_obs_50_goals_25_starts.csv", index=False)
print(f"Done!")
print(df.head())
print(df.tail())




