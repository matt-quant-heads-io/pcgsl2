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


actions_list = [act for act in list(TILES_MAP.values())]
prob = ZeldaProblem()
rep = NarrowRepresentation()
rep._x = 0
rep._y = 0

# playable_map = to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/narrow/init_maps_lvl0/init_map_4.txt')
playable_map = to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps_lg/zelda_lvl15.txt')
# playable_map = to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps/zelda_lvl3.txt')
# playable_map = to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps/zelda_lvl2.txt')
# playable_map = to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps/zelda_lvl1.txt')
# playable_map = to_2d_array_level(f'/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps/zelda_lvl0.txt')


render_map(playable_map, prob, rep, filename="", ret_image=False)