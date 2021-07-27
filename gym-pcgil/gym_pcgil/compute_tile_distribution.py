REV_TILES_MAP = { "door": "g",
                  "key": "+",
                  "player": "A",
                  "bat": "1",
                  "spider": "2",
                  "scorpion": "3",
                  "solid": "w",
                  "empty": "."}

TILES_MAP = {"g": "door",
             "+": "key",
             "A": "player",
             "1": "bat",
             "2": "spider",
             "3": "scorpion",
             "w": "solid",
             ".": "empty"}

def to_2d_array_level(file_name):
    level = []

    with open(file_name, 'r') as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != '\n':
                    new_row.append(TILES_MAP[char])
            level.append(new_row)

    # Remove the border
    truncated_level = level[1: len(level) - 1]
    level = []
    for row in truncated_level:
        new_row = row[1: len(row) - 1]
        level.append(new_row)
    return level


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

    level_as_str = []
    for row in level:
        level_as_str.append(''.join(row))

    return ''.join(level_as_str)



training_root_path = '/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/narrow/init_maps_lvl{}/init_map_{}.txt'
tile_freq = {"g": 0, "+": 0, "A": 0, "1": 0, "2": 0, "3": 0, "w": 0, ".": 0}

for i in range(50):
    for j in range(25):
        map_path = training_root_path.format(i, j)
        map = to_char_level(to_2d_array_level(map_path))
        for char_tile in map:
            tile_freq[char_tile] += 1

print(tile_freq)
tile_distribution = {"g": 0, "+": 0, "A": 0, "1": 0, "2": 0, "3": 0, "w": 0, ".": 0}
for k, v in tile_freq.items():
    tile_distribution[k] = v / (77*50.0*25.0)

print(tile_distribution)

