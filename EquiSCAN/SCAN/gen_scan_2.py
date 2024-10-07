import random
import os 
from nltk import CFG
from nltk.parse.generate import generate

from utils import *


GRAMMAR = """
C -> "(" S ")" "and" "(" S ")" | "(" S ")" "after" "(" S ")" | "(" S ")"
S -> "[" V "]" "twice" | "[" V "]" "thrice" | "[" V "]"
V -> D | U | D "opposite" D | D "around" D
D -> U "right" | U "left" | U "up" | U "down" | "turn" "left" | "turn" "right" | "turn" "up" | "turn" "down"
U -> "walk" | "look" | "run" | "jump" """

_map = {
    "Y left": "I_TURN_LEFT Y",
    "Y right": "I_TURN_RIGHT Y",
    "Y up": "I_TURN_UP Y",
    "Y down": "I_TURN_DOWN Y",

    "Y opposite left": "turn opposite left Y",
    "Y opposite right": "turn opposite right Y",
    "Y around left": "I_TURN_LEFT Y I_TURN_LEFT Y I_TURN_LEFT Y I_TURN_LEFT Y",
    "Y around right": "I_TURN_RIGHT Y I_TURN_RIGHT Y I_TURN_RIGHT Y I_TURN_RIGHT Y",

    "Y opposite up": "turn opposite up Y",
    "Y opposite down": "turn opposite down Y",
    "Y around up": "I_TURN_UP Y I_TURN_UP Y I_TURN_UP Y I_TURN_UP Y",
    "Y around down": "I_TURN_DOWN Y I_TURN_DOWN Y I_TURN_DOWN Y I_TURN_DOWN Y",

    "walk": "I_WALK",
    "look": "I_LOOK",
    "run": "I_RUN",
    "jump": "I_JUMP",

    "turn left": "I_TURN_LEFT",
    "turn right": "I_TURN_RIGHT",
    "turn up": "I_TURN_UP",
    "turn down": "I_TURN_DOWN",

    "turn opposite left": "I_TURN_LEFT I_TURN_LEFT",
    "turn opposite right": "I_TURN_RIGHT I_TURN_RIGHT",
    "turn around left": "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT",
    "turn around right": "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT",

    "turn opposite up": "I_TURN_UP I_TURN_UP",
    "turn opposite down": "I_TURN_DOWN I_TURN_DOWN",
    "turn around up": "I_TURN_UP I_TURN_UP I_TURN_UP I_TURN_UP",
    "turn around down": "I_TURN_DOWN I_TURN_DOWN I_TURN_DOWN I_TURN_DOWN",
}

map = {}
for k, v in _map.items():
    if 'Y' in k:
        for y in ['walk', 'look', 'run', 'jump']:
            map.update({k.replace('Y', y): v.replace('Y', y)})
    else:
        map.update({k: v})

d3 = {
    "X twice": "X" "X",
    "X thrice": "X" "X" "X",
    "X1 and X2": "X1" "X2",
    "X1 after X2": "X2" "X1",
}

grammar = CFG.fromstring(GRAMMAR)

depth = 6
data = list(generate(grammar, depth=depth))
data = clean_data(data)
data = set(data)
data = translate(map, data)

os.makedirs('expanded_dataset/add_prim_split/', exist_ok=True)

# Writing Dataset:
with open('expanded_dataset/expanded_scan.txt', 'w') as f:
    for d in data:
        f.write(d + '\n')

splits = ['turn up', 'jump', 'turn left']

commands = []
for split in splits:
    commands.append(f"IN: {split} OUT: {map[split]}")

train = []
test_splits = {s: [] for s in splits}
excluded = []
for d in data:
    if d:
        exc = None
        for s in splits:
            if s in d and s not in commands:
                exc = s
                test_splits[s].append(d)
        if not exc:
            train.append(d)

n = int(len(train)*0.06)
for c in commands:
    train.extend([c]*n)

random.shuffle(train)
with open(f'expanded_dataset/add_prim_split/tasks_train_addprim.txt', 'w') as f:
    for d in train:
        f.write(d + '\n')

for split, split_data in test_splits.items():
    random.shuffle(split_data)
    with open(f'expanded_dataset/add_prim_split/tasks_test_addprim_{split.replace(" ", "_")}.txt', 'w') as f:
        for d in split_data:
            f.write(d + '\n')


with open(f'expanded_dataset/add_prim_split/tasks_test_addprim_{"_".join([split.replace(" ", "_") for split in splits])}.txt', 'w') as f:
    for split, split_data in test_splits.items():
        random.shuffle(split_data)
        for d in split_data:
            f.write(d + '\n')