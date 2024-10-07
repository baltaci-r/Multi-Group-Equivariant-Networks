from nltk import CFG
from nltk.parse.generate import generate

from utils import *


GRAMMAR = """
C -> "(" S ")" "and" "(" S ")" | "(" S ")" "after" "(" S ")" | "(" S ")"
S -> "[" V "]" "twice" | "[" V "]" "thrice" | "[" V "]"
V -> D | U | D "opposite" D | D "around" D
D -> U "right" | U "left" | "turn" "left" | "turn" "right"
U -> "walk" | "look" | "run" | "jump" """

_map = {
    "Y left": "I_TURN_LEFT Y",
    "Y right": "I_TURN_RIGHT Y",
    "Y opposite left": "turn opposite left Y",
    "Y opposite right": "turn opposite right Y",
    "Y around left": "I_TURN_LEFT Y I_TURN_LEFT Y I_TURN_LEFT Y I_TURN_LEFT Y",
    "Y around right": "I_TURN_RIGHT Y I_TURN_RIGHT Y I_TURN_RIGHT Y I_TURN_RIGHT Y",
    
    "walk": "I_WALK",                       
    "look": "I_LOOK",
    "run": "I_RUN",
    "jump": "I_JUMP",
    "turn left": "I_TURN_LEFT",                   
    "turn right": "I_TURN_RIGHT",                          
    "turn opposite left": "I_TURN_LEFT I_TURN_LEFT",   
    "turn opposite right": "I_TURN_RIGHT I_TURN_RIGHT", 
    "turn around left": "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT",
    "turn around right": "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT",
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

# Writing Dataset:
with open('expanded_dataset/scan.txt', 'w') as f:
    for d in data:
        f.write(d + '\n')

# Comparing to SCAN
scan_data = set()
for tag in ['test', 'train']:
    for split in ['jump', 'turn_left']:
        fname = f'add_prim_split/tasks_{tag}_addprim_{split}.txt'
        with open(fname, 'r') as f:
            scan_data.update(f.read().split('\n'))
        scan_data.remove('')

assert sorted(scan_data) == sorted(data)
