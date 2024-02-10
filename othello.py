import os
import pgn
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing
import pickle
import psutil
import seaborn as sns
import itertools
from copy import copy, deepcopy
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

mask = np.zeros(64).reshape(8, 8)
mask[3, 3] = 1
mask[3, 4] = 1
mask[4, 3] = 1
mask[4, 4] = 1
mask = mask.astype(bool)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Othello is a strategy board game for two players (Black and White), played on an 8 by 8 board. 
# The game traditionally begins with four discs placed in the middle of the board as shown below. Black moves first.
# W (27) B (28)
# B (35) W (36)

def permit(s):
    s = s.lower()
    if len(s) != 2:
        return -1
    if s[0] not in rows or s[1] not in columns:
        return -1
    return rows.index(s[0]) * 8 + columns.index(s[1])

def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])

start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

wanna_use = "othello_synthetic"

class Othello:
    def __init__(self, ood_perc=0., data_root=None, wthor=False, ood_num=1000):
        # ood_perc: probability of swapping an in-distribution game (real championship game)
        # with a generated legit but stupid game, when data_root is None, should set to 0
        # data_root: if provided, will load pgn files there, else load from data/gen10e5
        # ood_num: how many simulated games to use, if -1, load 200 * 1e5 games = 20 million
        self.ood_perc = ood_perc
        self.sequences = []
        self.results = []
        self.board_size = 8 * 8
        criteria = lambda fn: fn.endswith("pgn") if wthor else fn.startswith("liveothello")
        if data_root is None:
            if ood_num == 0:
                return
            else:
                if ood_num != -1:  # this setting used for generating synthetic dataset
                    num_proc = multiprocessing.cpu_count() # use all processors
                    p = multiprocessing.Pool(num_proc)
                    for can in tqdm(p.imap(get_ood_game, range(ood_num)), total=ood_num):
                        if not can in self.sequences:
                            self.sequences.append(can)
                    p.close()
                    t_start = time.strftime("_%Y%m%d_%H%M%S")
                    if ood_num > 1000:
                        with open(f'./data/{wanna_use}/gen10e5_{t_start}.pickle', 'wb') as handle:
                            pickle.dump(self.sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    bar = tqdm(os.listdir(f"./data/{wanna_use}"))
                    trash = []
                    cnt = 0 
                    for f in bar:
                        if not f.endswith(".pickle"):
                            continue
                        with open(os.path.join(f"./data/{wanna_use}", f), 'rb') as handle:
                            cnt += 1
                            if cnt > 250:
                                break
                            b = pickle.load(handle)
                            if len(b) < 9e4:  # should be 1e5 each
                                trash.append(f)
                                continue
                            self.sequences.extend(b)
                        process = psutil.Process(os.getpid())
                        mem_gb = process.memory_info().rss / 2 ** 30
                        bar.set_description(f"Mem Used: {mem_gb:.4} GB")
                    print("Deduplicating...")
                    seq = self.sequences
                    seq.sort()
                    self.sequences = [k for k, _ in itertools.groupby(seq)]
                    for t in trash:
                        os.remove(os.path.join(f"./data/{wanna_use}", f))
                    print(f"Deduplicating finished with {len(self.sequences)} games left")
                    self.val = self.sequences[20000000:]
                    self.sequences = self.sequences[:20000000]
                    print(f"Using 20 million for training, {len(self.val)} for validation")
        else:
            for fn in os.listdir(data_root):
                if criteria(fn):
                    with open(os.path.join(data_root, fn), "r") as f:
                        pgn_text = f.read()
                    games = pgn.loads(pgn_text)
                    num_ldd = len(games)
                    processed = []
                    res = []
                    for game in games:
                        tba = []
                        for move in game.moves:
                            x = permit(move)
                            if x != -1:
                                tba.append(x)
                            else:
                                break
                        if len(tba) != 0:
                            try:
                                rr = [int(s) for s in game.result.split("-")]
                            except:
                                # print(game.result)
                                # break
                                rr = [0, 0]
                            res.append(rr)
                            processed.append(tba)

                    num_psd = len(processed)
                    print(f"Loaded {num_psd}/{num_ldd} (qualified/total) sequences from {fn}")
                    self.sequences.extend(processed)
                    self.results.extend(res)
        
    def __len__(self, ):
        return len(self.sequences)
    def __getitem__(self, i):
        if random.random() < self.ood_perc:
            tbr = get_ood_game(0)
        else:
            tbr = self.sequences[i]
        return tbr
    
def get_ood_game(_):
    tbr = []
    ab = OthelloBoardState()
    possible_next_steps = ab.get_valid_moves()
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        ab.update([next_step, ])
        possible_next_steps = ab.get_valid_moves()
    return tbr
    
def get(ood_perc=0., data_root=None, wthor=False, ood_num=1000):
    return Othello(ood_perc, data_root, wthor, ood_num)
    
class OthelloBoardState():
        return container

if __name__ == "__main__":
    pass