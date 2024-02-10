
import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

from neel_plotly import imshow

torch.set_grad_enabled(True)





# A class to calculate the Othello Board State, shamelessly ripped from Kenneth Li's code base

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

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


class OthelloBoardState():
    # 1 is black, -1 is white
    def __init__(self, board_size = 8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []

    def get_occupied(self, ):
        board = self.state
        tbr = board.flatten() != 0
        return tbr.tolist()
    def get_state(self, ):
        board = self.state + 1  # white 0, blank 1, black 2
        tbr = board.flatten()
        return tbr.tolist()
    def get_age(self, ):
        return self.age.flatten().tolist()
    def get_next_hand_color(self, ):
        return (self.next_hand_color + 1) // 2
    
    def update(self, moves, prt=False):
        # takes a new move or new moves and update state
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()

    def umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
                
        self.age += 1
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)
        
    def __print__(self, ):
        print("-"*20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
            # tbp.append("\n")
            print(" ".join([a[k]] + tbp))
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-"*20)
        
    def tentative_move(self, move):
        # tentatively put a piece, do nothing to state
        # returns 0 if this is not a move at all: occupied or both player have to forfeit
        # return 1 if regular move
        # return 2 if forfeit happens but the opponent can drop piece at this place
        r, c = move // 8, move % 8
        if not self.state[r, c] == 0:
            return 0
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) != 0:
            return 1
        else:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            # self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
            if len(tbf) == 0:
                return 0
            else:
                return 2
        
    def get_valid_moves(self, ):
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
            else:
                pass
        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
 
    def get_gt(self, moves, func, prt=False):
        # takes a new move or new moves and update state
        container = []
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            container.append(getattr(self, func)())  
            # to predict first y, we need already know the first x
            if prt:
                self.__print__()
        return container


# 
# try:
#     othello
#     print("Othello dataset exists")

# except:
#     print("Making dataset")
#     othello = get_othello(ood_num=-1, data_root=None, wthor=True)
#     train_dataset = CharDataset(othello)

# # made_othello=True
# 
# full_seqs = list(filter(lambda x: len(x)==60, train_dataset.data.sequences))
# print(len(full_seqs))
# board_seqs = torch.tensor(full_seqs)
# print(board_seqs.numel())
# 
# # n = 50000
# # board_seqs = torch.zeros((n, 60), dtype=int)
# # for c, seq in enumerate(tqdm(othello.sequences)):
# #     board_seqs[c, :len(seq)] = torch.tensor(seq)
# #     if c == n-1:
# #         break
# 
# board_seqs_string = board_seqs
# print(board_seqs_string.numel())
# 
# board_seqs_int = board_seqs_string.clone()
# board_seqs_int[board_seqs_string < 29] += 1
# board_seqs_int[(board_seqs_string >= 29) & (board_seqs_string <= 34)] -= 1
# board_seqs_int[(board_seqs_string > 34)] -= 3
# rand = torch.randint(0, 1000000, (20,))
# print(board_seqs_int.flatten()[rand])
# print(board_seqs_string.flatten()[rand])
# # torch.save(board_seqs, "board_seqs.pt")
# 
# indices = torch.randperm(len(board_seqs_int))
# board_seqs_int = board_seqs_int[indices]
# board_seqs_string = board_seqs_string[indices]
# torch.save(board_seqs_int, "board_seqs_int.pth")
# torch.save(board_seqs_string, "board_seqs_string.pth")

# board_seqs_int = torch.load("board_seqs_int.pth")
# board_seqs_string = torch.load("board_seqs_string.pth")
# print(board_seqs_int.shape)
# imshow(board_seqs_int[:5], title="Board Seqs Int Test")
# imshow(board_seqs_string[:5], title="Board Seqs String Test")

itos = {
    0: -100,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    26: 25,
    27: 26,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 33,
    33: 34,
    34: 37,
    35: 38,
    36: 39,
    37: 40,
    38: 41,
    39: 42,
    40: 43,
    41: 44,
    42: 45,
    43: 46,
    44: 47,
    45: 48,
    46: 49,
    47: 50,
    48: 51,
    49: 52,
    50: 53,
    51: 54,
    52: 55,
    53: 56,
    54: 57,
    55: 58,
    56: 59,
    57: 60,
    58: 61,
    59: 62,
    60: 63,
}

stoi = {
    -100: 0,
    -1: 0,
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    29: 28,
    30: 29,
    31: 30,
    32: 31,
    33: 32,
    34: 33,
    37: 34,
    38: 35,
    39: 36,
    40: 37,
    41: 38,
    42: 39,
    43: 40,
    44: 41,
    45: 42,
    46: 43,
    47: 44,
    48: 45,
    49: 46,
    50: 47,
    51: 48,
    52: 49,
    53: 50,
    54: 51,
    55: 52,
    56: 53,
    57: 54,
    58: 55,
    59: 56,
    60: 57,
    61: 58,
    62: 59,
    63: 60,
}

stoi_indices = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    29,
    30,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
]
alpha = "ABCDEFGH"


def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"


board_labels = list(map(to_board_label, stoi_indices))

def str_to_int(s):
    return stoi[s] - 1


def to_int(x):
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_int(x.item())
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_int(i) for i in x]
    elif isinstance(x, int):
        return stoi[x]
    elif isinstance(x, str):
        x = x.upper()
        return to_int(to_string(x))


def to_string(x):
    """Confusingly, maps it to an int, but a board pos label not a token label (token labels have 0 == pass, and middle board cells don't exist)"""
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_string(x.item())
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_string(i) for i in x]
    elif isinstance(x, int):
        return itos[x]
    elif isinstance(x, str):
        x = x.upper()
        return 8 * alpha.index(x[0]) + int(x[1])


def to_label(x, from_int=True):
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_label(x.item(), from_int=from_int)
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_label(i, from_int=from_int) for i in x]
    elif isinstance(x, int):
        if from_int:
            return to_board_label(to_string(x))
        else:
            return to_board_label(x)
    elif isinstance(x, str):
        return x


int_to_label = to_label
string_to_label = partial(to_label, from_int=False)
str_to_label = string_to_label

def moves_to_state(moves):
    # moves is a list of string entries (ints)
    state = np.zeros((8, 8), dtype=bool)
    for move in moves:
        state[move // 8, move % 8] = 1.0
    return state

int_labels = (
    list(range(1, 28))
    + ["X", "X"]
    + list(range(28, 34))
    + ["X", "X"]
    + list(range(34, 61))
)



def get_valid_moves(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    board = OthelloBoardState()
    return board.get_gt(sequence, "get_valid_moves")


# get_valid_moves(board_seqs_string[0])

def make_plot_state(board):
    state = np.copy(board.state).flatten()
    valid_moves = board.get_valid_moves()
    next_move = board.get_next_hand_color()
    # print(next_move, valid_moves)
    for move in valid_moves:
        state[move] = next_move - 0.5
    return state


def add_counter(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    fig.layout.shapes += (
        dict(
            type="circle",
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            fillcolor="black" if is_black else "white",
            line_color="green",
            line_width=0.5,
        ),
    )
    return fig


def counter_shape(position, color, mode="normal"):
    is_black = color > 0
    row = position // 8
    col = position % 8
    shape = dict(
        type="circle",
        fillcolor="black" if is_black else "white",
    )
    if mode == "normal":
        shape.update(
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            line_color="green",
            line_width=0.5,
        )
    elif mode == "flipped":
        shape.update(
            x0=col - 0.22,
            y0=row - 0.22,
            x1=col + 0.22,
            y1=row + 0.22,
            line_color="purple",
            line_width=3,
        )
    elif mode == "new":
        shape.update(
            line_color="red",
            line_width=4,
            x0=col - 0.25,
            y0=row - 0.25,
            x1=col + 0.25,
            y1=row + 0.25,
        )
    return shape


def plot_board(moves, return_fig=False):
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    states = []
    states.append(make_plot_state(board))
    for move in moves:
        board.umpire(move)
        states.append(make_plot_state(board))
    states = np.stack(states, axis=0)
    fig = imshow(
        states.reshape(-1, 8, 8),
        color_continuous_scale="Geyser",
        aspect="equal",
        return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        animation_index=[
            f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i]) if i>=0 else 'X'} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
            for i in range(-1, len(moves))
        ],
        animation_name="Move",
    )
    fig = fig.update_layout(title_x=0.5)
    fig.update_traces(
        text=[[str(i + 8 * j) for i in range(8)] for j in range(8)],
        texttemplate="%{text}",
    )
    for c, frame in enumerate(fig.frames):
        for i in range(64):
            if states[c].flatten()[i] == 1:
                frame = add_counter(frame, i, True)
            elif states[c].flatten()[i] == -1:
                frame = add_counter(frame, i, False)
    fig.layout.shapes = fig.frames[0].layout.shapes
    if return_fig:
        return fig
    else:
        fig.show()


# plot_board(board_seqs_string[0, :5])

def add_ring(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    offset = 0.3
    fig.layout.shapes += (
        dict(
            type="rect",
            x0=col - offset,
            y0=row - offset,
            x1=col + offset,
            y1=row + offset,
            line_color="black" if is_black else "red",
            line_width=5,
            fillcolor=None,
        ),
    )
    return fig


def plot_board_log_probs(moves, logits, return_fig=False, use_counters=False):
    logits = logits.squeeze(0)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    # print(moves)
    assert len(moves) == len(logits)
    board = OthelloBoardState()
    states = []
    # states.append(make_plot_state(board))
    for move in moves:
        board.umpire(move)
        states.append(make_plot_state(board))
    states = np.stack(states, axis=0)

    log_probs = logits.log_softmax(dim=-1)
    log_probs_template = torch.zeros((len(moves), 64)).cuda() - 100
    if log_probs.shape[-1] == 61:
        log_probs_template[:, stoi_indices] = log_probs[:, 1:]
    else:
        log_probs_template[:, stoi_indices] = log_probs[:, :]
    log_probs_template = log_probs_template.reshape(-1, 8, 8)

    fig = imshow(
        log_probs_template,
        color_continuous_scale="Blues",
        zmin=-6.0,
        zmax=0.0,
        aspect="equal",
        return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        animation_index=[
            f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i])} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
            for i in range(len(moves))
        ],
        animation_name="Move",
    )
    fig = fig.update_layout(title_x=0.5)
    # fig.update_traces(text=[[str(i+8*j) for i in range(8)] for j in range(8)], texttemplate="%{text}")
    for c, frame in enumerate(tqdm(fig.frames)):
        text = []
        shapes = []
        for i in range(64):
            text.append("")
            counter_text = "O" if moves[c] != i else "X"
            if states[c].flatten()[i] == 1:
                if use_counters:
                    shapes.append(counter_shape(i, True))
                else:
                    # black = red
                    text[
                        -1
                    ] = f"<b style='font-size: 24em; color: red; '>{counter_text}</b>"
            elif states[c].flatten()[i] == -1:
                if use_counters:
                    shapes.append(counter_shape(i, False))
                else:
                    # white = green
                    text[
                        -1
                    ] = f"<b style='font-size: 24em; color: green;'>{counter_text}</b>"
            else:
                if states[c].flatten()[i] > 0.2:
                    text[
                        -1
                    ] = f"<span style='font-size: 12em; '>{to_board_label(i)}</span>"
                    # print(i, c, "b")
                    # frame = add_ring(frame, i, True)
                elif states[c].flatten()[i] < -0.2:
                    text[
                        -1
                    ] = f"<span style='font-size: 12em; color: white'>{to_board_label(i)}</span>"
                    # print(i, c, "w")
                    # frame = add_ring(frame, i, False)
        frame.layout.shapes = tuple(shapes)
        frame.data[0]["text"] = np.array(text).reshape(8, 8)
        frame.data[0]["texttemplate"] = "%{text}"
        frame.data[0][
            "hovertemplate"
        ] = "<b>%{y}%{x}</b><br>log prob: %{z}<br>prob=%{customdata}<extra></extra>"
        frame.data[0]["customdata"] = to_numpy(log_probs_template[c].exp())
    # print(states)
    fig.layout.shapes = fig.frames[0].layout.shapes
    fig.data[0]["text"] = fig.frames[0].data[0]["text"]
    fig.data[0]["texttemplate"] = fig.frames[0].data[0]["texttemplate"]
    fig.data[0]["customdata"] = fig.frames[0].data[0]["customdata"]
    fig.data[0]["hovertemplate"] = fig.frames[0].data[0]["hovertemplate"]
    if return_fig:
        return fig
    else:
        fig.show()

def plot_single_board(moves, model=None, return_fig=False, title=None):
    # moves is a list of string entries (ints)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    if len(moves) > 1:
        board.update(moves[:-1])

    prev_state = np.copy(board.state)
    prev_player = board.next_hand_color
    prev_valid_moves = board.get_valid_moves()
    board.umpire(moves[-1])
    next_state = np.copy(board.state)
    next_player = board.next_hand_color
    next_valid_moves = board.get_valid_moves()

    empty = (prev_state == 0) & (next_state == 0)
    new = (prev_state == 0) & (next_state != 0)
    flipped = (prev_state != 0) & (next_state != prev_state) & (~new)
    prev_valid = moves_to_state(prev_valid_moves)
    next_valid = moves_to_state(next_valid_moves)

    state = np.copy(next_state)
    state[flipped] *= 0.9
    state[prev_valid] = 0.1 * prev_player
    state[next_valid] = 0.5 * next_player
    state[new] = 0.9 * prev_player
    if model is not None:
        logits = model(torch.tensor(to_int(moves)).cuda().unsqueeze(0)).cpu()
        log_probs = logits.log_softmax(-1)
        lps = torch.zeros(64) - 15.0
        lps[stoi_indices] = log_probs[0, -1, 1:]

    if title is None:
        title = f"{'Black' if prev_player!=1 else 'White'} To Play. Board State After {'Black' if prev_player==1 else 'White'} Plays {to_label(moves[-1], from_int=False)} "

    fig = imshow(
        state,
        color_continuous_scale="Geyser",
        title=title,
        y=[i for i in alpha],
        x=[str(i) for i in range(8)],
        aspect="equal",
        return_fig=True,
    )
    fig = fig.update_layout(title_x=0.5)
    fig.data[0]["hovertemplate"] = "<b>%{y}%{x}</b><br>%{customdata}<extra></extra>"

    shapes = []
    texts = []
    for i in range(64):
        texts.append("")
        if empty.flatten()[i]:
            texts[-1] = to_label(i, from_int=False)
        elif flipped.flatten()[i]:
            shapes.append(counter_shape(i, prev_player == 1, mode="flipped"))
        elif new.flatten()[i]:
            shapes.append(counter_shape(i, prev_player == 1, mode="new"))
        elif prev_state.flatten()[i] != 0:
            shapes.append(counter_shape(i, prev_state.flatten()[i] == 1, mode="normal"))
        else:
            raise ValueError(i)
    fig.layout.shapes = tuple(shapes)
    fig.data[0]["text"] = np.array(texts).reshape(8, 8)
    fig.data[0]["texttemplate"] = "%{text}"
    if model is not None:
        fig.data[0]["customdata"] = np.array(
            [f"LP:{lps[i].item():.4f}<br>I:{int_labels[i]}<br>S:{i}" for i in range(64)]
        ).reshape(8, 8)
    else:
        fig.data[0]["customdata"] = np.array(
            [f"I:{int_labels[i]}<br>S:{i}" for i in range(64)]
        ).reshape(8, 8)

    if return_fig:
        return fig
    else:
        fig.show()
    return


# def one_hot(list_of_ints, num_classes=64):
#     out = torch.zeros((num_classes,), dtype=torch.float32)
#     out[list_of_ints] = 1.
#     return out
# offset = 4123456
# num_games = 2000
# games_int = board_seqs_int[offset:offset+num_games]
# games_str = board_seqs_string[offset:offset+num_games]
# big_states = np.zeros((num_games, 59, 8, 8), dtype=np.float32)
# big_valid_moves = torch.zeros((num_games, 59, 64), dtype=torch.float32)
# for i in tqdm(range(num_games)):
#     board = OthelloBoardState()
#     for j in range(59):
#         board.umpire(games_str[i][j])
#         big_states[i][j] = board.state
#         big_valid_moves[i][j] = one_hot(board.get_valid_moves())
# big_valid_moves = einops.rearrange(big_valid_moves, "num_games pos (r c) -> num_games pos r c", r=8, c=8)
# 
# big_othello_state_dict = {
#     "big_states": big_states,
#     "big_valid_moves": big_valid_moves,
#     "offset": offset,
#     "games_str": games_str,
#     "games_int": games_int,
# }
# torch.save(big_othello_state_dict, "/workspace/_scratch/big_othello_state_dict.pth")

# big_othello_state_dict = torch.load("/workspace/_scratch/big_othello_state_dict.pth")
# big_states = big_othello_state_dict["big_states"]
# big_valid_moves = big_othello_state_dict["big_valid_moves"]
# offset = big_othello_state_dict["offset"]
# num_games = 2000
# games_str = big_othello_state_dict["games_str"]
# games_int = big_othello_state_dict["games_int"]


# import transformer_lens.utils as utils

# cfg = HookedTransformerConfig(
#     n_layers=8,
#     d_model=512,
#     d_head=64,
#     n_heads=8,
#     d_mlp=2048,
#     d_vocab=61,
#     n_ctx=59,
#     act_fn="gelu",
#     normalization_type="LNPre",
# )
# model = HookedTransformer(cfg)


# sd = utils.download_file_from_hf(
#     "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
# )
# # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
# model.load_state_dict(sd)
# 
# with torch.inference_mode():
#     big_logits, big_cache = model.run_with_cache(games_int[:, :-1].cuda())