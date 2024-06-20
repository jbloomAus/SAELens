# %% IMPORTS
%load_ext autoreload
%autoreload 2
import wandb
from sae_training.utils import LMSparseAutoencoderSessionloader
import torch
import numpy as np
import sys
from pathlib import Path
from torch import Tensor
from jaxtyping import Bool , Float
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import pandas as pd

# %% Download Othello stuff
!git clone https://github.com/likenneth/othello_world
OTHELLO_ROOT = Path("./othello_world/")
OTHELLO_MECHINT_ROOT = OTHELLO_ROOT / "mechanistic_interpretability"
sys.path.append(str(OTHELLO_ROOT/"mechanistic_interpretability"))

from mech_interp_othello_utils import (
    plot_board,
    plot_single_board,
    plot_board_log_probs,
    to_string,
    to_int,
    int_to_label,
    string_to_label,
    OthelloBoardState
)
# %% Download and load SAE stuff
entity = "andyrdt"
project = "othello_gpt_sae"

for artifact in [
    # SAEs trained over all seq positions:
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512:v0",
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024:v0",
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_2048:v3",
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_4096:v4",

    # SAEs trained excluding first seq position:
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512:v2",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024:v2",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_2048:v5",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_4096:v6",

    # feature sparsity data
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512_log_feature_sparsity:v2",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024_log_feature_sparsity:v2",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_2048_log_feature_sparsity:v5",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_4096_log_feature_sparsity:v6",
]:
    artifact_path = f"{entity}/{project}/{artifact}"
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    if not Path(f"./artifacts/{artifact.name}").exists():
        artifact.download()
    
path ="./artifacts/sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512:v2/final_sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512.pt"

model, sparse_autoencoder, activations_loader = LMSparseAutoencoderSessionloader.load_session_from_pretrained(path)
sparse_autoencoder.eval()
entity = "andyrdt"
project = "othello_gpt_sae"
density_path = "artifacts/sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512_log_feature_sparsity:v2/final_sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512_log_feature_sparsity.pt"
log_feature_density = torch.load(density_path)

# %% Load some games
# Load board data as ints (i.e. 0 to 60)
board_seqs_int = torch.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=torch.long)[:500]
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = torch.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=torch.long)[:500]

num_games, length_of_game = board_seqs_int.shape
print("Example game:", board_seqs_int[0])
print("Example game str:", board_seqs_string[0])

print("Number of games:", num_games,)
print("Length of game:", length_of_game)

# %% Compute whether the ground truth is blank
is_blank_ground_truth: Bool[Tensor, "num_games game_length board_pos"] = torch.ones(num_games, length_of_game, 64, dtype=torch.bool)
# mark the middle four squares as occupied always
is_blank_ground_truth[:, :, 27] = 0
is_blank_ground_truth[:, :, 28] = 0
is_blank_ground_truth[:, :, 35] = 0
is_blank_ground_truth[:, :, 36] = 0
for game in tqdm(range(num_games)):
    for seq in range(length_of_game):
        move = board_seqs_string[game, seq]
        # Mark the position as occupied for this and all future moves in this game
        is_blank_ground_truth[game, seq:, move] = 0

# %% Sanity check the is_blank computation
game = 0
game_0_data = is_blank_ground_truth[0]  # Shape will be (game_length, board_pos)

# Number of time steps for game 0
num_time_steps = game_0_data.shape[0]

# Create a figure with a slider
fig = go.Figure()

# Add a heatmap for each time step in game 0
for time_step in range(num_time_steps):
    fig.add_trace(
        px.imshow(game_0_data[time_step].float().view(8, 8), color_continuous_scale="RdBu").data[0]
    )

# Create steps for the slider
steps = []
for i in range(num_time_steps):
    step = dict(
        method="update",
        args=[{"visible": [False] * num_time_steps},  # Hide all traces
              {"title": f"Time step: {i}"}],  # Update the title to show the current time step
        label=f'{i}'
    )
    step["args"][0]["visible"][i] = True  # Show only the i-th trace
    steps.append(step)

# Create and add the slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time step: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title="Game 0 Progress Over Time",
)

# add annotations for the cell index
for i, r in enumerate(range(8)):
    for j, c in enumerate(range(8)):
        fig.add_annotation(
            x=c, y=r,
            text=str(8*r + c),
            showarrow=False,
            font=dict(color="black", size=10),
        )
    

# Show the figure
fig.show()

print("Game 0 moves: ", board_seqs_string[0])
# %%
logits, cache = model.run_with_cache(board_seqs_int[:, :-1].cuda(), names_filter=lambda x: x == sparse_autoencoder.cfg.hook_point)
# %%
sae_input_act_name = list(cache.keys())[0]
sae_input_act = cache[sae_input_act_name]
sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid, feature_acts_pre = sparse_autoencoder(sae_input_act, return_pre=True)
# %%
print("is_blank_ground_truth shape:", is_blank_ground_truth.shape)
print("SAE input act shape:", sae_input_act.shape)
print("SAE output shape:", sae_out.shape)
print("Feature acts shape:", feature_acts.shape)
print("Loss shape:", loss.shape)
print("MSE loss shape:", mse_loss.shape)
print("L1 loss shape:", l1_loss.shape)
print("MSE loss ghost resid shape:", mse_loss_ghost_resid.shape)
print("Feature acts pre shape:", feature_acts_pre.shape)

# %% encode the mean activation of each feature for each board pos
num_features = feature_acts.shape[-1]
is_blank_ground_truth_drop_last = is_blank_ground_truth[:, :-1, :].to(DEVICE)
is_blank_exclude_first_seq_pos = is_blank_ground_truth_drop_last[:, 1:, :]
is_blank_squeezed = is_blank_exclude_first_seq_pos.reshape(-1, 64)
feature_acts_exclude_first_seq_pos = feature_acts_pre[:, 1:, :]
feature_acts_pre_squeezed = feature_acts_exclude_first_seq_pos.flatten(0, 1)
print(is_blank_squeezed.shape)
print(feature_acts_pre_squeezed.shape)
feature_acts_pre_means: Float[Tensor, "num_features board_pos is_blank"] = torch.zeros(num_features, 64, 2).to(DEVICE)
for feature in tqdm(range(num_features)):
    for board_pos in range(64):
        is_blank = is_blank_squeezed[:, board_pos]
        feature_act = feature_acts_pre_squeezed[:, feature]
        feature_acts_pre_means[feature, board_pos, 0] = torch.mean(feature_act[is_blank])
        feature_acts_pre_means[feature, board_pos, 1] = torch.mean(feature_act[~is_blank])

# %% plot the mean diff
feature_acts_diff = feature_acts_pre_means[:, :, 0] - feature_acts_pre_means[:, :, 1]
px.imshow(feature_acts_diff.detach().cpu().numpy(), color_continuous_scale="RdBU", color_continuous_midpoint=0.0, labels=dict(x="Board Position", y="Feature Index"), title="Feature isBlank mean diffs").show()
#%% get the rows which look the most sparse
# remove 27, 28, 35, 36
feature_acts_no_nans = feature_acts_diff.clone()
feature_acts_no_nans[:, 27] = 0
feature_acts_no_nans[:, 28] = 0
feature_acts_no_nans[:, 35] = 0
feature_acts_no_nans[:, 36] = 0
norm = feature_acts_no_nans ** 2
max_norm = torch.max(norm, dim=1).values
norm_occupied_by_max = max_norm / torch.sum(norm, dim=1)
px.line(norm_occupied_by_max.detach().cpu().numpy(), title="Percent of norm of isBlank mean diff on single board position", labels=dict(x="Feature Index", y="Percent occupied by max diff")).show()
k = 40
top_k_features = torch.argsort(norm_occupied_by_max, descending=True)[:k]
print("Top k features:", top_k_features)
feature_act_diffs_top_k = feature_acts_diff[top_k_features]
dead_features = torch.sum(feature_act_diffs_top_k[:, :26], dim=1) == 0
num_dead_features = torch.sum(dead_features).to(int)
# remove dead
feature_act_diffs_top_k = feature_act_diffs_top_k[~dead_features]
print("Dead features:", dead_features)
px.imshow(feature_act_diffs_top_k.detach().cpu().numpy(), color_continuous_scale="RdBU", color_continuous_midpoint=0.0, labels=dict(x="Board Position", y="Feature Index"), title="Top k Feature isBlank mean diffs", y = [f"Feature {i}" for i in top_k_features][num_dead_features:]).show()

# %% find the best feature for each board position
feature_acts_diff_abs = torch.abs(feature_acts_diff)
best_features = torch.argmax(feature_acts_diff_abs, dim=0)
# get the feature distribution for each board position
feature_acts_pre_remove_first = feature_acts_pre[:, 1:, :]
feature_acts_pre_squeezed = feature_acts_pre_remove_first.flatten(0, 1)
is_blank_exclude_first_seq_pos_squeezed = is_blank_exclude_first_seq_pos.flatten(0, 1)
feature_acts_df = pd.DataFrame(feature_acts_pre_squeezed.detach().cpu().numpy(), columns=[f"Feature {i}" for i in range(num_features)])
is_blank_df = pd.DataFrame(is_blank_exclude_first_seq_pos_squeezed.detach().cpu().numpy(), columns=[f"Board Pos {i}" for i in range(64)])
# join the two
feature_acts_df = feature_acts_df.join(is_blank_df, how="left")
feature_acts_df 

# %% plot the best features
for board_pos in range(40, 50):
    best_feature = best_features[board_pos].item()
    is_blank_data = feature_acts_df[feature_acts_df[f"Board Pos {board_pos}"] == True][f"Feature {best_feature}"]
    not_blank_data = feature_acts_df[feature_acts_df[f"Board Pos {board_pos}"] == False][f"Feature {best_feature}"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=is_blank_data, name="is_blank"))
    fig.add_trace(go.Histogram(x=not_blank_data, name="not_blank"))
    fig.update_layout(barmode='overlay')
    fig.update_layout(title=f"Best feature {best_feature} distribution for board pos {board_pos}")
    fig.update_traces(opacity=0.75)
    fig.show()



# %%
