"""

https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab?fbclid=IwAR04OCGu_unvxezvDWkys9_6MJPEnXuu6GSqU6ScrMkAb1bvdSYFOeS35AY
https://github.com/callummcdougall/sae-exercises-mats?fbclid=IwAR3qYAELbyD_x5IAYN4yCDFQzxXHeuH6CwMi_E7g4Qg6G1QXRNAYabQ4xGs

"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import einops
import numpy as np
import torch
import torch as t
from IPython.display import clear_output
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider  # , Button
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

device = "cpu"


def linear_lr(step: int, steps: int):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step: int, steps: int):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device: str | torch.device = device,
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None:
            feature_probability = t.ones(())
        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        assert isinstance(
            feature_probability, Tensor
        )  # pyright can't seem to infer this
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_instances, cfg.n_features)
        )
        if importance is None:
            importance = t.ones(())
        if isinstance(importance, float):
            importance = t.tensor(importance)
        assert isinstance(importance, Tensor)  # pyright can't seem to infer this
        self.importance = importance.to(device).broadcast_to(
            (cfg.n_instances, cfg.n_features)
        )

        self.W = nn.Parameter(
            nn.init.xavier_normal_(
                t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))
            )
        )
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)

    def forward(
        self, features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = einops.einsum(
            features,
            self.W,
            "... instances features, instances hidden features -> ... instances hidden",
        )
        out = einops.einsum(
            hidden,
            self.W,
            "... instances hidden, instances hidden features -> ... instances features",
        )
        return F.relu(out + self.b_final)

    # def generate_batch(self, batch_size) -> Float[Tensor, "batch_size instances features"]:
    #     '''
    #     Generates a batch of data. We'll return to this function later when we apply correlations.
    #     '''
    #     feat = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
    #     feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
    #     feat_is_present = feat_seeds <= self.feature_probability
    #     batch = t.where(
    #         feat_is_present,
    #         feat,
    #         t.zeros((), device=self.W.device),
    #     )
    #     return batch

    def generate_correlated_features(
        self, batch_size: int, n_correlated_pairs: int
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of correlated features.
        Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
        """
        feat = t.rand(
            (batch_size, self.cfg.n_instances, 2 * n_correlated_pairs),
            device=self.W.device,
        )
        feat_set_seeds = t.rand(
            (batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.W.device
        )
        feat_set_is_present = feat_set_seeds <= self.feature_probability[:, [0]]
        feat_is_present = einops.repeat(
            feat_set_is_present,
            "batch instances features -> batch instances (features pair)",
            pair=2,
        )
        return t.where(feat_is_present, feat, 0.0)

    def generate_anticorrelated_features(
        self, batch_size: int, n_anticorrelated_pairs: int
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of anti-correlated features.
        Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
        """
        feat = t.rand(
            (batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs),
            device=self.W.device,
        )
        feat_set_seeds = t.rand(
            (batch_size, self.cfg.n_instances, n_anticorrelated_pairs),
            device=self.W.device,
        )
        first_feat_seeds = t.rand(
            (batch_size, self.cfg.n_instances, n_anticorrelated_pairs),
            device=self.W.device,
        )
        feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, [0]]
        first_feat_is_present = first_feat_seeds <= 0.5
        first_feats = t.where(
            feat_set_is_present & first_feat_is_present,
            feat[:, :, :n_anticorrelated_pairs],
            0.0,
        )
        second_feats = t.where(
            feat_set_is_present & (~first_feat_is_present),
            feat[:, :, n_anticorrelated_pairs:],
            0.0,
        )
        return einops.rearrange(
            t.concat([first_feats, second_feats], dim=-1),
            "batch instances (pair features) -> batch instances (features pair)",
            pair=2,
        )

    def generate_uncorrelated_features(
        self, batch_size: int, n_uncorrelated: int
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of uncorrelated features.
        """
        feat = t.rand(
            (batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device
        )
        feat_seeds = t.rand(
            (batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device
        )
        feat_is_present = feat_seeds <= self.feature_probability[:, [0]]
        return t.where(feat_is_present, feat, 0.0)

    def generate_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of data, with optional correlated & anticorrelated features.
        """
        n_uncorrelated = (
            self.cfg.n_features
            - 2 * self.cfg.n_correlated_pairs
            - 2 * self.cfg.n_anticorrelated_pairs
        )
        data = []
        if self.cfg.n_correlated_pairs > 0:
            data.append(
                self.generate_correlated_features(
                    batch_size, self.cfg.n_correlated_pairs
                )
            )
        if self.cfg.n_anticorrelated_pairs > 0:
            data.append(
                self.generate_anticorrelated_features(
                    batch_size, self.cfg.n_anticorrelated_pairs
                )
            )
        if n_uncorrelated > 0:
            data.append(self.generate_uncorrelated_features(batch_size, n_uncorrelated))
        batch = t.cat(data, dim=-1)
        return batch

    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(
            error, "batch instances features -> instances", "mean"
        ).sum()
        return loss

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps), desc="Training Toy Model")

        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / self.cfg.n_instances, lr=step_lr
                )


Arr = np.ndarray


def plot_features_in_2d(
    values: Float[Tensor, "timesteps instances d_hidden feats"],
    colors: Optional[list[Any]] = None,  # shape [timesteps instances feats]
    title: Optional[str | list[str]] = None,
    subplot_titles: Optional[list[str] | list[list[str]]] = None,
    save: Optional[str] = None,
    colab: bool = False,
):
    """
    Visualises superposition in 2D.

    If values is 4D, the first dimension is assumed to be timesteps, and an animation is created.
    """
    # Convert values to 4D for consistency
    if values.ndim == 3:
        values = values.unsqueeze(0)
    values = values.transpose(-1, -2)

    # Get dimensions
    n_timesteps, n_instances, n_features, _ = values.shape

    # If we have a large number of features per plot (i.e. we're plotting projections of data) then use smaller lines
    linewidth, markersize = (1, 4) if (n_features >= 25) else (2, 10)

    # Convert colors to 3D, if it's 2D (i.e. same colors for all instances)
    if isinstance(colors, list) and isinstance(colors[0], str):
        colors = [colors for _ in range(n_instances)]
    # Convert colors to something which has 4D, if it is 3D (i.e. same colors for all timesteps)
    if any(
        [
            colors is None,
            isinstance(colors, list)
            and isinstance(colors[0], list)
            and isinstance(colors[0][0], str),
            (isinstance(colors, Tensor) or isinstance(colors, Arr))
            and colors.ndim == 3,
        ]
    ):
        colors = [colors for _ in range(values.shape[0])]
    # Now that colors has length `timesteps` in some sense, we can convert it to lists of strings
    assert colors is not None  # keep pyright happy
    colors = [
        parse_colors_for_superposition_plot(c, n_instances, n_features) for c in colors
    ]

    # Same for subplot titles & titles
    if subplot_titles is not None:
        if isinstance(subplot_titles, list) and isinstance(subplot_titles[0], str):
            subplot_titles = [
                cast(list[str], subplot_titles) for _ in range(values.shape[0])
            ]
    if title is not None:
        if isinstance(title, str):
            title = [title for _ in range(values.shape[0])]

    # Create a figure and axes
    fig, axs = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5))
    if n_instances == 1:
        axs = [axs]

    # If there are titles, add more spacing for them
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
    if title:
        fig.subplots_adjust(top=0.8)
    # Initialize lines and markers
    lines = []
    markers = []
    for instance_idx, ax in enumerate(axs):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal", adjustable="box")
        instance_lines = []
        instance_markers = []
        for feature_idx in range(n_features):
            (line,) = ax.plot(
                [], [], color=colors[0][instance_idx][feature_idx], lw=linewidth
            )
            (marker,) = ax.plot(
                [],
                [],
                color=colors[0][instance_idx][feature_idx],
                marker="o",
                markersize=markersize,
            )
            instance_lines.append(line)
            instance_markers.append(marker)
        lines.append(instance_lines)
        markers.append(instance_markers)

    def update(val: float):
        # I think this doesn't work unless I at least reference the nonlocal slider object
        # It works if I use t = int(val), so long as I put something like X = slider.val first. Idk why!
        if n_timesteps > 1:
            _ = slider.val
        t = int(val)
        for instance_idx in range(n_instances):
            for feature_idx in range(n_features):
                x, y = values[t, instance_idx, feature_idx].tolist()
                lines[instance_idx][feature_idx].set_data([0, x], [0, y])
                markers[instance_idx][feature_idx].set_data(x, y)
                lines[instance_idx][feature_idx].set_color(
                    colors[t][instance_idx][feature_idx]
                )
                markers[instance_idx][feature_idx].set_color(
                    colors[t][instance_idx][feature_idx]
                )
            if title:
                fig.suptitle(title[t], fontsize=15)
            if subplot_titles:
                axs[instance_idx].set_title(
                    subplot_titles[t][instance_idx], fontsize=12
                )
        fig.canvas.draw_idle()

    def play(event: Any):
        _ = slider.val
        for i in range(n_timesteps):
            update(i)
            slider.set_val(i)
            plt.pause(0.05)
        fig.canvas.draw_idle()

    if n_timesteps > 1:
        # Create the slider
        ax_slider = plt.axes((0.15, 0.05, 0.7, 0.05), facecolor="lightgray")
        slider = Slider(
            ax_slider, "Time", 0, n_timesteps - 1, valinit=0, valfmt="%1.0f"
        )

        # Create the play button
        # ax_button = plt.axes([0.8, 0.05, 0.08, 0.05], facecolor='lightgray')
        # button = Button(ax_button, 'Play')

        # Call the update function when the slider value is changed / button is clicked
        slider.on_changed(update)
        # button.on_clicked(play)

        # Initialize the plot
        play(0)
    else:
        update(0)

    # Save
    if isinstance(save, str):
        ani = FuncAnimation(
            fig, cast(Any, update), frames=n_timesteps, interval=0.04, repeat=False
        )
        ani.save(save, writer="pillow", fps=25)
    elif colab:
        ani = FuncAnimation(
            fig, cast(Any, update), frames=n_timesteps, interval=0.04, repeat=False
        )
        clear_output()
        return ani


def parse_colors_for_superposition_plot(
    colors: Optional[
        Union[Tuple[int, int], List[List[str]], Float[Tensor, "instances feats"]]
    ],
    n_instances: int,
    n_feats: int,
) -> List[List[str]]:
    """
    There are lots of different ways colors can be represented in the superposition plot.

    This function unifies them all by turning colors into a list of lists of strings, i.e. one color for each instance & feature.
    """
    # If colors is a tensor, we assume it's the importances tensor, and we color according to a viridis color scheme
    # if isinstance(colors, Tensor):
    #     colors = t.broadcast_to(colors, (n_instances, n_feats))
    #     colors = [
    #         [helper_get_viridis(v.item()) for v in colors_for_this_instance]
    #         for colors_for_this_instance in colors
    #     ]

    # If colors is a tuple of ints, it's interpreted as number of correlated / anticorrelated pairs
    if isinstance(colors, tuple):
        n_corr, n_anti = colors
        n_indep = n_feats - 2 * (n_corr - n_anti)
        return [
            ["blue", "blue", "limegreen", "limegreen", "purple", "purple"][: n_corr * 2]
            + ["red", "red", "orange", "orange", "brown", "brown"][: n_anti * 2]
            + ["black"] * n_indep
            for _ in range(n_instances)
        ]

    # If colors is a string, make all datapoints that color
    elif isinstance(colors, str):
        return [[colors] * n_feats] * n_instances

    # Lastly, if colors is None, make all datapoints black
    elif colors is None:
        return [["black"] * n_feats] * n_instances

    assert isinstance(colors, list)
    return colors
