"""

https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab?fbclid=IwAR04OCGu_unvxezvDWkys9_6MJPEnXuu6GSqU6ScrMkAb1bvdSYFOeS35AY
https://github.com/callummcdougall/sae-exercises-mats?fbclid=IwAR3qYAELbyD_x5IAYN4yCDFQzxXHeuH6CwMi_E7g4Qg6G1QXRNAYabQ4xGs

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import einops
import numpy as np
import torch
import torch as t
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider  # , Button
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint

device = "cpu"


def linear_lr(step: int, steps: int):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step: int, steps: int):
    return np.cos(0.5 * np.pi * step / (steps - 1))


def _init_importance(
    importance: Optional[Union[float, Tensor]],
    n_features: int,
    device: str | torch.device,
) -> Tensor:
    if importance is None:
        importance = t.ones(())
    if isinstance(importance, float):
        importance = t.tensor(importance)
    assert isinstance(importance, Tensor)  # pyright can't seem to infer this
    return importance.to(device).broadcast_to(n_features)


@dataclass
class ToyConfig:
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feature_probability: Optional[Union[float, Tensor]] = None
    importance: Optional[Union[float, Tensor]] = None
    device: str | torch.device = device


class HookedToyModel(HookedRootModule, ABC):
    def __init__(self, cfg: ToyConfig, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.cfg = cfg

        feature_probability = cfg.feature_probability
        if feature_probability is None:
            feature_probability = t.ones(())
        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        assert isinstance(
            feature_probability, Tensor
        )  # pyright can't seem to infer this
        self.feature_probability = feature_probability.to(device).broadcast_to(
            cfg.n_features
        )

        self.importance = _init_importance(cfg.importance, cfg.n_features, device)

    @abstractmethod
    def forward(self, features: Tensor, return_type: str) -> Tensor:
        """Forward pass, to be implemented by subclasses"""

    @abstractmethod
    def calculate_loss(self, out: Tensor, batch: Tensor) -> Tensor:
        """Loss calculation, to be implemented by subclasses"""

    def generate_correlated_features(
        self, batch_size: int, n_correlated_pairs: int
    ) -> Float[Tensor, "batch_size features"]:
        """
        Generates a batch of correlated features.
        Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
        """
        feat = t.rand(
            (batch_size, 2 * n_correlated_pairs),
            device=self.W.device,
        )
        feat_set_seeds = t.rand((batch_size, n_correlated_pairs), device=self.W.device)
        feat_set_is_present = feat_set_seeds <= self.feature_probability[[0]]
        feat_is_present = einops.repeat(
            feat_set_is_present,
            "batch features -> batch (features pair)",
            pair=2,
        )
        return t.where(feat_is_present, feat, 0.0)

    def generate_anticorrelated_features(
        self, batch_size: int, n_anticorrelated_pairs: int
    ) -> Float[Tensor, "batch_size features"]:
        """
        Generates a batch of anti-correlated features.
        Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
        """
        feat = t.rand(
            (batch_size, 2 * n_anticorrelated_pairs),
            device=self.W.device,
        )
        feat_set_seeds = t.rand(
            (batch_size, n_anticorrelated_pairs),
            device=self.W.device,
        )
        first_feat_seeds = t.rand(
            (batch_size, n_anticorrelated_pairs),
            device=self.W.device,
        )
        feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[[0]]
        first_feat_is_present = first_feat_seeds <= 0.5
        first_feats = t.where(
            feat_set_is_present & first_feat_is_present,
            feat[:, :n_anticorrelated_pairs],
            0.0,
        )
        second_feats = t.where(
            feat_set_is_present & (~first_feat_is_present),
            feat[:, n_anticorrelated_pairs:],
            0.0,
        )
        return einops.rearrange(
            t.concat([first_feats, second_feats], dim=-1),
            "batch (pair features) -> batch (features pair)",
            pair=2,
        )

    def generate_uncorrelated_features(
        self, batch_size: int, n_uncorrelated: int
    ) -> Float[Tensor, "batch_size features"]:
        """
        Generates a batch of uncorrelated features.
        """
        feat = t.rand((batch_size, n_uncorrelated), device=self.W.device)
        feat_seeds = t.rand((batch_size, n_uncorrelated), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability[[0]]
        return t.where(feat_is_present, feat, 0.0)

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch_size features"]:
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
        return t.cat(data, dim=-1)

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
                progress_bar.set_postfix(loss=loss.item(), lr=step_lr)


class ReluOutputModel(HookedToyModel):
    """
    Anthropic's ReLU Output Model as described in the Toy Models paper:
            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-model
    """

    W: Float[Tensor, "n_hidden n_features"]
    b_final: Tensor
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(self, cfg: ToyConfig, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)

        self.W = nn.Parameter(
            nn.init.xavier_normal_(t.empty((cfg.n_hidden, cfg.n_features)))
        )
        self.b_final = nn.Parameter(t.zeros(cfg.n_features))
        self.to(device)

        # Add and setup hookpoints.
        self.hook_hidden = HookPoint()
        self.hook_out_prebias = HookPoint()
        self.setup()

    def forward(
        self,
        features: Float[Tensor, "... features"],
        return_type: str = "reconstruction",
    ) -> Float[Tensor, "... features"]:
        hidden = self.hook_hidden(
            einops.einsum(
                features,
                self.W,
                "... features, hidden features -> ... hidden",
            )
        )
        out = self.hook_out_prebias(
            einops.einsum(
                hidden,
                self.W,
                "... hidden, hidden features -> ... features",
            )
        )
        reconstructed = F.relu(out + self.b_final)

        if return_type == "loss":
            return self.calculate_loss(reconstructed, features)
        if return_type == "reconstruction":
            return reconstructed
        raise ValueError(f"Unknown return type: {return_type}")

    def calculate_loss(
        self,
        out: Float[Tensor, "batch features"],
        batch: Float[Tensor, "batch features"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """
        error = self.importance * ((batch - out) ** 2)
        return einops.reduce(error, "batch features -> ()", "mean").sum()


class ReluOutputModelCE(ReluOutputModel):
    """
    A variant of Anthropic's ReLU Output Model.
    This model is trained with a Cross Entropy loss instead of MSE loss.
    The model task is to identify which feature has the largest magnitude activation in the input.
    The model has an extra feature dimension which is set to a constant nonzero value,
    which allows for proper classification when all features are zero.
    """

    W: Float[Tensor, "n_hidden n_features"]
    b_final: Tensor
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: ToyConfig,
        device: torch.device = torch.device("cpu"),
        extra_feature_value: float = 1e-6,
    ):
        super().__init__(cfg)
        self.extra_feature_value = extra_feature_value

        self.W = nn.Parameter(
            nn.init.xavier_normal_(t.empty((cfg.n_hidden, cfg.n_features + 1)))
        )
        self.b_final = nn.Parameter(t.zeros(cfg.n_features + 1))
        self.importance = _init_importance(cfg.importance, cfg.n_features + 1, device)
        self.to(device)

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch_size features"]:
        """Adds an extra feature to the batch, which is set to a constant nonzero value."""
        batch = super().generate_batch(batch_size)
        extra_feature = self.extra_feature_value * t.ones((batch_size, 1)).to(
            batch.device
        )
        return t.cat((batch, extra_feature), dim=-1)

    def calculate_loss(
        self,
        out: Float[Tensor, "batch features"],
        batch: Float[Tensor, "batch features"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch.
        Loss is calculated using Cross Entropy loss, where the true probability distribution
        is a one-hot encoding of the feature with the largest magnitude activation in the input.
        Model outputs (raw logits) are weighted by importance before being passed through CE loss.

        Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """
        max_feat_indices = t.argmax(batch, dim=-1)
        return F.cross_entropy(
            (self.importance * out).squeeze(), max_feat_indices.squeeze()
        )


Arr = np.ndarray


def plot_features_in_2d(
    values: Float[Tensor, "timesteps d_hidden feats"],
    colors: Optional[list[Any]] = None,  # shape [timesteps feats]
    title: Optional[str | list[str]] = None,
    subplot_titles: Optional[list[str] | list[list[str]]] = None,  # type: ignore
    save: Optional[str] = None,
    colab: bool = False,
):
    """
    Visualises superposition in 2D.

    If values is 3D, the first dimension is assumed to be timesteps, and an animation is created.
    """
    # Convert values to 3D for consistency
    if values.ndim == 2:
        values = values.unsqueeze(0)
    values = values.transpose(-1, -2)

    # Get dimensions
    n_timesteps, n_features, _ = values.shape

    # If we have a large number of features per plot (i.e. we're plotting projections of data) then use smaller lines
    linewidth, markersize = (1, 4) if (n_features >= 25) else (2, 10)

    # Convert colors to something which has 4D, if it is 3D (i.e. same colors for all timesteps)
    if any(
        [
            colors is None,
            isinstance(colors, list) and isinstance(colors[0], str),
            (isinstance(colors, (Tensor, Arr))) and colors.ndim == 2,
        ]
    ):
        colors = [colors for _ in range(values.shape[0])]
    # Now that colors has length `timesteps` in some sense, we can convert it to lists of strings
    assert colors is not None  # keep pyright happy
    colors = [parse_colors_for_superposition_plot(c, n_features) for c in colors]

    # Same for subplot titles & titles
    if (
        subplot_titles is not None
        and isinstance(subplot_titles, list)
        and isinstance(subplot_titles[0], str)
    ):
        subplot_titles = [
            cast(list[str], subplot_titles) for _ in range(values.shape[0])
        ]
    if title is not None and isinstance(title, str):
        title = [title for _ in range(values.shape[0])]

    # Create a figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # If there are titles, add more spacing for them
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
    if title:
        fig.subplots_adjust(top=0.8)

    # Initialize lines and markers
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal", adjustable="box")
    lines = []
    markers = []
    for feature_idx in range(n_features):
        (line,) = ax.plot([], [], color=colors[0][feature_idx], lw=linewidth)
        (marker,) = ax.plot(
            [],
            [],
            color=colors[0][feature_idx],
            marker="o",
            markersize=markersize,
        )
        lines.append(line)
        markers.append(marker)

    def update(val: float):
        # I think this doesn't work unless I at least reference the nonlocal slider object
        # It works if I use t = int(val), so long as I put something like X = slider.val first. Idk why!
        if n_timesteps > 1:
            _ = slider.val
        t = int(val)
        for feature_idx in range(n_features):
            x, y = values[t, feature_idx].tolist()
            lines[feature_idx].set_data([0, x], [0, y])
            markers[feature_idx].set_data(x, y)
            lines[feature_idx].set_color(colors[t][feature_idx])
            markers[feature_idx].set_color(colors[t][feature_idx])
        if title:
            fig.suptitle(title[t], fontsize=15)
        if subplot_titles:
            ax.set_title(subplot_titles[t], fontsize=12)  # type: ignore
        fig.canvas.draw_idle()

    def play(event: Any):  # noqa: ARG001
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
        from IPython.display import clear_output  # type: ignore

        clear_output()
        return ani
    return None


def parse_colors_for_superposition_plot(
    colors: Optional[Union[Tuple[int, int], List[str], Tensor]],
    n_feats: int,
) -> List[str]:
    """
    There are lots of different ways colors can be represented in the superposition plot.

    This function unifies them all by turning colors into a list of lists of strings, i.e. one color for each instance & feature.
    """
    # If colors is a tensor, we assume it's the importances tensor, and we color according to a viridis color scheme
    # if isinstance(colors, Tensor):
    #     colors = t.broadcast_to(colors, (n_feats))
    #     colors = [
    #         [helper_get_viridis(v.item()) for v in colors_for_this_instance]
    #         for colors_for_this_instance in colors
    #     ]

    # If colors is a tuple of ints, it's interpreted as number of correlated / anticorrelated pairs
    if isinstance(colors, tuple):
        n_corr, n_anti = colors
        n_indep = n_feats - 2 * (n_corr - n_anti)
        return (
            ["blue", "blue", "limegreen", "limegreen", "purple", "purple"][: n_corr * 2]
            + ["red", "red", "orange", "orange", "brown", "brown"][: n_anti * 2]
            + ["black"] * n_indep
        )

    # If colors is a string, make all datapoints that color
    if isinstance(colors, str):
        return [colors] * n_feats

    # Lastly, if colors is None, make all datapoints black
    if colors is None:
        return ["black"] * n_feats

    assert isinstance(colors, list)
    return colors
