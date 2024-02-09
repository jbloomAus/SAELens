import gzip
import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from eindex import eindex
from IPython.display import HTML, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

Arr = np.ndarray

from sae_analysis.visualizer.html_fns import (
    CSS,
    HTML_HOVERTEXT_SCRIPT,
    generate_histograms,
    generate_seq_html,
    generate_tables_html,
)
from sae_analysis.visualizer.model_fns import AutoEncoder
from sae_analysis.visualizer.utils_fns import (
    TopK,
    extract_and_remove_scripts,
    k_largest_indices,
    merge_lists,
    random_range_indices,
    to_str_tokens,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HistogramData:
    """
    Class for storing all the data necessary to construct a histogram (because e.g.
    for a vector with length `d_vocab`, we don't need to store it all!).

    This is initialised with a tensor of data, and it automatically calculates & stores
    the 3 things which are needed: bar_heights, bar_values, tick_vals.

    This isn't a dataclass, because the things we hold at the end are not the same as
    the things we start with!
    """

    def __init__(self, data: Tensor, n_bins: int, tickmode: str):
        if data.numel() == 0:
            self.bar_heights = []
            self.bar_values = []
            self.tick_vals = []
            return

        # Get min and max of data
        max_value = data.max().item()
        min_value = data.min().item()

        # divide range up into 40 bins
        bin_size = (max_value - min_value) / n_bins
        bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
        # calculate the heights of each bin
        bar_heights = torch.histc(data, bins=n_bins)
        bar_values = bin_edges[:-1] + bin_size / 2

        # choose tickvalues (super hacky and terrible, should improve this)
        assert tickmode in ["ints", "5 ticks"]

        if tickmode == "ints":
            top_tickval = int(max_value)
            tick_vals = torch.arange(0, top_tickval + 1, 1).tolist()
        elif tickmode == "5 ticks":
            # ticks chosen in multiples of 0.1, so we have 3 on the longer side
            if max_value > -min_value:
                tickrange = 0.1 * int(1e-4 + max_value / (3 * 0.1))
                num_positive_ticks = 3
                num_negative_ticks = int(-min_value / tickrange)
            else:
                tickrange = 0.1 * int(1e-4 + -min_value / (3 * 0.1))
                num_negative_ticks = 3
                num_positive_ticks = int(max_value / tickrange)
            tick_vals = merge_lists(
                reversed(
                    [-tickrange * i for i in range(1, 1 + num_negative_ticks)]
                ),  # negative values (if exist)
                [0],  # zero (always is a tick)
                [
                    tickrange * i for i in range(1, 1 + num_positive_ticks)
                ],  # positive values
            )

        self.bar_heights = bar_heights.tolist()
        self.bar_values = bar_values.tolist()
        self.tick_vals = tick_vals


@dataclass
class SequenceData:
    """
    Class to store data for a given sequence, which will be turned into a JavaScript visulisation.

    Before hover:
        str_tokens: list of string tokens in the sequence
        feat_acts: sizes of activations on this sequence
        change_in_loss: effect on loss of this feature, on this token
        repeat: whether to have 🔁

    On hover:
        top5_str_tokens: list of the top 5 logit-boosted tokens by this feature
        top5_logit_changes: list of the corresponding 5 changes in logits for those tokens
        bottom5_str_tokens: list of the bottom 5 logit-boosted tokens by this feature
        bottom5_logit_changes: list of the corresponding 5 changes in logits for those tokens
    """

    token_ids: List[str]
    feat_acts: List[float]
    contribution_to_loss: List[float]
    repeat: bool
    top5_token_ids: List[List[str]]
    top5_logit_contributions: List[List[float]]
    bottom5_token_ids: List[List[str]]
    bottom5_logit_contributions: List[List[float]]

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return f"SequenceData({''.join(self.token_ids)})"

    def __post_init__(self):
        """Filters down the data, by deleting the "on hover" information if the activations are zero."""
        self.top5_logit_contributions, self.top5_token_ids = self._filter(
            self.top5_logit_contributions, self.top5_token_ids
        )
        self.bottom5_logit_contributions, self.bottom5_token_ids = self._filter(
            self.bottom5_logit_contributions, self.bottom5_token_ids
        )

    def _filter(self, float_list: List[List[float]], int_list: List[List[str]]):
        float_list = [[f for f in floats if f != 0] for floats in float_list]
        int_list = [
            [i for i, f in zip(ints, floats)]
            for ints, floats in zip(int_list, float_list)
        ]
        return float_list, int_list


class SequenceDataBatch:
    """
    Class to store a list of SequenceData objects at once, by passing in tensors or objects
    with an extra dimension at the start.

    Note, I'll be creating these objects by passing in objects which are either 2D (k seq_len)
    or 3D (k seq_len top5), but which are all lists (of strings/ints/floats).

    """

    def __init__(self, **kwargs):
        self.seqs = [
            SequenceData(
                token_ids=kwargs["token_ids"][k],
                feat_acts=kwargs["feat_acts"][k],
                contribution_to_loss=kwargs["contribution_to_loss"][k],
                repeat=kwargs["repeat"],
                top5_token_ids=kwargs["top5_token_ids"][k],
                top5_logit_contributions=kwargs["top5_logit_contributions"][k],
                bottom5_token_ids=kwargs["bottom5_token_ids"][k],
                bottom5_logit_contributions=kwargs["bottom5_logit_contributions"][k],
            )
            for k in range(len(kwargs["token_ids"]))
        ]

    def __getitem__(self, idx: int) -> SequenceData:
        return self.seqs[idx]

    def __len__(self) -> int:
        return len(self.seqs)

    def __str__(self) -> str:
        return "\n".join([str(seq) for seq in self.seqs])


@dataclass
class FeatureData:
    """
    Class to store all data for a feature that will be used in the visualization.

    Also has a bunch of methods to create visualisations. So this is the main important class.

    The biggest arg is `sequence_data`, it's explained below. The other args are individual, and are used
    to construct the left-hand visualisations.

    Args for the right-hand sequences:

        sequence_data: Dict[str, SequenceDataBatch]
            This contains all the data which will be used to construct the right-hand visualisation.
            Each key is a group name (there are 12 in total: top, bottom, 10 quantiles), and each value
            is a SequenceDataBatch object (i.e. it contains a batch of SequenceData objects, one for each
            sequence in the group). See these classes for more on how these are used.

    Args for the middle column:

        top10_logits: Tuple[TopK, TopK]
            Contains the most neg / pos 10 logits, used for the logits table

        logits_histogram_data: HistogramData,
            Contains the data for making the logits histogram (see `html_fns.py` for how this is used)

        frequencies_histogram_data: HistogramData,
            Contains the data for making the frequencies histogram (see `html_fns.py` for how this is used)

        frac_nonzero: float
            Also used for frequencies histogram, this is the fraction of activations which are non-zero

    Args for the left-hand column

        neuron_alignment: Tuple[TopK, Tensor]
            first element is the topk aligned neurons (i.e. argmax on decoder weights)
            second element is the fraction of L1 norm this neuron makes up, in this decoder weight vector.

        neurons_correlated: Tuple[TopK, TopK]
            the topk neurons most correlated with each other, i.e. this feature has (N,) activations and
            the neurons have (d_mlp, N) activations on these tokens, where N = batch_size * seq_len, and
            we find the neuron (column of second tensor) with highest correlation. Contains Pearson &
            Cosine sim (difference is that Pearson centers weights first).

        b_features_correlated: Tuple[TopK, TopK]
            same datatype as neurons_correlated, but now we're looking at this feature's (N,) activations
            and comparing them to the (h, N) activations of the encoder-B features (where h is the hidden
            size of the encoder, i.e. it's d_mlp*8 = 512*8 = 4096 in this case).

    Args (non-data-containing):

        model: HookedTransformer
            The thing you're actually doing forward passes through, and finding features of

        encoder: AutoEncoder
            The encoder of the model, which you're using to find features

        buffer: Tuple[int, int]
            This determines how mnay tokens you'll have on either side, for the right-hand visualisations.
            By default it's (5, 5).

        n_groups, first_group_size, other_groups_size
            All params to determine size of the sequences in right hand of visualisation.
    """

    sequence_data: Dict[str, SequenceDataBatch]

    top10_logits: Tuple[TopK, TopK]
    logits_histogram_data: HistogramData
    frequencies_histogram_data: HistogramData
    frac_nonzero: float

    neuron_alignment: Tuple[TopK, Arr]
    neurons_correlated: Tuple[TopK, TopK]
    b_features_correlated: Tuple[TopK, TopK]

    vocab_dict: Dict[int, str]
    buffer: Tuple[int, int] = (5, 5)
    n_groups: int = 10
    first_group_size: int = 20
    other_groups_size: int = 5

    def return_save_dict(self) -> dict:
        """Returns a dict we use for saving (pickling)."""
        return {k: v for k, v in self.__dict__.items() if k not in ["vocab_dict"]}

    @classmethod
    def load_from_save_dict(self, save_dict, vocab_dict):
        """Loads this object from a dict (e.g. from a pickle file)."""
        return FeatureData(**save_dict, vocab_dict=vocab_dict)

    @classmethod
    def save_batch(
        cls,
        batch: Dict[int, "FeatureData"],
        filename: str,
        save_type: Literal["pkl", "gzip"],
    ) -> None:
        """Saves a batch of FeatureData objects to a pickle file."""
        assert (
            "." not in filename
        ), "You should pass in the filename without the extension."
        filename = filename + ".pkl" if (save_type == "pkl") else filename + ".pkl.gz"
        save_obj = {k: v.return_save_dict() for k, v in batch.items()}
        if save_type == "pkl":
            with open(filename, "wb") as f:
                pickle.dump(save_obj, f)
        elif save_type == "gzip":
            with gzip.open(filename, "wb") as f:
                pickle.dump(save_obj, f)
        return filename

    @classmethod
    def load_batch(
        cls,
        filename: str,
        save_type: Literal["pkl", "gzip"],
        vocab_dict: Dict[int, str],
        feature_idx: Optional[int] = None,
    ) -> Union["FeatureData", Dict[int, "FeatureData"]]:
        """Loads a batch of FeatureData objects from a pickle file."""
        assert (
            "." not in filename
        ), "You should pass in the filename without the extension."
        filename = (
            filename + ".pkl" if save_type.startswith("pkl") else filename + ".pkl.gz"
        )
        if save_type.startswith("pkl"):
            with open(filename, "rb") as f:
                save_obj = pickle.load(f)
        elif save_type == "gzip":
            with gzip.open(filename, "rb") as f:
                save_obj = pickle.load(f)

        if feature_idx is None:
            return {
                k: FeatureData.load_from_save_dict(v, vocab_dict)
                for k, v in save_obj.items()
            }
        else:
            return FeatureData.load_from_save_dict(save_obj[feature_idx], vocab_dict)

    def save(self, filename: str, save_type: Literal["pkl", "gzip"]) -> None:
        """Saves this object to a pickle file (we don't need to save the model and encoder too, just the data)."""
        assert (
            "." not in filename
        ), "You should pass in the filename without the extension."
        filename = filename + ".pkl" if (save_type == "pkl") else filename + ".pkl.gz"
        save_obj = self.return_save_dict()
        if save_type.startswith("pkl"):
            with open(filename, "wb") as f:
                pickle.dump(save_obj, f)
        elif save_type == "gzip":
            with gzip.open(filename, "wb") as f:
                pickle.dump(save_obj, f)
        return filename

    def __str__(self) -> str:
        num_sequences = sum([len(batch) for batch in self.sequence_data.values()])
        return f"FeatureData(num_sequences={num_sequences})"

    def get_sequences_html(self) -> str:
        sequences_html_dict = {}

        for group_name, sequences in self.sequence_data.items():
            full_html = f"<h4>{group_name}</h4>"  # style="padding-left:25px;"

            for seq in sequences:
                html_output = generate_seq_html(
                    self.vocab_dict,
                    token_ids=seq.token_ids,
                    feat_acts=seq.feat_acts,
                    contribution_to_loss=seq.contribution_to_loss,
                    bold_idx=self.buffer[
                        0
                    ],  # e.g. the 6th item, with index 5, if buffer=(5, 5)
                    is_repeat=seq.repeat,
                    pos_ids=seq.top5_token_ids,
                    neg_ids=seq.bottom5_token_ids,
                    pos_val=seq.top5_logit_contributions,
                    neg_val=seq.bottom5_logit_contributions,
                )
                full_html += html_output

            sequences_html_dict[group_name] = full_html

        # Now, wrap all the values of this dictionary into grid-items: (top, groups of 3 for middle, bottom)
        html_top, html_bottom, *html_sampled = sequences_html_dict.values()
        sequences_html = ""
        sequences_html += f"<div class='grid-item'>{html_top}</div>"
        while len(html_sampled) > 0:
            L = min(3, len(html_sampled))
            html_next, html_sampled = html_sampled[:L], html_sampled[L:]
            sequences_html += "<div class='grid-item'>" + "".join(html_next) + "</div>"
        sequences_html += f"<div class='grid-item'>{html_bottom}</div>"

        return sequences_html + HTML_HOVERTEXT_SCRIPT

    def get_tables_html(self) -> Tuple[str, str]:
        bottom10_logits, top10_logits = self.top10_logits

        # Get the negative and positive background values (darkest when equals max abs). Easier when in tensor form
        max_value = max(
            np.absolute(bottom10_logits.values).max(),
            np.absolute(top10_logits.values).max(),
        )
        neg_bg_values = np.absolute(bottom10_logits.values) / max_value
        pos_bg_values = np.absolute(top10_logits.values) / max_value

        # Generate the html
        left_tables_html, logit_tables_html = generate_tables_html(
            neuron_alignment_indices=self.neuron_alignment[0].indices.tolist(),
            neuron_alignment_values=self.neuron_alignment[0].values.tolist(),
            neuron_alignment_l1=self.neuron_alignment[1].tolist(),
            correlated_neurons_indices=self.neurons_correlated[0].indices.tolist(),
            correlated_neurons_pearson=self.neurons_correlated[0].values.tolist(),
            correlated_neurons_l1=self.neurons_correlated[1].values.tolist(),
            correlated_features_indices=None,  # self.b_features_correlated[0].indices.tolist(),
            correlated_features_pearson=None,  # self.b_features_correlated[0].values.tolist(),
            correlated_features_l1=None,  # self.b_features_correlated[1].values.tolist(),
            neg_str=to_str_tokens(self.vocab_dict, bottom10_logits.indices),
            neg_values=bottom10_logits.values.tolist(),
            neg_bg_values=neg_bg_values,
            pos_str=to_str_tokens(self.vocab_dict, top10_logits.indices),
            pos_values=top10_logits.values.tolist(),
            pos_bg_values=pos_bg_values,
        )

        # Return both items (we'll be wrapping them in 'grid-item' later)
        return left_tables_html, logit_tables_html

    def get_histograms(self) -> Tuple[str, str]:
        """
        From the histogram data, returns the actual histogram HTML strings.
        """
        frequencies_histogram, logits_histogram = generate_histograms(
            self.frequencies_histogram_data, self.logits_histogram_data
        )
        return (
            f"<h4>ACTIVATIONS<br>DENSITY = {self.frac_nonzero:.3%}</h4><div>{frequencies_histogram}</div>",
            f"<div>{logits_histogram}</div>",
        )

    def get_all_html(self, debug: bool = False, split_scripts: bool = False) -> str:
        # Get the individual HTML
        left_tables_html, logit_tables_html = self.get_tables_html()
        sequences_html = self.get_sequences_html()
        frequencies_histogram, logits_histogram = self.get_histograms()

        # Wrap them in grid-items
        left_tables_html = f"<div class='grid-item'>{left_tables_html}</div>"
        logit_tables_html = f"<div class='grid-item'>{frequencies_histogram}{logit_tables_html}{logits_histogram}</div>"

        # Create the full HTML string, by wrapping everything together
        html_string = f"""
<style>
{CSS}
</style>

<div class='grid-container'>

    {left_tables_html}
    {logit_tables_html}
    {sequences_html}

</div>
"""
        # idk why this bug is here, for representing newlines the wrong way
        html_string = html_string.replace("Ċ", "&bsol;n")

        if debug:
            display(HTML(html_string))

        if split_scripts:
            scripts, html_string = extract_and_remove_scripts(html_string)
            return scripts, html_string
        else:
            return html_string


class BatchedCorrCoef:
    """
    This class allows me to calculate corrcoef (both Pearson and cosine sim) between two
    batches of vectors without needing to store them all in memory.

    x.shape = (X, N), y.shape = (Y, N), and we calculate every pairwise corrcoef between
    the X*Y pairs of vectors.

    It's based on the following formulas (for vectors).

        cos_sim(x, y) = xy_sum / ((x2_sum ** 0.5) * (y2_sum ** 0.5))

        pearson_corrcoef(x, y) = num / denom

            num = n * xy_sum - x_sum * y_sum
            denom = (n * x2_sum - x_sum ** 2) ** 0.5 * (n * y2_sum - y_sum ** 2) ** 0.5

        ...and all these quantities (x_sum, xy_sum, etc) can be tracked on a rolling basis.
    """

    def __init__(self):
        self.n = 0
        self.x_sum = 0
        self.y_sum = 0
        self.xy_sum = 0
        self.x2_sum = 0
        self.y2_sum = 0

    def update(self, x: Float[Tensor, "X N"], y: Float[Tensor, "Y N"]):  # noqa
        assert x.ndim == 2 and y.ndim == 2, "Both x and y should be 2D"
        assert (
            x.shape[-1] == y.shape[-1]
        ), "x and y should have the same size in the last dimension"

        self.n += x.shape[-1]
        self.x_sum += einops.reduce(x, "X N -> X", "sum")
        self.y_sum += einops.reduce(y, "Y N -> Y", "sum")
        self.xy_sum += einops.einsum(x, y, "X N, Y N -> X Y")
        self.x2_sum += einops.reduce(x**2, "X N -> X", "sum")
        self.y2_sum += einops.reduce(y**2, "Y N -> Y", "sum")

    def corrcoef(self) -> Tuple[Float[Tensor, "X Y"], Float[Tensor, "X Y"]]:  # noqa
        cossim_numer = self.xy_sum
        cossim_denom = torch.sqrt(torch.outer(self.x2_sum, self.y2_sum)) + 1e-6
        cossim = cossim_numer / cossim_denom

        pearson_numer = self.n * self.xy_sum - torch.outer(self.x_sum, self.y_sum)
        pearson_denom = (
            torch.sqrt(
                torch.outer(
                    self.n * self.x2_sum - self.x_sum**2,
                    self.n * self.y2_sum - self.y_sum**2,
                )
            )
            + 1e-6
        )
        pearson = pearson_numer / pearson_denom

        return pearson, cossim

    def topk(self, k: int, largest: bool = True) -> Tuple[TopK, TopK]:
        """Returns the topk corrcoefs, using Pearson (and taking this over the y-tensor)"""
        pearson, cossim = self.corrcoef()
        X, Y = cossim.shape
        # Get pearson topk by actually taking topk
        pearson_topk = TopK(pearson.topk(dim=-1, k=k, largest=largest))  # shape (X, k)
        # Get cossim topk by indexing into cossim with the indices of the pearson topk: cossim[X, pearson_indices[X, k]]
        cossim_values = eindex(cossim, pearson_topk.indices, "X [X k]")
        cossim_topk = TopK((cossim_values, pearson_topk.indices))
        return pearson_topk, cossim_topk


@torch.inference_mode()
def get_feature_data(
    encoder: AutoEncoder,
    # encoder_B: AutoEncoder,
    model: HookedTransformer,
    hook_point: str,
    hook_point_layer: int,
    hook_point_head_index: Optional[int],
    tokens: Int[Tensor, "batch seq"],  # noqa
    feature_idx: Union[int, List[int]],
    max_batch_size: Optional[int] = None,
    left_hand_k: int = 3,
    buffer: Tuple[int, int] = (5, 5),
    n_groups: int = 10,
    first_group_size: int = 20,
    other_groups_size: int = 5,
    verbose: bool = False,
) -> Dict[int, FeatureData]:
    """
    Gets data that will be used to create the sequences in the HTML visualisation.

    Args:
        feature_idx: int
            The identity of the feature we're looking at (i.e. we slice the weights of the encoder). A list of
            features is accepted (the result will be a list of FeatureData objects).
        max_batch_size: Optional[int]
            Optionally used to chunk the tokens, if it's a large batch

        left_hand_k: int
            The number of items in the left-hand tables (by default they're all 3).
        buffer: Tuple[int, int]
            The number of tokens on either side of the feature, for the right-hand visualisation.

    Returns object of class FeatureData (see that class's docstring for more info).
    """
    t0 = time.time()

    model.reset_hooks(including_permanent=True)
    device = model.cfg.device

    # Make feature_idx a list, for convenience
    if isinstance(feature_idx, int):
        feature_idx = [feature_idx]
    n_feats = len(feature_idx)

    # Chunk the tokens, for less memory usage
    all_tokens = (tokens,) if max_batch_size is None else tokens.split(max_batch_size)
    all_tokens = [tok.to(device) for tok in all_tokens]

    # Create lists to store data, which we'll eventually concatenate & create the FeatureData object from
    all_feat_acts = []
    all_resid_post = []

    # Create objects to store the rolling correlation coefficients
    corrcoef_neurons = BatchedCorrCoef()
    # corrcoef_encoder_B = BatchedCorrCoef()

    # Get encoder & decoder directions
    feature_act_dir = encoder.W_enc[:, feature_idx]  # (d_in, feats)
    feature_bias = encoder.b_enc[feature_idx]  # (feats,)
    feature_out_dir = encoder.W_dec[feature_idx]  # (feats, d_in)

    if "resid_pre" in hook_point:
        feature_mlp_out_dir = feature_out_dir  # (feats, d_model)
    elif "resid_post" in hook_point:
        feature_mlp_out_dir = (
            feature_out_dir @ model.W_out[hook_point_layer]
        )  # (feats, d_model)
    elif "hook_q" in hook_point:
        # unembed proj onto residual stream
        feature_mlp_out_dir = (
            feature_out_dir @ model.W_Q[hook_point_layer, hook_point_head_index].T
        )  # (feats, d_model)ß
    assert (
        feature_act_dir.T.shape
        == feature_out_dir.shape
        == (len(feature_idx), encoder.cfg.d_in)
    )

    t1 = time.time()

    # ! Define hook function to perform feature ablation

    def hook_fn_act_post(
        act_post: Float[Tensor, "batch seq d_mlp"], hook: HookPoint  # noqa
    ):  # noqa
        """
        Encoder has learned x^j \approx b + \sum_i f_i(x^j)d_i where:
            - f_i are the feature activations
            - d_i are the feature output directions

        This hook function stores all the information we'll need later on. It doesn't actually perform feature ablation, because
        if we did this, then we'd have to run a different fwd pass for every feature, which is super wasteful! But later, we'll
        calculate the effect of feature ablation,  i.e. x^j <- x^j - f_i(x^j)d_i for i = feature_idx, only on the tokens we care
        about (the ones which will appear in the visualisation).
        """
        # Calculate & store the feature activations (we need to store them so we can get the right-hand visualisations later)
        x_cent = act_post - encoder.b_dec
        feat_acts_pre = einops.einsum(
            x_cent, feature_act_dir, "batch seq d_mlp, d_mlp feats -> batch seq feats"
        )
        feat_acts = F.relu(feat_acts_pre + feature_bias)
        all_feat_acts.append(feat_acts)

        # Update the CorrCoef object between feature activation & neurons
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(act_post, "batch seq d_mlp -> d_mlp (batch seq)"),
        )

        # Calculate encoder-B feature activations (we don't need to store them, cause it's just for the left-hand visualisations)
        # x_cent_B = act_post - encoder_B.b_dec
        # feat_acts_pre_B = einops.einsum(x_cent_B, encoder_B.W_enc, "batch seq d_mlp, d_mlp d_hidden -> batch seq d_hidden")
        # feat_acts_B = F.relu(feat_acts_pre_B + encoder.b_enc)

        # Update the CorrCoef object between feature activation & encoder-B features
        # corrcoef_encoder_B.update(
        #     einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
        #     einops.rearrange(feat_acts_B, "batch seq d_hidden -> d_hidden (batch seq)"),
        # )

    def hook_fn_query(
        hook_q: Float[Tensor, "batch seq n_head d_head"], hook: HookPoint  # noqa
    ):
        """

        Replace act_post with projection of query onto the resid by W_k^T.
        Encoder has learned x^j \approx b + \sum_i f_i(x^j)d_i where:
            - f_i are the feature activations
            - d_i are the feature output directions

        This hook function stores all the information we'll need later on. It doesn't actually perform feature ablation, because
        if we did this, then we'd have to run a different fwd pass for every feature, which is super wasteful! But later, we'll
        calculate the effect of feature ablation,  i.e. x^j <- x^j - f_i(x^j)d_i for i = feature_idx, only on the tokens we care
        about (the ones which will appear in the visualisation).
        """
        # Calculate & store the feature activations (we need to store them so we can get the right-hand visualisations later)
        hook_q = hook_q[:, :, hook_point_head_index]
        x_cent = hook_q - encoder.b_dec
        feat_acts_pre = einops.einsum(
            x_cent, feature_act_dir, "batch seq d_mlp, d_mlp feats -> batch seq feats"
        )
        feat_acts = F.relu(feat_acts_pre + feature_bias)
        all_feat_acts.append(feat_acts)

        # project this back up to resid stream size.
        act_resid_proj = hook_q @ model.W_Q[hook_point_layer, hook_point_head_index].T

        # Update the CorrCoef object between feature activation & neurons
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(
                act_resid_proj, "batch seq d_model -> d_model (batch seq)"
            ),
        )

    def hook_fn_resid_post(
        resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint  # noqa
    ):
        """
        This hook function stores the residual activations, which we'll need later on to calculate the effect of feature ablation.
        """
        all_resid_post.append(resid_post)

    # Run the model without hook (to store all the information we need, not to actually return anything)

    # ! Run the forward passes (triggering the hooks), concat all results
    iterator = tqdm(all_tokens, desc="Storing model activations")
    if "resid_pre" in hook_point:
        for _tokens in iterator:
            model.run_with_hooks(
                _tokens,
                return_type=None,
                fwd_hooks=[
                    (hook_point, hook_fn_act_post),
                    (
                        utils.get_act_name("resid_pre", hook_point_layer),
                        hook_fn_resid_post,
                    ),
                ],
            )
    # If we are using MLP activations, then we'd want this one.
    elif "resid_post" in hook_point:
        for _tokens in iterator:
            model.run_with_hooks(
                _tokens,
                return_type=None,
                fwd_hooks=[
                    (utils.get_act_name("post", hook_point_layer), hook_fn_act_post),
                    (
                        utils.get_act_name("resid_post", hook_point_layer),
                        hook_fn_resid_post,
                    ),
                ],
            )
    elif "hook_q" in hook_point:
        iterator = tqdm(all_tokens, desc="Storing model activations")
        for _tokens in iterator:
            model.run_with_hooks(
                _tokens,
                return_type=None,
                fwd_hooks=[
                    (hook_point, hook_fn_query),
                    (
                        utils.get_act_name("resid_post", hook_point_layer),
                        hook_fn_resid_post,
                    ),
                ],
            )

    t2 = time.time()

    # Stack the results, and check shapes (remember that we don't get loss for the last token)
    feat_acts = torch.concatenate(all_feat_acts)  # [batch seq feats]
    resid_post = torch.concatenate(all_resid_post)  # [batch seq d_model]
    assert feat_acts[:, :-1].shape == tokens[:, :-1].shape + (len(feature_idx),)

    t3 = time.time()

    # ! Calculate all data for the left-hand column visualisations, i.e. the 3 size-3 tables
    # First, get the logits of this feature
    logits = einops.einsum(
        feature_mlp_out_dir,
        model.W_U,
        "feats d_model, d_model d_vocab -> feats d_vocab",
    )
    # Second, get the neurons most aligned with this feature (based on output weights)
    top3_neurons_aligned = TopK(
        feature_out_dir.topk(dim=-1, k=left_hand_k, largest=True)
    )
    pct_of_l1 = (
        np.absolute(top3_neurons_aligned.values)
        / feature_out_dir.abs().sum(dim=-1, keepdim=True).cpu().numpy()
    )
    # Third, get the neurons most correlated with this feature (based on input weights)
    top_correlations_neurons = corrcoef_neurons.topk(k=left_hand_k, largest=True)
    # Lastly, get most correlated weights in B features
    # top_correlations_encoder_B = corrcoef_encoder_B.topk(k=left_hand_k, largest=True)

    t4 = time.time()

    # ! Calculate all data for the right-hand visualisations, i.e. the sequences
    # TODO - parallelize this (it could probably be sped up by batching indices & doing all sequences at once, although those would be large tensors)
    # We do this in 2 steps:
    #   (1) get the indices per group, from the feature activations, for each of the 12 groups (top, bottom, 10 quantiles)
    #   (2) get a batch of SequenceData objects per group. This usually involves using eindex (i.e. indexing into the `tensors`
    #       tensor with the group indices), and it also requires us to calculate the effect of ablations (using feature activations
    #       and the clean residual stream values).

    sequence_data_list = []

    iterator = (
        range(n_feats)
        if not (verbose)
        else tqdm(range(n_feats), desc="Getting sequence data", leave=False)
    )

    for feat in iterator:
        _feat_acts = feat_acts[..., feat]  # [batch seq]

        # (1)
        indices_dict = {
            f"TOP ACTIVATIONS<br>MAX = {_feat_acts.max():.3f}": k_largest_indices(
                _feat_acts, k=first_group_size, largest=True
            ),
            f"BOTTOM ACTIVATIONS<br>MIN = {_feat_acts.min():.3f}": k_largest_indices(
                _feat_acts, k=first_group_size, largest=False
            ),
        }

        quantiles = torch.linspace(0, _feat_acts.max(), n_groups + 1)
        for i in range(n_groups - 1, -1, -1):
            lower, upper = quantiles[i : i + 2]
            pct = ((_feat_acts >= lower) & (_feat_acts <= upper)).float().mean()
            indices = random_range_indices(
                _feat_acts, (lower, upper), k=other_groups_size
            )
            indices_dict[
                f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
            ] = indices

        # Concat all the indices together (in the next steps we do all groups at once)
        indices_full = torch.concat(list(indices_dict.values()))

        # (2)
        # ! We further split (2) up into 3 sections:
        #   (A) calculate the indices we'll use for this group (we need to get a buffer on either side of the target token for each seq),
        #       i.e. indices[..., 0] = shape (g, buf) contains the batch indices of the sequences, and indices[..., 1] = contains seq indices
        #   (B) index into all our tensors to get the relevant data (this includes calculating the effect of ablation)
        #   (C) construct the SequenceData objects, in the form of a SequenceDataBatch object

        # (A)
        # For each token index [batch, seq], we actually want [[batch, seq-buffer[0]], ..., [batch, seq], ..., [batch, seq+buffer[1]]]
        # We get one extra dimension at the start, because we need to see the effect on loss of the first token
        buffer_tensor = torch.arange(
            -buffer[0] - 1, buffer[1] + 1, device=indices_full.device
        )
        indices_full = einops.repeat(
            indices_full, "g two -> g buf two", buf=buffer[0] + buffer[1] + 2
        )
        indices_full = torch.stack(
            [indices_full[..., 0], indices_full[..., 1] + buffer_tensor], dim=-1
        ).cpu()

        # (B)
        # Template for indexing is new_tensor[k, seq] = tensor[indices_full[k, seq, 1], indices_full[k, seq, 2]], sometimes there's an extra dim at the end
        tokens_group = eindex(tokens, indices_full[:, 1:], "[g buf 0] [g buf 1]")
        feat_acts_group = eindex(_feat_acts, indices_full, "[g buf 0] [g buf 1]")
        resid_post_group = eindex(
            resid_post, indices_full, "[g buf 0] [g buf 1] d_model"
        )

        # From these feature activations, get the actual contribution to the final value of the residual stream
        resid_post_feature_effect = einops.einsum(
            feat_acts_group,
            feature_mlp_out_dir[feat],
            "g buf, d_model -> g buf d_model",
        )
        # Get the resulting new logits (by subtracting this effect from resid_post, then applying layernorm & unembedding)
        new_resid_post = resid_post_group - resid_post_feature_effect
        new_logits = (
            new_resid_post / new_resid_post.std(dim=-1, keepdim=True)
        ) @ model.W_U
        orig_logits = (
            resid_post_group / resid_post_group.std(dim=-1, keepdim=True)
        ) @ model.W_U

        # Get the top5 & bottom5 changes in logits
        # note - changes in logits are for hovering over predict-ING token, so it should align w/ tokens_group, hence we slice [:, 1:]
        contribution_to_logprobs = orig_logits.log_softmax(
            dim=-1
        ) - new_logits.log_softmax(dim=-1)
        top5_contribution_to_logits = TopK(
            contribution_to_logprobs[:, :-1].topk(k=5, largest=True)
        )
        bottom5_contribution_to_logits = TopK(
            contribution_to_logprobs[:, :-1].topk(k=5, largest=False)
        )
        # Get the change in loss (which is negative of change of logprobs for correct token)
        # note - changes in loss are for underlining predict-ED token, hence we slice [:, :-1]
        contribution_to_loss = eindex(
            -contribution_to_logprobs[:, :-1], tokens_group, "g buf [g buf]"
        )

        # (C)
        # Now that we've indexed everything, construct the batch of SequenceData objects
        sequence_data = {}
        g_total = 0
        for group_name, indices in indices_dict.items():
            lower, upper = g_total, g_total + len(indices)
            sequence_data[group_name] = SequenceDataBatch(
                token_ids=tokens_group[lower:upper].tolist(),
                feat_acts=feat_acts_group[lower:upper, 1:].tolist(),
                contribution_to_loss=contribution_to_loss[lower:upper].tolist(),
                repeat=False,
                top5_token_ids=top5_contribution_to_logits.indices[
                    lower:upper
                ].tolist(),
                top5_logit_contributions=top5_contribution_to_logits.values[
                    lower:upper
                ].tolist(),
                bottom5_token_ids=bottom5_contribution_to_logits.indices[
                    lower:upper
                ].tolist(),
                bottom5_logit_contributions=bottom5_contribution_to_logits.values[
                    lower:upper
                ].tolist(),
            )
            g_total += len(indices)

        # Add this feature's sequence data to the list
        sequence_data_list.append(sequence_data)

    t5 = time.time()

    # ! Get all data for the middle column visualisations, i.e. the two histograms & the logit table

    nonzero_feat_acts = []
    frac_nonzero = []
    frequencies_histogram_data = []
    top10_logits = []
    logits_histogram_data = []

    for feat in range(n_feats):
        _logits = logits[feat]

        # Get data for logits histogram
        # logits_histogram_data.append(HistogramData(_logits, n_bins=40, tickmode="5 ticks"))
        logits_histogram_data.append(HistogramData(_logits, n_bins=40, tickmode="ints"))

        # Get data for logits table
        top10_logits.append(
            (TopK(_logits.topk(k=10, largest=False)), TopK(_logits.topk(k=10)))
        )

        # Get data for feature activations histogram
        _feat_acts = feat_acts[..., feat]
        nonzero_feat_acts = _feat_acts[_feat_acts > 0]
        frac_nonzero.append(nonzero_feat_acts.numel() / _feat_acts.numel())
        frequencies_histogram_data.append(
            HistogramData(nonzero_feat_acts, n_bins=40, tickmode="ints")
        )

    t6 = time.time()

    # ! Return the output, as a dict of FeatureData items

    vocab_dict = model.tokenizer.vocab
    vocab_dict = {
        v: k.replace("Ġ", " ").replace("\n", "\\n") for k, v in vocab_dict.items()
    }

    return_obj = {
        feat: FeatureData(
            # For right-hand sequences
            sequence_data=sequence_data_list[i],
            # For middle column (logits table, and both histograms)
            top10_logits=top10_logits[i],
            logits_histogram_data=logits_histogram_data[i],
            frequencies_histogram_data=frequencies_histogram_data[i],
            frac_nonzero=frac_nonzero[i],
            # For left column, i.e. the 3 tables of size 3
            neuron_alignment=(top3_neurons_aligned[i], pct_of_l1[i]),
            neurons_correlated=(
                top_correlations_neurons[0][i],
                top_correlations_neurons[1][i],
            ),
            b_features_correlated=None,  # (top_correlations_encoder_B[0][i], top_correlations_encoder_B[1][i]),
            # Other stuff (not containing data)
            vocab_dict=vocab_dict,
            buffer=buffer,
            n_groups=n_groups,
            first_group_size=first_group_size,
            other_groups_size=other_groups_size,
        )
        for i, feat in enumerate(feature_idx)
    }

    # ! If verbose, try to estimate time it will take to generate data for all features, plus storage space

    if verbose:
        n_feats_total = encoder.cfg.d_sae

        # Get time
        total_time = t5 - t0
        table = Table("Task", "Time", "Pct %", title="Time taken for each task")
        for task, _time in zip(
            [
                "Setup code",
                "Fwd passes",
                "Concats",
                "Left-hand tables",
                "Right-hand sequences",
                "Middle column",
            ],
            [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5],
        ):
            frac = _time / total_time
            table.add_row(task, f"{_time:.2f}s", f"{frac:.1%}")
        rprint(table)
        est = (t3 - t0) + (n_feats_total / n_feats) * (t6 - t4) / 60
        print(f"Estimated time for all {n_feats_total} features = {est:.0f} minutes\n")

        # Get filesizes, for different methods of saving
        batch_size = 50
        if n_feats >= batch_size:
            print(
                f"Estimated filesize of all {n_feats_total} features if saved in groups of batch_size, with save type..."
            )
            save_obj = {
                k: v for k, v in return_obj.items() if k in feature_idx[:batch_size]
            }
            filename = str(Path(__file__).parent.resolve() / "temp")
            for save_type in ["pkl", "gzip"]:
                t0 = time.time()
                full_filename = FeatureData.save_batch(
                    save_obj, filename=filename, save_type=save_type
                )
                t1 = time.time()
                loaded_obj = FeatureData.load_batch(
                    filename, save_type=save_type, vocab_dict=vocab_dict
                )
                t2 = time.time()
                filesize = os.path.getsize(full_filename) / 1e6
                print(
                    f"{save_type:>5} = {filesize * n_feats_total / batch_size:>5.1f} MB, save time = {t1-t0:.3f}s, load time = {t2-t1:.3f}s"
                )
                os.remove(full_filename)

    return return_obj
