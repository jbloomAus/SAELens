import math
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib import colors

from sae_analysis.visualizer.utils_fns import to_str_tokens

"""
Key feature of these functions: the arguments should be descriptive of their role in the actual HTML
visualisation. If the arguments are super arcane features of the model data, this is bad!
"""

ROOT_DIR = Path(__file__).parent

CSS_DIR = Path(__file__).parent / "css"

CSS = "\n".join(
    [
        (CSS_DIR / "general.css").read_text(),
        (CSS_DIR / "sequences.css").read_text(),
        (CSS_DIR / "tables.css").read_text(),
    ]
)

HTML_DIR = Path(__file__).parent / "html"
HTML_TOKEN = (HTML_DIR / "token_template.html").read_text()
HTML_LEFT_TABLES = (HTML_DIR / "threerow_table_template.html").read_text()
HTML_LOGIT_TABLES = (HTML_DIR / "logit_table_template.html").read_text()
HTML_LOGIT_HIST = (HTML_DIR / "logits_histogram.html").read_text()
HTML_FREQ_HIST = (HTML_DIR / "frequency_histogram.html").read_text()
HTML_HOVERTEXT_SCRIPT = (HTML_DIR / "hovertext_script.html").read_text()


BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list(
    "bg_color_map", ["white", "darkorange"]
)


def generate_tok_html(
    vocab_dict: dict[int, str],
    this_token: int,
    underline_color: str,
    bg_color: str,
    is_bold: bool = False,
    feat_act: float = 0.0,
    contribution_to_loss: float = 0.0,
    pos_ids: List[int] = [0, 0, 0, 0, 0],
    pos_val: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
    neg_ids: List[int] = [0, 0, 0, 0, 0],
    neg_val: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0],
):
    """
    Creates a single sequence visualisation, by reading from the `token_template.html` file.

    Currently, a bunch of things are randomly chosen rather than actually calculated (we're going for
    proof of concept here).
    """
    html_output = (
        HTML_TOKEN.replace("this_token", to_str_tokens(vocab_dict, this_token))
        .replace("feat_activation", f"{feat_act:+.3f}")
        .replace("feature_ablation", f"{contribution_to_loss:+.3f}")
        .replace("font_weight", "bold" if is_bold else "normal")
        .replace("bg_color", bg_color)
        .replace("underline_color", underline_color)
    )

    # Figure out if the activations were zero on previous token, i.e. no predictions were affected
    is_empty = len(pos_ids) + len(neg_ids) == 0

    # Get the string tokens
    pos_str = [to_str_tokens(vocab_dict, i) for i in pos_ids]
    neg_str = [to_str_tokens(vocab_dict, i) for i in neg_ids]

    # Pad out the pos_str etc lists, because they might be short
    pos_str.extend([""] * 5)
    neg_str.extend([""] * 5)
    pos_val.extend([0.0] * 5)
    neg_val.extend([0.0] * 5)

    # Make all the substitutions
    html_output = re.sub(
        r"pos_str_(\d)",
        lambda m: pos_str[int(m.group(1))].replace(" ", "&nbsp;"),
        html_output,
    )
    html_output = re.sub(
        r"neg_str_(\d)",
        lambda m: neg_str[int(m.group(1))].replace(" ", "&nbsp;"),
        html_output,
    )
    html_output = re.sub(
        r"pos_val_(\d)", lambda m: f"{pos_val[int(m.group(1))]:+.3f}", html_output
    )
    html_output = re.sub(
        r"neg_val_(\d)", lambda m: f"{neg_val[int(m.group(1))]:+.3f}", html_output
    )

    # If the effect on loss is nothing (because feature isn't active), replace the HTML output with smth saying this
    if is_empty:
        html_output = (
            html_output.replace(
                '<div class="half-width-container">',
                '<div class="half-width-container" style="display: none;">',
            )
            .replace(
                "<!-- No effect! -->",
                '<p style="font-size:0.8em;">Feature not active on prev token;<br>no predictions were affected.</p>',
            )
            .replace(
                '<div class="tooltip">',
                '<div class="tooltip" style="height:175px; width:250px;">',
            )
        )
    # Also, delete the columns as appropriate if the number is between 0 and 5
    else:
        html_output = html_output.replace(
            '<tr><td class="right-aligned"><code></code></td><td class="left-aligned">+0.000</td></tr>',
            "",
        )

    return html_output


def generate_seq_html(
    vocab_dict: dict[int, str],
    token_ids: List[int],
    feat_acts: List[float],
    contribution_to_loss: List[float],
    pos_ids: List[List[int]],
    neg_ids: List[List[int]],
    pos_val: List[List[float]],
    neg_val: List[List[float]],
    bold_idx: Optional[int] = None,
    is_repeat: bool = False,
):
    assert (
        len(token_ids) == len(feat_acts) == len(contribution_to_loss)
    ), "All input lists must be of the same length."

    # ! Clip values in [0, 1] range (temporary)
    bg_values = np.clip(feat_acts, 0, 1)
    underline_values = np.clip(contribution_to_loss, -1, 1)

    # Decide whether the sequence is repeated or not
    # TODO - decide whether this is actually worthwhile (probably not)
    # repeat_obj = "<span style='display: inline-block; width:25px;'>🔁</span>"
    # if not(is_repeat): repeat_obj = repeat_obj.replace("🔁", "")

    # Define the HTML object, which we'll iteratively add to
    html_output = '<div class="seq">'  # + repeat_obj
    underline_color = "transparent"

    for i in range(len(token_ids)):
        # Get background color, which is {0: transparent, +1: darkorange}
        bg_val = bg_values[i]
        bg_color = colors.rgb2hex(BG_COLOR_MAP(bg_val))

        # Get underline color, which is {-1: blue, 0: transparent, +1: red}
        underline_val = underline_values[i]
        if math.isnan(underline_val):
            underline_color = "transparent"
        elif underline_val < 0:
            v = int(255 * (underline_val + 1))
            underline_color = f"rgb({v}, {v}, 255)"
        elif underline_val >= 0:
            v = int(255 * (1 - underline_val))
            underline_color = f"rgb(255, {v}, {v})"

        html_output += generate_tok_html(
            vocab_dict=vocab_dict,
            this_token=token_ids[i],
            underline_color=underline_color,
            bg_color=bg_color,
            pos_ids=pos_ids[i],
            neg_ids=neg_ids[i],
            pos_val=pos_val[i],
            neg_val=neg_val[i],
            is_bold=(bold_idx is not None) and (bold_idx == i),
            feat_act=feat_acts[i],
            contribution_to_loss=contribution_to_loss[i],
        )

    html_output += "</div>"

    return html_output


def generate_tables_html(
    # First, all the arguments for the left-hand tables
    neuron_alignment_indices: List[int],
    neuron_alignment_values: List[float],
    neuron_alignment_l1: List[float],
    correlated_neurons_indices: List[int],
    correlated_neurons_pearson: List[float],
    correlated_neurons_l1: List[float],
    correlated_features_indices: List[int] | None,
    correlated_features_pearson: List[float] | None,
    correlated_features_l1: List[float] | None,
    # Second, all the arguments for the middle tables (neg/pos logits)
    neg_str: List[str],
    neg_values: List[float],
    neg_bg_values: List[float],
    pos_str: List[str],
    pos_values: List[float],
    pos_bg_values: List[float],
):
    """
    See the file `threerow_table_template.html` (with the CSS in the other 3 files), for this to make more sense.
    """
    html_output = HTML_LEFT_TABLES

    for letter, mylist, myformat in zip(
        "IVLIPCIPC",
        [
            neuron_alignment_indices,
            neuron_alignment_values,
            neuron_alignment_l1,
            correlated_neurons_indices,
            correlated_neurons_pearson,
            correlated_neurons_l1,
            # correlated_features_indices,
            # correlated_features_pearson,
            # correlated_features_l1,
            # duplicate these to remove.
            correlated_neurons_indices,
            correlated_neurons_pearson,
            correlated_neurons_l1,
        ],
        [None, "+.2f", ".1%", None, "+.2f", "+.2f", None, "+.2f", "+.2f"],
    ):
        fn = lambda m: (
            str(mylist[int(m.group(1))])
            if myformat is None
            else format(mylist[int(m.group(1))], myformat)
        )
        html_output = re.sub(letter + r"(\d)", fn, html_output, count=3)

    html_output_2 = HTML_LOGIT_TABLES

    neg_bg_colors = [
        f"rgba(255, {int(255 * (1 - v))}, {int(255 * (1 - v))}, 0.5)"
        for v in neg_bg_values
    ]
    pos_bg_colors = [
        f"rgba({int(255 * (1 - v))}, {int(255 * (1 - v))}, 255, 0.5)"
        for v in pos_bg_values
    ]

    for letter, mylist in zip(
        "SVCSVC",
        [neg_str, neg_values, neg_bg_colors, pos_str, pos_values, pos_bg_colors],
    ):
        if letter == "S":
            fn = lambda m: str(mylist[int(m.group(1))]).replace(" ", "&nbsp;")
        elif letter == "V":
            fn = lambda m: format(mylist[int(m.group(1))], "+.2f")
        elif letter == "C":
            fn = lambda m: str(mylist[int(m.group(1))])
        else:
            raise ValueError("This should never happen.")

        html_output_2 = re.sub(letter + r"(\d)", fn, html_output_2, count=10)

    return (html_output, html_output_2)


def generate_histograms(freq_hist_data: Any, logits_hist_data: Any) -> Tuple[str, str]:
    """This generates both histograms at once."""

    # Start off high, cause we want closer to orange than white for the left-most bars
    freq_bar_values = freq_hist_data.bar_values
    freq_bar_values_clipped = [
        (0.4 * max(freq_bar_values) + 0.6 * v) / max(freq_bar_values)
        for v in freq_bar_values
    ]
    freq_bar_colors = [colors.rgb2hex(BG_COLOR_MAP(v)) for v in freq_bar_values_clipped]

    return (
        HTML_FREQ_HIST,
        (
            HTML_LOGIT_HIST
            # Fill in all the freq histogram data
            .replace("BAR_HEIGHTS_FREQ", str(list(freq_hist_data.bar_heights)))
            .replace("BAR_VALUES_FREQ", str(list(freq_hist_data.bar_values)))
            .replace("BAR_COLORS_FREQ", str(list(freq_bar_colors)))
            .replace("TICK_VALS_FREQ", str(list(freq_hist_data.tick_vals)))
            # Fill in all the freq logits histogram data
            .replace("BAR_HEIGHTS_LOGITS", str(list(logits_hist_data.bar_heights)))
            .replace("BAR_VALUES_LOGITS", str(list(logits_hist_data.bar_values)))
            .replace("TICK_VALS_LOGITS", str(list(logits_hist_data.tick_vals)))
        ),
    )
