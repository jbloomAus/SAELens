import os
from typing import Any, Dict, List, Optional, Union, cast

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time

import numpy as np
import torch
from matplotlib import colors
from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SequencesConfig,
)
from sae_vis.data_fetching_fns import get_feature_data
from tqdm import tqdm

from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

OUT_OF_RANGE_TOKEN = "<|outofrange|>"

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list(
    "bg_color_map", ["white", "darkorange"]
)


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


class NeuronpediaRunner:

    def __init__(
        self,
        sae_path: str,
        feature_sparsity_path: Optional[str] = None,
        neuronpedia_parent_folder: str = "./neuronpedia_outputs",
        init_session: bool = True,
        # token pars
        n_batches_to_sample_from: int = 2**12,
        n_prompts_to_select: int = 4096 * 6,
        # sampling pars
        n_features_at_a_time: int = 1024,
        buffer_tokens_left: int = 8,
        buffer_tokens_right: int = 8,
        # start and end batch
        start_batch_inclusive: int = 0,
        end_batch_inclusive: Optional[int] = None,
    ):
        self.sae_path = sae_path
        if init_session:
            self.init_sae_session()

        self.feature_sparsity_path = feature_sparsity_path
        self.n_features_at_a_time = n_features_at_a_time
        self.buffer_tokens_left = buffer_tokens_left
        self.buffer_tokens_right = buffer_tokens_right
        self.n_batches_to_sample_from = n_batches_to_sample_from
        self.n_prompts_to_select = n_prompts_to_select
        self.start_batch = start_batch_inclusive
        self.end_batch = end_batch_inclusive

        # Deal with file structure
        if not os.path.exists(neuronpedia_parent_folder):
            os.makedirs(neuronpedia_parent_folder)
        self.neuronpedia_folder = (
            f"{neuronpedia_parent_folder}/{self.get_folder_name()}"
        )
        if not os.path.exists(self.neuronpedia_folder):
            os.makedirs(self.neuronpedia_folder)

    def get_folder_name(self):
        model = self.sparse_autoencoder.cfg.model_name
        hook_point = self.sparse_autoencoder.cfg.hook_point
        d_sae = self.sparse_autoencoder.cfg.d_sae
        dashboard_folder_name = f"{model}_{hook_point}_{d_sae}"

        return dashboard_folder_name

    def init_sae_session(self):
        (
            self.model,
            sae_group,
            self.activation_store,
        ) = LMSparseAutoencoderSessionloader.load_pretrained_sae(self.sae_path)
        # TODO: handle multiple autoencoders
        self.sparse_autoencoder = next(iter(sae_group))[1]

    def get_tokens(
        self, n_batches_to_sample_from: int = 2**12, n_prompts_to_select: int = 4096 * 6
    ):
        all_tokens_list = []
        pbar = tqdm(range(n_batches_to_sample_from))
        for _ in pbar:
            batch_tokens = self.activation_store.get_batch_tokens()
            batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
                : batch_tokens.shape[0]
            ]
            all_tokens_list.append(batch_tokens)

        all_tokens = torch.cat(all_tokens_list, dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
        return all_tokens[:n_prompts_to_select]

    def round_list(self, to_round: list[float]):
        return list(np.round(to_round, 3))

    def to_str_tokens_safe(
        self, vocab_dict: Dict[int, str], tokens: Union[int, List[int], torch.Tensor]
    ):
        """
        does to_str_tokens, except handles out of range
        """
        vocab_max_index = self.model.cfg.d_vocab - 1
        # Deal with the int case separately
        if isinstance(tokens, int):
            if tokens > vocab_max_index:
                return OUT_OF_RANGE_TOKEN
            return vocab_dict[tokens]

        # If the tokens are a (possibly nested) list, turn them into a tensor
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        # Get flattened list of tokens
        str_tokens = [
            (vocab_dict[t] if t <= vocab_max_index else OUT_OF_RANGE_TOKEN)
            for t in tokens.flatten().tolist()
        ]

        # Reshape
        return np.reshape(str_tokens, tokens.shape).tolist()

    def run(self):
        """
        Generate the Neuronpedia outputs.
        """

        if self.model is None:
            self.init_sae_session()

        self.n_features = self.sparse_autoencoder.cfg.d_sae
        assert self.n_features is not None

        # if we have feature sparsity, then use it to only generate outputs for non-dead features
        self.target_feature_indexes: list[int] = []
        if self.feature_sparsity_path:
            loaded = torch.load(
                self.feature_sparsity_path, map_location=self.sparse_autoencoder.device
            )
            self.target_feature_indexes = (
                (loaded > -5).nonzero(as_tuple=True)[0].tolist()
            )
        else:
            self.target_feature_indexes = list(range(self.n_features))
            print("No feat sparsity path specified - doing all indexes.")

        # divide into batches
        feature_idx = torch.tensor(self.target_feature_indexes)
        n_subarrays = np.ceil(len(feature_idx) / self.n_features_at_a_time).astype(int)
        feature_idx = np.array_split(feature_idx, n_subarrays)
        feature_idx = [x.tolist() for x in feature_idx]

        print(f"==== Starting at batch: {self.start_batch}")
        if self.end_batch is not None:
            print(f"==== Ending at batch: {self.end_batch}")

        if self.start_batch > len(feature_idx) + 1:
            print(
                f"Start batch {self.start_batch} is greater than number of batches + 1 {len(feature_idx)}, exiting"
            )
            exit()

        # write dead into file so we can create them as dead in Neuronpedia
        skipped_indexes = set(range(self.n_features)) - set(self.target_feature_indexes)
        skipped_indexes_json = json.dumps({"skipped_indexes": list(skipped_indexes)})
        with open(f"{self.neuronpedia_folder}/skipped_indexes.json", "w") as f:
            f.write(skipped_indexes_json)

        print(f"Total features to run: {len(self.target_feature_indexes)}")
        print(f"Total skipped: {len(skipped_indexes)}")
        print(f"Total batches: {len(feature_idx)}")

        print(f"Hook Point Layer: {self.sparse_autoencoder.cfg.hook_point_layer}")
        print(f"Hook Point: {self.sparse_autoencoder.cfg.hook_point}")
        print(f"Writing files to: {self.neuronpedia_folder}")

        # get tokens:
        start = time.time()
        tokens = self.get_tokens(
            self.n_batches_to_sample_from, self.n_prompts_to_select
        )
        end = time.time()
        print(f"Time to get tokens: {end - start}")

        vocab_dict = cast(Any, self.model.tokenizer).vocab
        vocab_dict = {
            v: k.replace("Ġ", " ").replace("\n", "\\n").replace("Ċ", "\n")
            for k, v in vocab_dict.items()
        }
        # pad with blank tokens to the actual vocab size
        for i in range(len(vocab_dict), self.model.cfg.d_vocab):
            vocab_dict[i] = OUT_OF_RANGE_TOKEN

        with torch.no_grad():
            feature_batch_count = 0
            for features_to_process in tqdm(feature_idx):
                feature_batch_count = feature_batch_count + 1

                if feature_batch_count < self.start_batch:
                    # print(f"Skipping batch - it's after start_batch: {feature_batch_count}")
                    continue
                if self.end_batch is not None and feature_batch_count > self.end_batch:
                    # print(f"Skipping batch - it's after end_batch: {feature_batch_count}")
                    continue

                print(f"Doing batch: {feature_batch_count}")

                layout = SaeVisLayoutConfig(
                    columns=[
                        Column(
                            SequencesConfig(
                                stack_mode="stack-all",
                                buffer=(
                                    self.buffer_tokens_left,
                                    self.buffer_tokens_right,
                                ),
                                compute_buffer=False,
                                n_quantiles=10,
                                top_acts_group_size=20,
                                quantile_group_size=5,
                            ),
                            width=650,
                        ),
                        Column(
                            ActsHistogramConfig(),
                            FeatureTablesConfig(n_rows=5),
                            width=500,
                        ),
                    ],
                    height=1000,
                )
                feature_vis_params = SaeVisConfig(
                    hook_point=self.sparse_autoencoder.cfg.hook_point,
                    minibatch_size_features=256,
                    minibatch_size_tokens=64,
                    features=features_to_process,
                    verbose=False,
                    feature_centric_layout=layout,
                )

                feature_data = get_feature_data(
                    encoder=self.sparse_autoencoder,  # type: ignore
                    model=self.model,
                    tokens=tokens,
                    cfg=feature_vis_params,
                )

                features_outputs = []
                for _, feat_index in enumerate(feature_data.feature_data_dict.keys()):
                    feature = feature_data.feature_data_dict[feat_index]

                    feature_output = {}
                    feature_output["featureIndex"] = feat_index

                    top10_logits = self.round_list(feature.logits_table_data.top_logits)
                    bottom10_logits = self.round_list(
                        feature.logits_table_data.bottom_logits
                    )

                    # TODO: don't precompute/store these. should do it on the frontend
                    max_value = max(
                        np.absolute(bottom10_logits).max(),
                        np.absolute(top10_logits).max(),
                    )
                    neg_bg_values = self.round_list(
                        np.absolute(bottom10_logits) / max_value
                    )
                    pos_bg_values = self.round_list(
                        np.absolute(top10_logits) / max_value
                    )
                    feature_output["neg_bg_values"] = neg_bg_values
                    feature_output["pos_bg_values"] = pos_bg_values

                    if feature.feature_tables_data:
                        feature_output["neuron_alignment_indices"] = (
                            feature.feature_tables_data.neuron_alignment_indices
                        )
                        feature_output["neuron_alignment_values"] = self.round_list(
                            feature.feature_tables_data.neuron_alignment_values
                        )
                        feature_output["neuron_alignment_l1"] = self.round_list(
                            feature.feature_tables_data.neuron_alignment_l1
                        )
                        feature_output["correlated_neurons_indices"] = (
                            feature.feature_tables_data.correlated_neurons_indices
                        )
                        # TODO: this value doesn't exist in the new output type, commenting out for now
                        # there is a cossim value though - is that what's needed?
                        # feature_output["correlated_neurons_l1"] = self.round_list(
                        #     feature.feature_tables_data.correlated_neurons_l1
                        # )
                        feature_output["correlated_neurons_pearson"] = self.round_list(
                            feature.feature_tables_data.correlated_neurons_pearson
                        )
                        # feature_output["correlated_features_indices"] = (
                        #     feature.feature_tables_data.correlated_features_indices
                        # )
                        # feature_output["correlated_features_l1"] = self.round_list(
                        #     feature.feature_tables_data.correlated_features_l1
                        # )
                        # feature_output["correlated_features_pearson"] = self.round_list(
                        #     feature.feature_tables_data.correlated_features_pearson
                        # )

                    feature_output["neg_str"] = self.to_str_tokens_safe(
                        vocab_dict, feature.logits_table_data.bottom_token_ids
                    )
                    feature_output["neg_values"] = bottom10_logits
                    feature_output["pos_str"] = self.to_str_tokens_safe(
                        vocab_dict, feature.logits_table_data.top_token_ids
                    )
                    feature_output["pos_values"] = top10_logits

                    # TODO: don't know what this should be in the new version
                    # feature_output["frac_nonzero"] = (
                    #     feature.middle_plots_data.frac_nonzero
                    # )

                    freq_hist_data = feature.acts_histogram_data
                    freq_bar_values = self.round_list(freq_hist_data.bar_values)
                    feature_output["freq_hist_data_bar_values"] = freq_bar_values
                    feature_output["freq_hist_data_tick_vals"] = self.round_list(
                        freq_hist_data.tick_vals
                    )

                    # TODO: don't precompute/store these. should do it on the frontend
                    freq_bar_values_clipped = [
                        (0.4 * max(freq_bar_values) + 0.6 * v) / max(freq_bar_values)
                        for v in freq_bar_values
                    ]
                    freq_bar_colors = [
                        colors.rgb2hex(BG_COLOR_MAP(v)) for v in freq_bar_values_clipped
                    ]
                    feature_output["freq_hist_data_bar_heights"] = self.round_list(
                        freq_hist_data.bar_heights
                    )
                    feature_output["freq_bar_colors"] = freq_bar_colors

                    logits_hist_data = feature.logits_histogram_data
                    feature_output["logits_hist_data_bar_heights"] = self.round_list(
                        logits_hist_data.bar_heights
                    )
                    feature_output["logits_hist_data_bar_values"] = self.round_list(
                        logits_hist_data.bar_values
                    )
                    feature_output["logits_hist_data_tick_vals"] = self.round_list(
                        logits_hist_data.tick_vals
                    )

                    # TODO: check this
                    feature_output["num_tokens_for_dashboard"] = (
                        self.n_prompts_to_select
                    )

                    activations = []
                    sdbs = feature.sequence_data
                    for sgd in sdbs.seq_group_data:
                        for sd in sgd.seq_data:
                            if (
                                sd.top_token_ids is not None
                                and sd.bottom_token_ids is not None
                                and sd.top_logits is not None
                                and sd.bottom_logits is not None
                            ):
                                activation = {}
                                strs = []
                                posContribs = []
                                negContribs = []
                                for i in range(len(sd.token_ids)):
                                    strs.append(
                                        self.to_str_tokens_safe(
                                            vocab_dict, sd.token_ids[i]
                                        )
                                    )
                                    posContrib = {}
                                    posTokens = [
                                        self.to_str_tokens_safe(vocab_dict, j)
                                        for j in sd.top_token_ids[i]
                                    ]
                                    if len(posTokens) > 0:
                                        posContrib["t"] = posTokens
                                        posContrib["v"] = self.round_list(
                                            sd.top_logits[i]
                                        )
                                    posContribs.append(posContrib)
                                    negContrib = {}
                                    negTokens = [
                                        self.to_str_tokens_safe(vocab_dict, j)  # type: ignore
                                        for j in sd.bottom_token_ids[i]
                                    ]
                                    if len(negTokens) > 0:
                                        negContrib["t"] = negTokens
                                        negContrib["v"] = self.round_list(
                                            sd.bottom_logits[i]
                                        )
                                    negContribs.append(negContrib)

                                activation["logitContributions"] = json.dumps(
                                    {"pos": posContribs, "neg": negContribs}
                                )
                                activation["tokens"] = strs
                                activation["values"] = self.round_list(sd.feat_acts)
                                activation["maxValue"] = max(activation["values"])
                                activation["lossValues"] = self.round_list(
                                    sd.loss_contribution
                                )

                                activations.append(activation)
                    feature_output["activations"] = activations

                    features_outputs.append(feature_output)

                json_object = json.dumps(features_outputs, cls=NpEncoder)

                with open(
                    f"{self.neuronpedia_folder}/batch-{feature_batch_count}.json", "w"
                ) as f:
                    f.write(json_object)

        return
