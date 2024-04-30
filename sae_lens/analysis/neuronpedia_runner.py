import os
from typing import Any, Dict, List, Optional, Union, cast

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json

import numpy as np
import torch
from matplotlib import colors
from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SequencesConfig,
)
from sae_vis.data_storing_fns import SaeVisData
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens.toolkit.pretrained_saes import load_sparsity
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.sparse_autoencoder import SparseAutoencoder

OUT_OF_RANGE_TOKEN = "<|outofrange|>"

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list(
    "bg_color_map", ["white", "darkorange"]
)

DEFAULT_SPARSITY_THRESHOLD = -5

HTML_ANOMALIES = {
    "âĢĶ": "—",
    "âĢĵ": "–",
    "âĢľ": "“",
    "âĢĿ": "”",
    "âĢĺ": "‘",
    "âĢĻ": "’",
    "âĢĭ": " ",  # todo: this is actually zero width space
    "Ġ": " ",
    "Ċ": "\n",
    "ĉ": "\t",
}


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
        sae_id: str,
        sae_path: str,
        outputs_dir: str,
        sparsity_threshold: int = DEFAULT_SPARSITY_THRESHOLD,
        # token pars
        n_batches_to_sample_from: int = 2**12,
        n_prompts_to_select: int = 4096 * 6,
        # batching
        n_features_at_a_time: int = 128,
        start_batch_inclusive: int = 0,
        end_batch_inclusive: Optional[int] = None,
        # quantiles
        n_quantiles: int = 5,
        top_acts_group_size: int = 20,
        quantile_group_size: int = 5,
    ):

        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

        self.sae_path = sae_path
        self.sparse_autoencoder = SparseAutoencoder.load_from_pretrained(
            self.sae_path, device=self.device
        )
        loader = LMSparseAutoencoderSessionloader(self.sparse_autoencoder.cfg)
        self.model, _, self.activation_store = loader.load_sae_training_group_session()
        self.model_id = self.sparse_autoencoder.cfg.model_name
        self.layer = self.sparse_autoencoder.cfg.hook_point_layer
        self.sae_id = sae_id
        self.sparsity_threshold = sparsity_threshold
        self.n_features_at_a_time = n_features_at_a_time
        self.n_batches_to_sample_from = n_batches_to_sample_from
        self.n_prompts_to_select = n_prompts_to_select
        self.start_batch = start_batch_inclusive
        self.end_batch = end_batch_inclusive
        self.n_quantiles = n_quantiles
        self.top_acts_group_size = top_acts_group_size
        self.quantile_group_size = quantile_group_size

        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        self.outputs_dir = outputs_dir

    def get_tokens(
        self,
        n_batches_to_sample_from: int = 2**12,
        n_prompts_to_select: int = 4096 * 6,
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
        self,
        vocab_dict: Dict[int, str],
        tokens: Union[int, List[int], torch.Tensor],
    ):
        """
        does to_str_tokens, except handles out of range
        """
        assert self.model is not None
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
        self.n_features = self.sparse_autoencoder.cfg.d_sae
        assert self.n_features is not None

        # if we have feature sparsity, then use it to only generate outputs for non-dead features
        self.target_feature_indexes: list[int] = []
        sparsity = load_sparsity(self.sae_path)
        # convert sparsity to logged sparsity if it's not
        # TODO: standardize the sparsity file format
        if len(sparsity) > 0 and sparsity[0] >= 0:
            sparsity = torch.log10(sparsity + 1e-10)
        sparsity = sparsity.to(self.device)
        self.target_feature_indexes = (
            (sparsity > self.sparsity_threshold).nonzero(as_tuple=True)[0].tolist()
        )

        # divide into batches
        feature_idx = torch.tensor(self.target_feature_indexes)
        n_subarrays = np.ceil(len(feature_idx) / self.n_features_at_a_time).astype(int)
        feature_idx = np.array_split(feature_idx, n_subarrays)
        feature_idx = [x.tolist() for x in feature_idx]

        # print(f"==== Starting Batch: {self.start_batch}")
        # if self.end_batch is not None and self.end_batch != self.start_batch:
        #     print(f"==== Ending at Batch: {self.end_batch}")

        if self.start_batch > len(feature_idx) + 1:
            print(
                f"Start batch {self.start_batch} is greater than number of batches + 1 {len(feature_idx)}, exiting"
            )
            exit()

        # write dead into file so we can create them as dead in Neuronpedia
        skipped_indexes = set(range(self.n_features)) - set(self.target_feature_indexes)
        skipped_indexes_json = json.dumps(
            {
                "model_id": self.model_id,
                "layer": str(self.layer),
                "sae_id": self.sae_id,
                "log_sparsity": self.sparsity_threshold,
                "skipped_indexes": list(skipped_indexes),
            }
        )
        with open(f"{self.outputs_dir}/skipped_indexes.json", "w") as f:
            f.write(skipped_indexes_json)

        tokens_file = f"{self.outputs_dir}/tokens_{self.n_batches_to_sample_from}_{self.n_prompts_to_select}.pt"
        if os.path.isfile(tokens_file):
            print("Tokens exist, loading them.")
            tokens = torch.load(tokens_file)
        else:
            print("Tokens don't exist, making them.")
            tokens = self.get_tokens(
                self.n_batches_to_sample_from, self.n_prompts_to_select
            )
            torch.save(
                tokens,
                tokens_file,
            )

        vocab_dict = self.model.tokenizer.vocab
        new_vocab_dict = {}
        # Replace substrings in the keys of vocab_dict using HTML_ANOMALIES
        for k, v in vocab_dict.items():
            modified_key = k
            for anomaly in HTML_ANOMALIES:
                modified_key = modified_key.replace(anomaly, HTML_ANOMALIES[anomaly])
            new_vocab_dict[v] = modified_key
        vocab_dict = new_vocab_dict

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

                output_file = f"{self.outputs_dir}/batch-{feature_batch_count}.json"
                # if output_file exists, skip
                if os.path.isfile(output_file):
                    logline = f"\n++++++++++ Skipping Batch #{feature_batch_count} output. File exists: {output_file} ++++++++++\n"
                    print(logline)
                    continue

                print(f"========== Running Batch #{feature_batch_count} ==========")

                layout = SaeVisLayoutConfig(
                    columns=[
                        Column(
                            SequencesConfig(
                                stack_mode="stack-all",
                                buffer=None,  # type: ignore
                                compute_buffer=True,
                                n_quantiles=self.n_quantiles,
                                top_acts_group_size=self.top_acts_group_size,
                                quantile_group_size=self.quantile_group_size,
                            ),
                            ActsHistogramConfig(),
                            LogitsHistogramConfig(),
                            LogitsTableConfig(),
                            FeatureTablesConfig(n_rows=3),
                        )
                    ]
                )
                feature_vis_params = SaeVisConfig(
                    hook_point=self.sparse_autoencoder.cfg.hook_point,
                    minibatch_size_features=128,
                    minibatch_size_tokens=64,
                    features=features_to_process,
                    verbose=True,
                    feature_centric_layout=layout,
                )
                feature_data = SaeVisData.create(
                    encoder=self.sparse_autoencoder,  # type: ignore
                    model=cast(HookedTransformer, self.model),
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
                        feature_output["correlated_neurons_l1"] = self.round_list(
                            feature.feature_tables_data.correlated_neurons_cossim
                        )
                        feature_output["correlated_neurons_pearson"] = self.round_list(
                            feature.feature_tables_data.correlated_neurons_pearson
                        )
                        feature_output["correlated_features_indices"] = (
                            feature.feature_tables_data.correlated_features_indices
                        )
                        feature_output["correlated_features_l1"] = self.round_list(
                            feature.feature_tables_data.correlated_features_cossim
                        )
                        feature_output["correlated_features_pearson"] = self.round_list(
                            feature.feature_tables_data.correlated_features_pearson
                        )

                    feature_output["neg_str"] = self.to_str_tokens_safe(
                        vocab_dict, feature.logits_table_data.bottom_token_ids
                    )
                    feature_output["neg_values"] = bottom10_logits
                    feature_output["pos_str"] = self.to_str_tokens_safe(
                        vocab_dict, feature.logits_table_data.top_token_ids
                    )
                    feature_output["pos_values"] = top10_logits

                    feature_output["frac_nonzero"] = (
                        float(
                            feature.acts_histogram_data.title.split(" = ")[1].split(
                                "%"
                            )[0]
                        )
                        / 100
                        if feature.acts_histogram_data.title is not None
                        else 0
                    )

                    freq_hist_data = feature.acts_histogram_data
                    freq_bar_values = self.round_list(freq_hist_data.bar_values)
                    feature_output["freq_hist_data_bar_values"] = freq_bar_values
                    feature_output["freq_hist_data_bar_heights"] = self.round_list(
                        freq_hist_data.bar_heights
                    )

                    logits_hist_data = feature.logits_histogram_data
                    feature_output["logits_hist_data_bar_heights"] = self.round_list(
                        logits_hist_data.bar_heights
                    )
                    feature_output["logits_hist_data_bar_values"] = self.round_list(
                        logits_hist_data.bar_values
                    )

                    feature_output["num_tokens_for_dashboard"] = (
                        self.n_prompts_to_select
                    )

                    activations = []
                    sdbs = feature.sequence_data
                    for sgd in sdbs.seq_group_data:
                        binMin = 0
                        binMax = 0
                        binContains = 0
                        if "TOP ACTIVATIONS" in sgd.title:
                            binMin = -1
                            try:
                                binMax = float(sgd.title.split(" = ")[-1])
                            except ValueError:
                                print(f"Error parsing top acts: {sgd.title}")
                                binMax = 99
                            binContains = -1
                        elif "INTERVAL" in sgd.title:
                            try:
                                split = sgd.title.split("<br>")
                                firstSplit = split[0].split(" ")
                                binMin = float(firstSplit[1])
                                binMax = float(firstSplit[-1])
                                secondSplit = split[1].split(" ")
                                binContains = float(secondSplit[-1].rstrip("%")) / 100
                            except ValueError:
                                print(f"Error parsing interval: {sgd.title}")
                        for sd in sgd.seq_data:
                            if (
                                sd.top_token_ids is not None
                                and sd.bottom_token_ids is not None
                                and sd.top_logits is not None
                                and sd.bottom_logits is not None
                            ):
                                activation = {}
                                activation["binMin"] = binMin
                                activation["binMax"] = binMax
                                activation["binContains"] = binContains

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

                to_write = {
                    "model_id": self.model_id,
                    "layer": str(self.layer),
                    "sae_id": self.sae_id,
                    "features": features_outputs,
                    "n_batches_to_sample_from": self.n_batches_to_sample_from,
                    "n_prompts_to_select": self.n_prompts_to_select,
                }
                json_object = json.dumps(to_write, cls=NpEncoder)

                with open(
                    output_file,
                    "w",
                ) as f:
                    f.write(json_object)

                logline = f"\n========== Completed Batch #{feature_batch_count} output: {output_file} ==========\n"

        return
