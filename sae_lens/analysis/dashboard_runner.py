import os
from typing import Any, Optional, cast

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import time
import uuid

import pandas as pd
import plotly
import plotly.express as px
import torch
import wandb
from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SequencesConfig,
)
from sae_vis.data_fetching_fns import get_feature_data
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader


class DashboardRunner:

    model: HookedTransformer | None = None

    def __init__(
        self,
        sae_path: Optional[str] = None,
        dashboard_parent_folder: str = "./feature_dashboards",
        wandb_artifact_path: Optional[str] = None,
        init_session: bool = True,
        # token pars
        n_batches_to_sample_from: int = 2**12,
        n_prompts_to_select: int = 4096 * 6,
        # sampling pars
        n_features_at_a_time: int = 1024,
        max_batch_size: int = 256,
        buffer_tokens: int = 8,
        # util pars
        use_wandb: bool = False,
        continue_existing_dashboard: bool = True,
        final_index: Optional[int] = None,
    ):
        """ """

        if wandb_artifact_path is not None:
            artifact_dir = f"artifacts/{wandb_artifact_path.split('/')[2]}"
            if not os.path.exists(artifact_dir):
                print("Downloading artifact")
                run = wandb.init()
                assert run is not None  # keep pyright happy
                artifact = run.use_artifact(wandb_artifact_path)
                artifact_dir = artifact.download()
                path_to_artifact = f"{artifact_dir}/{os.listdir(artifact_dir)[0]}"
                # feature sparsity
                feature_sparsity_path = self.get_feature_sparsity_path(
                    wandb_artifact_path
                )
                artifact = run.use_artifact(feature_sparsity_path)
                artifact_dir = artifact.download()
                # add it as a property
                self.feature_sparsity = torch.load(
                    f"{artifact_dir}/{os.listdir(artifact_dir)[0]}"
                )
            else:
                print("Artifact already downloaded")
                path_to_artifact = f"{artifact_dir}/{os.listdir(artifact_dir)[0]}"

                feature_sparsity_path = self.get_feature_sparsity_path(
                    wandb_artifact_path
                )
                artifact_dir = f"artifacts/{feature_sparsity_path.split('/')[2]}"
                feature_sparsity_file = os.listdir(artifact_dir)[0]
                self.feature_sparsity = torch.load(
                    f"{artifact_dir}/{feature_sparsity_file}"
                )

            self.sae_path = path_to_artifact
        else:
            assert sae_path is not None
            self.sae_path = sae_path
            self.feature_sparsity = None

        if init_session:
            self.init_sae_session()

        self.n_features_at_a_time = n_features_at_a_time
        self.max_batch_size = max_batch_size
        self.buffer_tokens = buffer_tokens
        self.use_wandb = use_wandb
        self.final_index = (
            final_index
            if final_index is not None
            else self.sparse_autoencoder.cfg.d_sae
        )
        self.n_batches_to_sample_from = n_batches_to_sample_from
        self.n_prompts_to_select = n_prompts_to_select

        # Deal with file structure
        if not os.path.exists(dashboard_parent_folder):
            os.makedirs(dashboard_parent_folder)

        self.dashboard_folder = (
            f"{dashboard_parent_folder}/{self.get_dashboard_folder_name()}"
        )
        if not os.path.exists(self.dashboard_folder):
            os.makedirs(self.dashboard_folder)

        if not continue_existing_dashboard:
            # check if there are files there and if so abort
            if len(os.listdir(self.dashboard_folder)) > 0:
                raise ValueError("Dashboard folder not empty. Aborting.")

    def get_feature_sparsity_path(self, wandb_artifact_path: str):
        prefix = wandb_artifact_path.split(":")[0]
        return f"{prefix}_log_feature_sparsity:v9"

    def get_dashboard_folder_name(self):
        model = self.sparse_autoencoder.cfg.model_name
        hook_point = self.sparse_autoencoder.cfg.hook_point
        d_sae = self.sparse_autoencoder.cfg.d_sae
        dashboard_folder_name = f"{model}_{hook_point}_{d_sae}"

        return dashboard_folder_name

    def init_sae_session(self):
        (
            model,
            sae_group,
            self.activation_store,
        ) = LMSparseAutoencoderSessionloader.load_pretrained_sae(self.sae_path)
        assert isinstance(
            model, HookedTransformer
        )  # only HookedTransformer is allowed to be used in the dashboard
        self.model = model
        # TODO: handle multiple autoencoders
        self.sparse_autoencoder = next(iter(sae_group))[1]

    def get_tokens(
        self, n_batches_to_sample_from: int = 2**12, n_prompts_to_select: int = 4096 * 6
    ):
        """
        Get the tokens needed for dashboard generation.
        """

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

    def get_index_to_resume_from(self):
        i = 0
        assert self.n_features is not None  # keep pyright happy
        for i in range(self.n_features):
            if not os.path.exists(f"{self.dashboard_folder}/data_{i:04}.html"):
                break

        assert self.sparse_autoencoder.cfg.d_sae is not None  # keep pyright happy
        assert self.final_index is not None  # keep pyright happy
        n_features = self.sparse_autoencoder.cfg.d_sae
        n_features_at_a_time = self.n_features_at_a_time
        id_of_last_feature_without_dashboard = i
        n_features_remaining = self.final_index - id_of_last_feature_without_dashboard
        n_batches_to_do = n_features_remaining // n_features_at_a_time
        if self.final_index == n_features:
            id_to_start_from = max(
                0, n_features - (n_batches_to_do + 1) * n_features_at_a_time
            )
        else:
            id_to_start_from = 0  # testing purposes only

        print(f"File {i} does not exist")
        print(f"features left to do: {n_features_remaining}")
        print(f"id_to_start_from: {id_to_start_from}")
        print(
            f"number of batches to do: {(n_features - id_to_start_from) // n_features_at_a_time}"
        )

        return id_to_start_from

    @torch.no_grad()
    def get_feature_property_df(self):
        sparse_autoencoder = self.sparse_autoencoder
        feature_sparsity = (
            self.feature_sparsity
            if self.feature_sparsity is not None
            else torch.tensor(0)
        )

        W_dec_normalized = (
            sparse_autoencoder.W_dec.cpu()
        )  # / sparse_autoencoder.W_dec.cpu().norm(dim=-1, keepdim=True)
        W_enc_normalized = (
            sparse_autoencoder.W_enc.cpu()
            / sparse_autoencoder.W_enc.cpu().norm(dim=-1, keepdim=True)
        )
        d_e_projection = cosine_similarity(W_dec_normalized, W_enc_normalized.T)

        assert sparse_autoencoder.cfg.d_sae is not None  # keep pyright happy
        temp_df = pd.DataFrame(
            {
                "log_feature_sparsity": feature_sparsity + 1e-10,
                "d_e_projection": d_e_projection,
                # "d_e_projection_normalized": d_e_projection_normalized,
                "b_enc": sparse_autoencoder.b_enc.detach().cpu(),
                "feature": [
                    f"feature_{i}" for i in range(sparse_autoencoder.cfg.d_sae)
                ],
                "index": torch.arange(sparse_autoencoder.cfg.d_sae),
                "dead_neuron": (feature_sparsity < -9).cpu(),
            }
        )

        return temp_df

    def run(self):
        """
        Generate the dashboard.
        """

        run = None
        if self.use_wandb:
            # get name from wandb
            random_suffix = str(uuid.uuid4())[:8]
            name = f"{self.get_dashboard_folder_name()}_{random_suffix}"
            run = wandb.init(
                project="feature_dashboards",
                config=cast(Any, self.sparse_autoencoder.cfg),
                name=name,
                tags=[
                    f"model_{self.sparse_autoencoder.cfg.model_name}",
                    f"hook_point_{self.sparse_autoencoder.cfg.hook_point}",
                ],
            )

        if self.model is None:
            self.init_sae_session()

        # generate all the plots
        if self.use_wandb and self.feature_sparsity is not None:
            feature_property_df = self.get_feature_property_df()

            fig = px.histogram(
                self.feature_sparsity + 1e-10,
                nbins=100,
                log_x=False,
                title="Feature sparsity",
            )
            wandb.log(
                {"plots/feature_density_histogram": wandb.Html(plotly.io.to_html(fig))}
            )

            fig = px.histogram(
                self.sparse_autoencoder.b_enc.detach().cpu(), title="b_enc", nbins=100
            )
            wandb.log({"plots/b_enc_histogram": wandb.Html(plotly.io.to_html(fig))})

            fig = px.histogram(
                feature_property_df.d_e_projection, nbins=100, title="D/E projection"
            )
            wandb.log(
                {"plots/d_e_projection_histogram": wandb.Html(plotly.io.to_html(fig))}
            )

            fig = px.histogram(
                self.sparse_autoencoder.b_dec.detach().cpu(),
                nbins=100,
                title="b_dec projection onto W_dec",
            )
            wandb.log(
                {"plots/b_dec_projection_histogram": wandb.Html(plotly.io.to_html(fig))}
            )

            fig = px.scatter_matrix(
                feature_property_df,
                dimensions=["log_feature_sparsity", "d_e_projection", "b_enc"],
                color="dead_neuron",
                hover_name="feature",
                opacity=0.2,
                height=800,
                width=1400,
            )
            wandb.log({"plots/scatter_matrix": wandb.Html(plotly.io.to_html(fig))})

        self.n_features = self.sparse_autoencoder.cfg.d_sae
        id_to_start_from = self.get_index_to_resume_from()
        id_to_end_at = self.n_features if self.final_index is None else self.final_index
        assert id_to_end_at is not None  # keep pyright happy

        # divide into batches
        feature_idx = torch.tensor(range(id_to_start_from, id_to_end_at))
        feature_idx = feature_idx.reshape(-1, self.n_features_at_a_time)
        feature_idx = [x.tolist() for x in feature_idx]

        print(f"Hook Point Layer: {self.sparse_autoencoder.cfg.hook_point_layer}")
        print(f"Hook Point: {self.sparse_autoencoder.cfg.hook_point}")
        print(f"Writing files to: {self.dashboard_folder}")

        # get tokens:
        start = time.time()
        tokens = self.get_tokens(
            self.n_batches_to_sample_from, self.n_prompts_to_select
        )
        end = time.time()
        print(f"Time to get tokens: {end - start}")
        if self.use_wandb:
            wandb.log({"time/time_to_get_tokens": end - start})

        assert self.model is not None
        vocab_dict = cast(Any, self.model.tokenizer).vocab
        vocab_dict = {
            v: k.replace("Ġ", " ").replace("\n", "\\n") for k, v in vocab_dict.items()
        }

        with torch.no_grad():
            for interesting_features in tqdm(feature_idx):
                print(interesting_features)

                layout = SaeVisLayoutConfig(
                    columns=[
                        Column(
                            SequencesConfig(
                                stack_mode="stack-all",
                                buffer=(self.buffer_tokens, self.buffer_tokens),
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
                    features=interesting_features,
                    verbose=True,
                    feature_centric_layout=layout,
                )

                feature_data = get_feature_data(
                    encoder=self.sparse_autoencoder,  # type: ignore
                    model=self.model,
                    tokens=tokens,
                    cfg=feature_vis_params,
                )

                for i, test_idx in enumerate(feature_data.feature_data_dict.keys()):
                    feature_data.save_feature_centric_vis(
                        f"{self.dashboard_folder}/data_{test_idx:04}.html",
                        feature_idx=test_idx,
                    )

                    if i < 10 and self.use_wandb:
                        # upload the html as an artifact
                        artifact = wandb.Artifact(f"feature_{test_idx}", type="feature")
                        artifact.add_file(
                            f"{self.dashboard_folder}/data_{test_idx:04}.html"
                        )
                        assert run is not None  # keep pyright happy
                        run.log_artifact(artifact)

                        # also upload as html to dashboard
                        wandb.log(
                            {
                                "features/feature_dashboard": wandb.Html(
                                    f"{self.dashboard_folder}/data_{test_idx:04}.html"
                                )
                            },
                            step=test_idx,
                        )

        if self.use_wandb:
            # when done zip the folder
            shutil.make_archive(self.dashboard_folder, "zip", self.dashboard_folder)

            # then upload the zip as an artifact
            artifact = wandb.Artifact("dashboard", type="zipped_feature_dashboards")
            artifact.add_file(f"{self.dashboard_folder}.zip")
            assert run is not None  # keep pyright happy
            run.log_artifact(artifact)

            # terminate the run
            run.finish()

            # delete the dashboard folder
            shutil.rmtree(self.dashboard_folder)

        return
