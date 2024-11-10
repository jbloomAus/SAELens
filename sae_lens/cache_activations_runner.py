import io
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import einops
import torch
from datasets import Array2D, Dataset, Features
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from jaxtyping import Float
from tqdm import tqdm

from sae_lens.config import DTYPE_MAP, CacheActivationsRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.model = load_model(
            model_class_name=cfg.model_class_name,
            model_name=cfg.model_name,
            device=cfg.device,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )
        self.activations_store = ActivationsStore.from_config(
            self.model,
            cfg,
        )
        ctx_size = _get_sliced_context_size(self.cfg)
        self.features = Features(
            {
                f"{self.cfg.hook_name}": Array2D(
                    shape=(ctx_size, self.cfg.d_in), dtype=self.cfg.dtype
                )
            }
        )
        self.tokens_in_buffer = (
            self.cfg.n_batches_in_buffer * self.cfg.store_batch_size_prompts * ctx_size
        )
        self.n_buffers = math.ceil(self.cfg.training_tokens / self.tokens_in_buffer)

    def __str__(self):
        """
        Print the number of tokens to be cached.
        Print the number of buffers, and the number of tokens per buffer.
        Print the disk space required to store the activations.

        """

        bytes_per_token = (
            self.cfg.d_in * self.cfg.dtype.itemsize
            if isinstance(self.cfg.dtype, torch.dtype)
            else DTYPE_MAP[self.cfg.dtype].itemsize
        )
        total_training_tokens = self.cfg.training_tokens
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {self.n_buffers}\n"
            f"Tokens per buffer: {self.tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not includeing _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: Float[torch.Tensor, "(bs context_size) num_layers d_in"],
    ) -> Dataset:
        hook_names = [self.cfg.hook_name]  # allow multiple hooks in future

        buffer = einops.rearrange(
            buffer,
            "(bs context_size) num_layers d_in -> num_layers bs context_size d_in",
            bs=self.cfg.n_batches_in_buffer * self.cfg.store_batch_size_prompts,
            context_size=_get_sliced_context_size(self.cfg),
            d_in=self.cfg.d_in,
            num_layers=len(hook_names),
        )
        shard = Dataset.from_dict(
            {hook_name: act for hook_name, act in zip(hook_names, buffer)},
            features=self.features,
        )
        return shard

    @torch.no_grad()
    def run(self) -> Dataset:
        ### Paths setup
        assert self.cfg.new_cached_activations_path is not None
        final_cached_activation_path = Path(self.cfg.new_cached_activations_path)
        final_cached_activation_path.mkdir(exist_ok=True, parents=True)
        if any(final_cached_activation_path.iterdir()):
            raise Exception(
                f"Activations directory ({final_cached_activation_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )

        tmp_cached_activation_path = final_cached_activation_path / ".tmp_shards/"
        tmp_cached_activation_path.mkdir(exist_ok=False, parents=False)

        ### Create temporary sharded datasets

        print(f"Started caching {self.cfg.training_tokens} activations")

        for i in tqdm(range(self.n_buffers), desc="Caching activations"):
            try:
                # num activations in a single shard: n_batches_in_buffer * store_batch_size_prompts
                buffer = self.activations_store.get_buffer(
                    self.cfg.n_batches_in_buffer, shuffle=self.cfg.shuffle
                )
                shard = self._create_shard(buffer)
                shard.save_to_disk(
                    f"{tmp_cached_activation_path}/shard_{i:05d}", num_shards=1
                )
                del buffer, shard

            except StopIteration:
                print(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {self.n_buffers} batches. No more caching will occur."
                )
                break

        ### Concat sharded datasets together, shuffle and push to hub

        dataset = self._consolidate_shards(
            tmp_cached_activation_path, final_cached_activation_path, copy_files=False
        )

        if self.cfg.shuffle:
            print("Shuffling...")
            dataset = dataset.shuffle(seed=self.cfg.seed)

        if self.cfg.hf_repo_id:
            print("Pushing to hub...")
            dataset.push_to_hub(
                repo_id=self.cfg.hf_repo_id,
                num_shards=self.cfg.hf_num_shards or self.n_buffers,
                private=self.cfg.hf_is_private_repo,
                revision=self.cfg.hf_revision,
            )

            meta_io = io.BytesIO()
            meta_contents = json.dumps(
                asdict(self.cfg), indent=2, ensure_ascii=False
            ).encode("utf-8")
            meta_io.write(meta_contents)
            meta_io.seek(0)

            api = HfApi()
            api.upload_file(
                path_or_fileobj=meta_io,
                path_in_repo="cache_activations_runner_cfg.json",
                repo_id=self.cfg.hf_repo_id,
                repo_type="dataset",
                commit_message="Add cache_activations_runner metadata",
            )

        return dataset


def _get_sliced_context_size(cfg: CacheActivationsRunnerConfig) -> int:
    context_size = cfg.context_size
    if cfg.seqpos_slice:
        context_size = len(range(context_size)[slice(*cfg.seqpos_slice)])
    return context_size
