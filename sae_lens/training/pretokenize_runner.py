import sys
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_lens.training.config import PretokenizeRunnerConfig


def pretokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
):
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        cast(AutoTokenizer, tokenizer),
        streaming=False,
        max_length=cfg.context_size,
        add_bos_token=False,
        num_proc=cfg.num_proc,
    )
    if cfg.shuffle:
        tokenized_dataset = tokenized_dataset.shuffle(seed=cfg.seed)
    return tokenized_dataset.rename_column("tokens", "input_ids")


def push_to_hugging_face_hub(
    dataset: Dataset,
    cfg: PretokenizeRunnerConfig,
):
    assert cfg.hf_repo_id is not None
    return dataset.push_to_hub(
        repo_id=cfg.hf_repo_id,
        num_shards=cfg.hf_num_shards,
        private=cfg.hf_is_private_repo,
        revision=cfg.hf_revision,
    )


def pretokenize_runner(
    cfg: PretokenizeRunnerConfig,
):
    dataset = load_dataset(
        cfg.dataset_path,
        data_dir=cfg.data_dir,
        data_files=cfg.data_files,
        split=cfg.split,
        streaming=cfg.streaming,
    )
    if isinstance(dataset, DatasetDict):
        raise ValueError("Dataset has multiple splits. Must provide a 'split' param.")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.model_max_length = sys.maxsize
    tokenized_dataset = pretokenize_dataset(cast(Dataset, dataset), tokenizer, cfg)

    if cfg.save_path is not None:
        tokenized_dataset.save_to_disk(cfg.save_path)

    if cfg.hf_repo_id is not None:
        push_to_hugging_face_hub(tokenized_dataset, cfg)

    return tokenized_dataset
