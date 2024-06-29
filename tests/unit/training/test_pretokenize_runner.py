import json
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_lens import __version__
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import pretokenize_dataset, pretokenize_runner


@pytest.fixture
def ts_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")


def test_pretokenize_dataset_concatenates_text_until_context_size(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token=None,
        sequence_separator_token=None,
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert tokenized_dataset["input_ids"].shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "hello worldhello worldhello worldhello worldhello world"
    )


def test_pretokenize_dataset_can_add_bos_tokens_to_the_start_of_each_batch(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token="bos",
        sequence_separator_token=None,
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert tokenized_dataset["input_ids"].shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "<|endoftext|>hello worldhello worldhello worldhello worldhello"
    )
    for batch in tokenized_dataset["input_ids"]:
        assert ts_tokenizer.decode(batch[0]) == "<|endoftext|>"


def test_pretokenize_dataset_can_separate_sequences_with_bos(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token=None,
        sequence_separator_token="bos",
        begin_sequence_token=None,
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert tokenized_dataset["input_ids"].shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "hello world<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>hello"
    )


def test_pretokenize_dataset_can_begin_sequences_with_bos(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token=None,
        sequence_separator_token=None,
        begin_sequence_token="bos",
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert tokenized_dataset["input_ids"].shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>"
    )


def test_pretokenize_dataset_dedupes_bos(
    ts_tokenizer: PreTrainedTokenizerBase,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 30)
    cfg = PretokenizeRunnerConfig(
        context_size=10,
        num_proc=1,
        shuffle=False,
        begin_batch_token="bos",
        sequence_separator_token="bos",
        begin_sequence_token="bos",
    )

    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert tokenized_dataset["input_ids"].shape[1] == cfg.context_size
    assert (
        ts_tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>hello world<|endoftext|>"
    )


def test_pretokenize_dataset_can_shuffle(ts_tokenizer: PreTrainedTokenizerBase):
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 5000
    )
    cfg = PretokenizeRunnerConfig(context_size=10, num_proc=1, shuffle=True)

    # assert ts_model.tokenizer is not None
    tokenized_dataset1 = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    tokenized_dataset2 = cast(Any, pretokenize_dataset(dataset, ts_tokenizer, cfg))
    assert len(tokenized_dataset1) == len(tokenized_dataset2)
    assert (
        tokenized_dataset1["input_ids"].tolist()
        != tokenized_dataset2["input_ids"].tolist()
    )


def test_pretokenize_runner_save_dataset_locally(tmp_path: Path):
    save_path = tmp_path / "ds"
    cfg = PretokenizeRunnerConfig(
        tokenizer_name="gpt2",
        context_size=10,
        num_proc=2,
        shuffle=True,
        save_path=str(save_path),
        dataset_path="NeelNanda/c4-10k",
        split="train[:20]",
        begin_batch_token="bos",
        sequence_separator_token="eos",
    )
    dataset = pretokenize_runner(cfg)
    assert save_path.exists()
    loaded_dataset = Dataset.load_from_disk(str(save_path))
    assert len(dataset) == len(loaded_dataset)
    assert dataset["input_ids"].tolist() == loaded_dataset["input_ids"].tolist()  # type: ignore
    with open(save_path / "sae_lens.json") as f:
        metadata_dict = json.load(f)
    assert metadata_dict["original_dataset"] == "NeelNanda/c4-10k"
    assert metadata_dict["original_split"] == "train[:20]"
    assert metadata_dict["context_size"] == 10
    assert metadata_dict["shuffled"] is True
    assert metadata_dict["begin_batch_token"] == "bos"
    assert metadata_dict["begin_sequence_token"] is None
    assert metadata_dict["sequence_separator_token"] == "eos"
    assert metadata_dict["sae_lens_version"] == __version__
