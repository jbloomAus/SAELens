from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

from sae_lens.training.config import PretokenizeRunnerConfig
from sae_lens.training.pretokenize_runner import pretokenize_dataset, pretokenize_runner


def test_pretokenize_dataset_concatenates_text_until_context_size(
    ts_model: HookedTransformer,
):
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    cfg = PretokenizeRunnerConfig(context_size=10, num_proc=1, shuffle=False)

    assert ts_model.tokenizer is not None
    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_model.tokenizer, cfg))
    assert tokenized_dataset["input_ids"].shape[1] == cfg.context_size
    assert (
        ts_model.tokenizer.decode(tokenized_dataset["input_ids"][0])
        == "hello worldhello worldhello worldhello worldhello world"
    )


def test_pretokenize_dataset_matches_tlens(ts_model: HookedTransformer):
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 5000
    )
    cfg = PretokenizeRunnerConfig(context_size=10, num_proc=1, shuffle=False)

    assert ts_model.tokenizer is not None
    tokenized_dataset = cast(Any, pretokenize_dataset(dataset, ts_model.tokenizer, cfg))

    tl_tokenized_res = cast(
        Any,
        tokenize_and_concatenate(
            dataset,
            cast(Any, ts_model.tokenizer),
            max_length=cfg.context_size,
            add_bos_token=False,
        ),
    )

    assert (
        tokenized_dataset["input_ids"].tolist() == tl_tokenized_res["tokens"].tolist()
    )


def test_pretokenize_dataset_can_shuffle(ts_model: HookedTransformer):
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 5000
    )
    cfg = PretokenizeRunnerConfig(context_size=10, num_proc=1, shuffle=True)

    assert ts_model.tokenizer is not None
    tokenized_dataset1 = cast(
        Any, pretokenize_dataset(dataset, ts_model.tokenizer, cfg)
    )
    tokenized_dataset2 = cast(
        Any, pretokenize_dataset(dataset, ts_model.tokenizer, cfg)
    )
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
    )
    dataset = pretokenize_runner(cfg)
    assert save_path.exists()
    loaded_dataset = Dataset.load_from_disk(str(save_path))
    assert len(dataset) == len(loaded_dataset)
    assert dataset["input_ids"].tolist() == loaded_dataset["input_ids"].tolist()  # type: ignore
