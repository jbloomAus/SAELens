"""Utility functions for creating an instance of RAVEL for a target LM."""

import collections
import random
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class RAVELMetadata:
    """Metadata for instantiating a RAVEL instance."""

    instance: str
    entity_to_split: dict
    attr_to_prompt: dict
    attr_prompt_to_split: dict
    entity_prompt_to_split: dict
    prompt_to_output: dict
    split_to_entities: dict = None

    def get_entities(self, split):
        if not self.split_to_entities:
            self.split_to_entities = {}
        if split not in self.split_to_entities:
            self.split_to_entities[split] = [
                k for k, v in self.entity_to_split.items() if v == split
            ]
        return self.split_to_entities[split]

    def sample_entities(self, split, k):
        if not self.split_to_entities:
            self.split_to_entities = {}
        if split not in self.split_to_entities:
            self.split_to_entities[split] = [
                k for k, v in self.entity_to_split.items() if v == split
            ]
        return random.sample(
            self.split_to_entities[split], k=min(k, len(self.split_to_entities[split]))
        )


def gen_context_test_split(metadata, extract_label_fn, filter_example_fn, first_n=256):
    eval_split_to_raw_example = {}
    # For each base prompts, sample entities and source examples.
    for prompt, (attr, split) in metadata.attr_prompt_to_split.items():
        if split == "train" or attr == "Other":
            continue
        base_task_inputs = [
            ((prompt, entity), metadata.prompt_to_output[prompt % entity])
            for entity in metadata.get_entities("train")
        ]
        if len(base_task_inputs) < 5:
            print(f"SKIP - NOT ENOUGH BASE EXAMPLES: {subsplit} {prompt}")
            continue
        random.shuffle(base_task_inputs)
        # We include three types of source prompts:
        # output (source entity has causal effect on the last token +
        # source output is the label)
        # causal (source entity has causal effect on the last token)
        # other  (source entity has no causal effect on the last token)
        subsplits_filter = {
            "output": lambda x: attr == x,
            "causal": lambda x: attr != x and x != "Other",
            "other": lambda x: x == "Other",
        }
        for subsplit, filter_fn in subsplits_filter.items():
            # Sample source examples.
            source_task_inputs = []
            for source_prompt, (
                source_attr,
                source_split,
            ) in metadata.attr_prompt_to_split.items():
                if not (source_split == split and filter_fn(source_attr)):
                    continue
                source_entities = []
                if (
                    source_attr == "Other"
                    and metadata.entity_prompt_to_split[source_prompt]["entity"]
                ):
                    source_entities.append(
                        metadata.entity_prompt_to_split[source_prompt]["entity"]
                    )
                else:
                    source_entities.extend(metadata.sample_entities("train", k=100))
                source_task_inputs.extend(
                    [
                        (
                            (source_prompt, entity),
                            metadata.prompt_to_output[source_prompt % entity],
                        )
                        for entity in source_entities
                        if (
                            entity in metadata.get_entities("train")
                            and len(metadata.prompt_to_output[source_prompt % entity])
                            > 1
                        )
                    ]
                )
            # Random sample weighted by output label distribution.
            source_task_inputs_label = [
                extract_label_fn(metadata.prompt_to_output[prompt % s_a])
                for (_, s_a), _ in source_task_inputs
            ]
            source_label_counters = collections.Counter(source_task_inputs_label)
            source_task_inputs_weights = [
                1 / (20 + source_label_counters[x]) for x in source_task_inputs_label
            ]
            source_task_inputs_weights = np.array(source_task_inputs_weights) / np.sum(
                source_task_inputs_weights
            )
            if len(source_task_inputs) < 5:
                print(f"SKIP {subsplit} {prompt}")
                continue
            eval_split_to_raw_example[f"{prompt}-{subsplit}-{split}"] = [
                {
                    "input": p % a,
                    "label": extract_label_fn(v),
                    "source_input": s_p % s_a,
                    "source_label": extract_label_fn(source_v),
                    "inv_label": extract_label_fn(metadata.prompt_to_output[p % s_a]),
                    "split": p,
                    "source_split": s_p,
                    "entity": a,
                    "source_entity": s_a,
                }
                for (p, a), v in base_task_inputs
                for (s_p, s_a), source_v in random.choices(
                    source_task_inputs,
                    weights=source_task_inputs_weights,
                    k=max(1, round(first_n / len(base_task_inputs))),
                )
                if filter_example_fn(v, metadata.prompt_to_output[p % s_a])
            ]
            print(
                attr,
                prompt,
                split,
                len(base_task_inputs),
                len(
                    set(
                        [
                            e["entity"]
                            for e in eval_split_to_raw_example[
                                f"{prompt}-{subsplit}-{split}"
                            ]
                        ]
                    )
                ),
                len(
                    set(
                        [
                            e["source_entity"]
                            for e in eval_split_to_raw_example[
                                f"{prompt}-{subsplit}-{split}"
                            ]
                        ]
                    )
                ),
            )
    eval_split_to_raw_example = {
        k: v for k, v in eval_split_to_raw_example.items() if len(v) > 0
    }
    return eval_split_to_raw_example


def gen_entity_test_split(metadata, extract_label_fn, filter_example_fn, first_n=256):
    eval_split_to_raw_example = {}
    for prompt, (attr, orig_split) in metadata.attr_prompt_to_split.items():
        if orig_split != "train" or attr == "Other":
            continue
        for split in ("test", "val"):
            base_task_inputs = [
                ((prompt, entity), metadata.prompt_to_output[prompt % entity])
                for entity in metadata.sample_entities(split, k=first_n)
            ]
            # We include three types of source prompts:
            # output (source entity has causal effect on the last token +
            # source output is the label)
            # causal (source entity has causal effect on the last token)
            # other  (source entity has no causal effect on the last token)
            subsplits_filter = {
                "output": lambda x: attr == x,
                "causal": lambda x: attr != x and x != "Other",
                "other": lambda x: x == "Other",
            }
            for subsplit, filter_fn in subsplits_filter.items():
                source_task_inputs = [
                    (
                        (source_prompt, entity),
                        metadata.prompt_to_output[source_prompt % entity],
                    )
                    for source_prompt, (
                        source_attr,
                        source_split,
                    ) in metadata.attr_prompt_to_split.items()
                    if source_split == "train" and filter_fn(source_attr)
                    for entity in (
                        [metadata.entity_prompt_to_split[source_prompt]["entity"]]
                        if source_attr == "Other"
                        and metadata.entity_prompt_to_split[source_prompt]["entity"]
                        else metadata.sample_entities(split, k=100)
                    )
                    if entity in metadata.get_entities(split)
                    and (len(metadata.prompt_to_output[source_prompt % entity]) > 1)
                ]
                # Random sample need to be weighted by output label distribution
                source_task_inputs_label = [
                    extract_label_fn(metadata.prompt_to_output[prompt % s_a])
                    for (_, s_a), _ in source_task_inputs
                ]
                source_label_counters = collections.Counter(source_task_inputs_label)
                source_task_inputs_weights = [
                    1 / (10 + source_label_counters[x])
                    for x in source_task_inputs_label
                ]
                source_task_inputs_weights = np.array(
                    source_task_inputs_weights
                ) / np.sum(source_task_inputs_weights)
                if len(base_task_inputs) < 5 or len(source_task_inputs) < 5:
                    continue
                print(attr, prompt, split, len(base_task_inputs))
                eval_split_to_raw_example[f"{prompt}-{subsplit}-{split}"] = [
                    {
                        "input": p % a,
                        "label": extract_label_fn(v),
                        "source_input": s_p % s_a,
                        "source_label": extract_label_fn(source_v),
                        "inv_label": extract_label_fn(
                            metadata.prompt_to_output[p % s_a]
                        ),
                        "split": p,
                        "source_split": s_p,
                        "entity": a,
                        "source_entity": s_a,
                    }
                    for (p, a), v in base_task_inputs
                    for (s_p, s_a), source_v in random.choices(
                        source_task_inputs,
                        weights=source_task_inputs_weights,
                        k=max(1, round(first_n / len(base_task_inputs))),
                    )
                    if filter_example_fn(v, metadata.prompt_to_output[p % s_a])
                ]
    eval_split_to_raw_example = {
        k: v for k, v in eval_split_to_raw_example.items() if len(v) > 0
    }
    return eval_split_to_raw_example


def gen_train_split(metadata, extract_label_fn, filter_example_fn, first_n=256):
    split_to_raw_example = {}
    # Group by attributes.
    target_split = "train"
    for attr, prompt_to_split in metadata.attr_to_prompt.items():
        base_prompt_candiates = [
            p for p, s in prompt_to_split.items() if s == target_split
        ]
        base_task_inputs = [
            ((prompt, entity), metadata.prompt_to_output[prompt % entity])
            for entity in metadata.get_entities(target_split)
            for prompt in random.sample(
                base_prompt_candiates, k=min(10, len(base_prompt_candiates))
            )
        ]
        source_task_inputs = [
            ((source_prompt, entity), metadata.prompt_to_output[source_prompt % entity])
            for source_prompt, (
                source_attr,
                source_split,
            ) in metadata.attr_prompt_to_split.items()
            if source_split == target_split and source_attr != "Other"
            for entity in metadata.sample_entities(target_split, k=10)
        ]
        wiki_source_task_inputs = [
            ((source_prompt, entity), metadata.prompt_to_output[source_prompt % entity])
            for source_prompt, split_and_arg in metadata.entity_prompt_to_split.items()
            if split_and_arg["split"] == target_split
            and (
                split_and_arg["entity"] is None
                or split_and_arg["entity"] in metadata.get_entities(target_split)
            )
            for entity in (
                [split_and_arg["entity"]]
                if split_and_arg["entity"]
                else metadata.sample_entities(target_split, k=10)
            )
        ]
        source_task_inputs = source_task_inputs + wiki_source_task_inputs
        if len(base_task_inputs) < 5 or len(source_task_inputs) < 5:
            continue
        print(
            attr,
            target_split,
            len(base_task_inputs),
            len(source_task_inputs),
            len(wiki_source_task_inputs),
        )
        split_to_raw_example[f"{attr}-{target_split}"] = []
        for (p, a), v in base_task_inputs:
            source_input_candiates = [
                x
                for x in source_task_inputs
                if filter_example_fn(v, metadata.prompt_to_output[p % x[0][1]])
                and (len(x[1]) > 1)
            ]
            split_to_raw_example[f"{attr}-{target_split}"].extend(
                [
                    {
                        "input": p % a,
                        "label": extract_label_fn(v),
                        "source_input": s_p % s_a,
                        "source_label": extract_label_fn(source_v),
                        "inv_label": extract_label_fn(
                            metadata.prompt_to_output[p % s_a]
                        ),
                        "split": p,
                        "source_split": s_p,
                        "entity": a,
                        "source_entity": s_a,
                    }
                    for (s_p, s_a), source_v in random.sample(
                        source_input_candiates,
                        k=min(
                            len(source_input_candiates),
                            round(first_n / len(base_task_inputs)),
                        ),
                    )
                ]
            )
    split_to_raw_example = {k: v for k, v in split_to_raw_example.items() if len(v) > 0}
    return split_to_raw_example
