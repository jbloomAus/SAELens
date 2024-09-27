import collections
import datetime
import json
import os
import pickle as pkl
import random
import re
from zoneinfo import ZoneInfo

import datasets
import h5py
import numpy as np
import torch
from datasets import Dataset
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.generate_ravel_instance import RAVELMetadata
from utils.generation_utils import generate_batched

# Constants
REPO_DIR = "/share/u/can/ravel"
SRC_DIR = os.path.join(REPO_DIR, "src")
MODEL_DIR = os.path.join(REPO_DIR, "models")
DATA_DIR = os.path.join(REPO_DIR, "data")
INPUT_MAX_LEN = 48


def setup_environment():
    """
    Set up the environment by creating necessary directories and setting the random seed.
    """
    for d in [MODEL_DIR, DATA_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    import sys

    sys.path.append(REPO_DIR)
    sys.path.append(SRC_DIR)

    set_seed(0)


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_id, model_name):
    """
    Load the model and tokenizer.

    Args:
        model_id (str): The ID of the model to load.
        model_name (str): The name of the model.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    with open("/share/u/can/src/hf.txt", "r") as f:
        hf_token = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)  # avoid blowing up mem
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=MODEL_DIR,
        token=hf_token,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=MODEL_DIR,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return hf_model, tokenizer


def load_ravel_data(entity_type):
    """
    Load RAVEL dataset files.

    Args:
        entity_type (str): The type of entity (e.g., 'city').

    Returns:
        tuple: Loaded data structures.
    """
    attribute_prompts = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"ravel_{entity_type}_attribute_to_prompts.json"
            )
        )
    )
    prompt_splits = json.load(
        open(
            os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_prompt_to_split.json")
        )
    )
    entity_attributes = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"ravel_{entity_type}_entity_attributes.json"
            )
        )
    )
    ALL_ENTITY_SPLITS = json.load(
        open(
            os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_entity_to_split.json")
        )
    )
    ALL_ATTR_TO_PROMPTS = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"ravel_{entity_type}_attribute_to_prompts.json"
            )
        )
    )
    WIKI_PROMPT_SPLITS = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"wikipedia_{entity_type}_entity_prompts.json"
            )
        )
    )

    return (
        attribute_prompts,
        prompt_splits,
        entity_attributes,
        ALL_ENTITY_SPLITS,
        ALL_ATTR_TO_PROMPTS,
        WIKI_PROMPT_SPLITS,
    )


def generate_prompts_and_metadata(attribute_prompts, prompt_splits, entity_attributes):
    """
    Generate prompts and metadata for the RAVEL dataset.

    Args:
        attribute_prompts (dict): Mapping of attributes to prompts.
        prompt_splits (dict): Mapping of prompts to splits.
        entity_attributes (dict): Mapping of entities to their attributes.

    Returns:
        tuple: Generated prompts and metadata.
    """
    prompts_to_meta_data = {
        t % x: {"entity": x, "attr": a, "template": t}
        for x in entity_attributes
        for a, ts in attribute_prompts.items()
        for t in ts
    }

    return prompts_to_meta_data


def generate_outputs(hf_model, tokenizer, prompts, max_new_tokens=8, batch_size=64):
    """
    Generate outputs for given prompts using the model.

    Args:
        hf_model: The HuggingFace model.
        tokenizer: The tokenizer.
        prompts (list): List of input prompts.
        max_new_tokens (int): Maximum number of new tokens to generate.
        batch_size (int): Batch size for generation.

    Returns:
        dict: Mapping of prompts to their generated outputs.
    """

    prompt_and_output = generate_batched(
        hf_model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    return {k: v[len(k) :] for k, v in prompt_and_output.items()}


def extract_label(text):
    """
    Extract the label from generated text.

    Args:
        text (str): The generated text.

    Returns:
        str: The extracted label.
    """
    tokens = re.split(r'(["]|[.,;]\s|\n| \(|\sand)', text + " ")
    x = tokens[0]
    digit_match = re.search(r"\.\d\d", x)
    if digit_match:
        x = x[: digit_match.span(0)[1]]
    gender_match = re.match(r"\s?(his|her|himself|herself|she|he)[^\w]", x)
    if gender_match:
        x = x[: gender_match.span(1)[1]]
    if not x.strip():
        x = " ".join(text.split(" ")[:2]).rstrip('.,"\n')
    assert x.strip()
    return x


def filter_inv_example(base_output, inv_output):
    """
    Filter invariant examples based on output comparison.

    Args:
        base_output (str): The base output.
        inv_output (str): The invariant output.

    Returns:
        bool: True if the example passes the filter, False otherwise.
    """

    def get_first_token(x):
        return re.split(r"[^\w\+\-]", x.strip(), re.UNICODE)[0]

    different_outputs = get_first_token(base_output) != get_first_token(inv_output)
    valid_outputs = re.fullmatch(
        r"\s?[a-z0-9.:\-+]+", extract_label(base_output), re.IGNORECASE
    ) and re.fullmatch(r"\s?[a-z0-9.:\-+]+", extract_label(inv_output), re.IGNORECASE)
    return valid_outputs and different_outputs


def generate_split(
    metadata, extract_label_fn, filter_example_fn, split_type, first_n=256
):
    """
    Generate a split (context or entity) for the RAVEL dataset.

    Args:
        metadata (RAVELMetadata): The RAVEL metadata.
        extract_label_fn (function): Function to extract labels.
        filter_example_fn (function): Function to filter examples.
        split_type (str): Type of split ('context' or 'entity').
        first_n (int): Number of examples to keep per split.

    Returns:
        dict: The generated split.
    """
    if split_type == "context":
        from utils.generate_ravel_instance import gen_context_test_split

        return gen_context_test_split(
            metadata, extract_label_fn, filter_example_fn, first_n
        )
    elif split_type == "entity":
        from utils.generate_ravel_instance import gen_entity_test_split

        return gen_entity_test_split(
            metadata, extract_label_fn, filter_example_fn, first_n
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")


def merge_subsplits(split_to_raw_example):
    """
    Merge subsplits in the generated split.

    Args:
        split_to_raw_example (dict): The split with subsplits.

    Returns:
        dict: The merged split.
    """
    merged = collections.defaultdict(list)
    for split in split_to_raw_example:
        merged[re.sub(r"-causal|-output|-other", "", split)].extend(
            split_to_raw_example[split]
        )
    return dict(merged)


def extract_features(
    nnsight_model,
    layer_idx,
    entity_to_split,
    attribute_to_prompt_and_split,
    output_path,
    batch_size=64,
):
    """
    Extract features from the model for the RAVEL dataset.

    Args:
        nnsight_model: The NNsight model.
        layer_idx (int): The index of the layer to extract features from.
        entity_to_split (dict): Mapping of entities to their splits.
        attribute_to_prompt_and_split (dict): Mapping of attributes to prompts and splits.
        output_path (str): Path to save the extracted features.
        batch_size (int): Batch size for processing.
    """

    def get_resid_post_activations(nnsight_model, layer_idx, encoded_input):
        submodule = nnsight_model.model.layers[layer_idx]
        with torch.no_grad(), nnsight_model.trace(
            encoded_input.input_ids.to(device),
            attention_mask=encoded_input.attention_mask.to(device),
            **nnsight_tracer_kwargs,
        ):
            output = submodule.output[0].save()
        return output

    f_out = h5py.File(output_path, "a")
    splits = {
        "train": ("train", "train"),
        "val_entity": ("val", "train"),
        "val_context": ("train", "val"),
    }

    for split_name, (entity_split, prompt_split) in splits.items():
        for attr, prompt_to_split in attribute_to_prompt_and_split.items():
            inputs, entities, templates = zip(
                *[
                    (p[: p.index("%s")] + e, e, p)
                    for p in prompt_to_split
                    if prompt_to_split[p] == prompt_split
                    for e in entity_to_split
                    if entity_to_split[e] == entity_split
                ]
            )

            all_features = []
            for b_i in range(0, len(inputs), batch_size):
                input_batch = inputs[b_i : b_i + batch_size]
                encoded_input = tokenizer(
                    input_batch,
                    padding="max_length",
                    max_length=INPUT_MAX_LEN,
                    return_tensors="pt",
                    truncation=True,
                )
                with torch.no_grad():
                    outputs = get_resid_post_activations(
                        nnsight_model, layer_idx, encoded_input
                    )
                    for i in range(len(input_batch)):
                        all_features.append(
                            outputs[i : i + 1, -1, :].to(torch.float16).cpu().numpy()
                        )

            print(attr, split_name, np.concatenate(all_features).shape)
            f_out[f"{attr}-{split_name}"] = np.concatenate(all_features)
            f_out[f"{attr}-{split_name}" + "_input"] = np.void(pkl.dumps(inputs))
            f_out[f"{attr}-{split_name}" + "_template"] = np.void(pkl.dumps(templates))
            f_out[f"{attr}-{split_name}" + "_entity"] = np.void(pkl.dumps(entities))

    f_out.flush()
    f_out.close()


def gen_train_split(metadata, extract_label_fn, filter_example_fn, first_n=256):
    """
    Generate the train split for the RAVEL dataset.

    Args:
        metadata (RAVELMetadata): The RAVEL metadata.
        extract_label_fn (function): Function to extract labels.
        filter_example_fn (function): Function to filter examples.
        first_n (int): Number of examples to keep per split.

    Returns:
        dict: The generated train split.
    """
    split_to_raw_example = {}
    target_split = "train"
    for attr, prompt_to_split in metadata.attr_to_prompt.items():
        base_prompt_candiates = [
            p for p, s in prompt_to_split.items() if s == target_split
        ]
        base_task_inputs = [
            ((prompt, entity), metadata.prompt_to_output[prompt % entity])
            for entity in metadata.get_entities(target_split)
            for prompt in random.sample(
                base_prompt_candiates, k=min(2, len(base_prompt_candiates))
            )
        ]
        source_task_inputs = [
            ((source_prompt, entity), metadata.prompt_to_output[source_prompt % entity])
            for source_prompt, (source_attr, source_split) in KEPT_PROMPT_SPLITS.items()
            if source_split == target_split and source_attr != "Other"
            for entity in metadata.sample_entities(target_split, k=1)
        ]
        wiki_source_task_inputs = [
            ((source_prompt, entity), metadata.prompt_to_output[source_prompt % entity])
            for source_prompt, split_and_arg in metadata.entity_prompt_to_split.items()
            if split_and_arg["split"] == target_split
            for entity in (
                [split_and_arg["entity"]]
                if split_and_arg["entity"]
                else metadata.sample_entities(target_split, k=1)
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
                    if filter_example_fn(v, metadata.prompt_to_output[p % s_a])
                    and re.search("\w+", source_v)
                ]
            )
    split_to_raw_example = {k: v for k, v in split_to_raw_example.items() if len(v) > 0}
    return split_to_raw_example


def postprocess_labels(split_to_raw_example, attribute_to_prompts):
    """
    Postprocess labels in the split.

    Args:
        split_to_raw_example (dict): The split with raw examples.
        attribute_to_prompts (dict): Mapping of attributes to prompts.

    Returns:
        dict: The processed split.
    """
    for split in split_to_raw_example:
        for i in range(len(split_to_raw_example[split])):
            if (
                split.split("-")[0] in ["Latitude", "Longitude"]
                or split.split("-")[0] in attribute_to_prompts["Latitude"]
                or split.split("-")[0] in attribute_to_prompts["Longitude"]
            ):
                split_to_raw_example[split][i]["inv_label"] = (
                    split_to_raw_example[split][i]["inv_label"]
                    .replace("°", ".")
                    .split(".")[0]
                )
                split_to_raw_example[split][i]["label"] = (
                    split_to_raw_example[split][i]["label"]
                    .replace("°", ".")
                    .split(".")[0]
                )
    return split_to_raw_example


def calculate_intervention_positions(tokenizer, all_prompt_templates):
    """
    Calculate intervention positions for prompts.

    Args:
        tokenizer: The tokenizer.
        all_prompt_templates (set): Set of all prompt templates.

    Returns:
        dict: Mapping of prompts to their intervention positions.
    """
    SPLIT_TO_INV_POSITION = {}
    for prompt_template in all_prompt_templates:
        if prompt_template.count("%s") != 1:
            continue
        prompt_input = prompt_template.replace("%s", "000000", 1)
        input_ids = tokenizer(prompt_input)["input_ids"]
        toks = tokenizer.batch_decode(input_ids)
        for i in range(-1, -len(toks), -1):
            if (
                toks[i] == "0"
                and toks[i - 1] == "0"
                and toks[i - 2] == "0"
                and toks[i - 3] == "0"
            ):
                break
        SPLIT_TO_INV_POSITION[prompt_template] = i
    return SPLIT_TO_INV_POSITION


def main():
    """
    Main function to orchestrate the RAVEL dataset creation and feature extraction process.
    """
    setup_environment()

    model_id = "google/gemma-2-2b"
    model_name = "gemma-2-2b"
    entity_type = "city"

    hf_model, tokenizer = load_model_and_tokenizer(model_id, model_name)
    nnsight_model = NNsight(hf_model)
    nnsight_tracer_kwargs = {
        "scan": True,
        "validate": False,
        "use_cache": False,
        "output_attentions": False,
    }

    (
        attribute_prompts,
        prompt_splits,
        entity_attributes,
        ALL_ENTITY_SPLITS,
        ALL_ATTR_TO_PROMPTS,
        WIKI_PROMPT_SPLITS,
    ) = load_ravel_data(entity_type)

    prompts_to_meta_data = generate_prompts_and_metadata(
        attribute_prompts, prompt_splits, entity_attributes
    )

    # Generate outputs for all prompts
    prompt_to_output = generate_outputs(
        hf_model, tokenizer, list(prompts_to_meta_data.keys())
    )

    # Generate outputs for wiki prompts
    wiki_prompts = [
        (t % e)
        for t, s_e in WIKI_PROMPT_SPLITS.items()
        for e in (
            [s_e["entity"]]
            if s_e["entity"]
            else [
                a
                for a in KEPT_ENTITY_SPLITS
                if KEPT_ENTITY_SPLITS[a] == "train" or s_e["split"] == "train"
            ]
        )
    ]
    wiki_prompt_to_output = generate_outputs(hf_model, tokenizer, wiki_prompts)

    ALL_PROMPT_TO_OUTPUT = {**prompt_to_output, **wiki_prompt_to_output}

    # Define KEPT_ENTITY_SPLITS and KEPT_ATTR_TO_PROMPT_AND_SPLIT
    KEPT_ENTITY = list(ALL_ENTITY_SPLITS.keys())[
        :400
    ]  # Assuming we keep top 400 entities
    KEPT_ENTITY_SPLITS = {e: ALL_ENTITY_SPLITS[e] for e in KEPT_ENTITY}
    KEPT_ATTR_TO_PROMPT_AND_SPLIT = {
        k: {p: v for p, v in d.items() if p.count("%") == 1}
        for k, d in ALL_ATTR_TO_PROMPTS.items()
    }

    # Define KEPT_PROMPT_SPLITS
    KEPT_PROMPT_SPLITS = {
        k: (a, v)
        for a, d in KEPT_ATTR_TO_PROMPT_AND_SPLIT.items()
        for k, v in d.items()
        if k.count("%") == 1
    }
    for prompt in WIKI_PROMPT_SPLITS:
        KEPT_PROMPT_SPLITS[prompt] = ("Other", WIKI_PROMPT_SPLITS[prompt]["split"])

    # Create RAVELMetadata instance
    ravel_metadata = RAVELMetadata(
        model_name,
        KEPT_ENTITY_SPLITS,
        KEPT_ATTR_TO_PROMPT_AND_SPLIT,
        KEPT_PROMPT_SPLITS,
        WIKI_PROMPT_SPLITS,
        ALL_PROMPT_TO_OUTPUT,
    )

    # Generate splits
    for split_type in ["context", "entity"]:
        split_to_raw_example = generate_split(
            ravel_metadata, extract_label, filter_inv_example, split_type
        )
        split_to_raw_example = merge_subsplits(split_to_raw_example)
        split_to_raw_example = postprocess_labels(
            split_to_raw_example, attribute_prompts
        )

        output_json_path = os.path.join(
            DATA_DIR,
            f"{ravel_metadata.instance}/{ravel_metadata.instance}_{entity_type}_{split_type}_test.json",
        )
        json.dump(split_to_raw_example, open(output_json_path, "w"), ensure_ascii=False)

    # Generate train split
    train_split_to_raw_example = gen_train_split(
        ravel_metadata,
        extract_label_fn=extract_label,
        filter_example_fn=filter_inv_example,
        first_n=10240,
    )

    train_json_path = os.path.join(
        DATA_DIR,
        f"{ravel_metadata.instance}/{ravel_metadata.instance}_{entity_type}_train.json",
    )
    json.dump(
        train_split_to_raw_example, open(train_json_path, "w"), ensure_ascii=False
    )

    # Calculate intervention positions
    all_prompt_templates = set(WIKI_PROMPT_SPLITS) | set(
        prompt for prompts in ALL_ATTR_TO_PROMPTS.values() for prompt in prompts
    )
    SPLIT_TO_INV_POSITION = calculate_intervention_positions(
        tokenizer, all_prompt_templates
    )

    inv_position_json_path = os.path.join(
        DATA_DIR,
        model_name,
        f"{model_name}_{entity_type}_prompt_to_entity_position.json",
    )
    json.dump(
        SPLIT_TO_INV_POSITION,
        open(inv_position_json_path, "w"),
        ensure_ascii=False,
        indent=2,
    )

    # Extract features
    for layer in [10, 14]:
        output_path = os.path.join(
            DATA_DIR,
            model_name,
            f"ravel_{entity_type}_{model_name}_layer{layer}_representation.hdf5",
        )
        extract_features(
            nnsight_model,
            layer,
            KEPT_ENTITY_SPLITS,
            KEPT_ATTR_TO_PROMPT_AND_SPLIT,
            output_path,
        )


if __name__ == "__main__":
    main()
