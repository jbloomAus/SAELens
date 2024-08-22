# type: ignore
import json
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from sae_lens import SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    get_gemma_2_config,
    get_sae_config_from_hf,
    handle_config_defaulting,
)

INCLUDED_CFG = [
    "id",
    "architecture",
    "neuronpedia",
    # "model_name",
    "hook_name",
    "hook_layer",
    "d_sae",
    "context_size",
    "dataset_path",
    "normalize_activations",
]


def on_pre_build(config):
    print("Generating SAE table...")
    generate_sae_table()
    print("SAE table generation complete.")


def generate_sae_table():
    # Read the YAML file
    yaml_path = Path("sae_lens/pretrained_saes.yaml")
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Start the Markdown content
    markdown_content = "# Pretrained SAEs\n\n"
    markdown_content += "This is a list of SAEs importable from the SAELens package. Click on each link for more details.\n\n"  # Added newline
    markdown_content += "*This file contains the contents of `sae_lens/pretrained_saes.yaml` in Markdown*\n\n"

    # Generate content for each model
    for model_name, model_info in tqdm(data.items()):
        repo_link = f"https://huggingface.co/{model_info['repo_id']}"
        markdown_content += f"## [{model_name}]({repo_link})\n\n"
        markdown_content += f"- **Huggingface Repo**: {model_info['repo_id']}\n"
        markdown_content += f"- **model**: {model_info['model']}\n"

        if "links" in model_info:
            markdown_content += "- **Additional Links**:\n"
            for link_type, url in model_info["links"].items():
                markdown_content += f"    - [{link_type.capitalize()}]({url})\n"

        markdown_content += "\n"

        # get the config

        # for sae_info in model_info["saes"]:
        #     sae_cfg = get_sae_config_from_hf(
        #         model_info["repo_id"],
        #         sae_info["path"],
        #     )

        for info in tqdm(model_info["saes"]):

            # can remove this by explicitly overriding config in yaml. Do this later.
            if model_info["conversion_func"] == "connor_rob_hook_z":
                repo_id = model_info["repo_id"]
                folder_name = info["path"]
                config_path = folder_name.split(".pt")[0] + "_cfg.json"
                config_path = hf_hub_download(repo_id, config_path)
                old_cfg_dict = json.load(open(config_path, "r"))

                cfg = {
                    "architecture": "standard",
                    "d_in": old_cfg_dict["act_size"],
                    "d_sae": old_cfg_dict["dict_size"],
                    "dtype": "float32",
                    "device": "cpu",
                    "model_name": "gpt2-small",
                    "hook_name": old_cfg_dict["act_name"],
                    "hook_layer": old_cfg_dict["layer"],
                    "hook_head_index": None,
                    "activation_fn_str": "relu",
                    "apply_b_dec_to_input": True,
                    "finetuning_scaling_factor": False,
                    "sae_lens_training_version": None,
                    "prepend_bos": True,
                    "dataset_path": "Skylion007/openwebtext",
                    "context_size": 128,
                    "normalize_activations": "none",
                    "dataset_trust_remote_code": True,
                }
                cfg = handle_config_defaulting(cfg)
                cfg = SAEConfig.from_dict(cfg).to_dict()
                info.update(cfg)

            elif model_info["conversion_func"] == "gemma_2":
                repo_id = model_info["repo_id"]
                folder_name = info["path"]
                cfg = get_gemma_2_config(repo_id, folder_name)
                cfg = handle_config_defaulting(cfg)
                cfg = SAEConfig.from_dict(cfg).to_dict()
                info.update(cfg)
            else:
                cfg = get_sae_config_from_hf(
                    model_info["repo_id"],
                    info["path"],
                )
                cfg = handle_config_defaulting(cfg)
                cfg = SAEConfig.from_dict(cfg).to_dict()

            if "neuronpedia" not in info.keys():
                info["neuronpedia"] = None

            info.update(cfg)

        # cfg_to_in
        # Create DataFrame for SAEs
        df = pd.DataFrame(model_info["saes"])

        # Keep only 'id' and 'path' columns
        df = df[INCLUDED_CFG]

        # Generate Markdown table
        table = df.to_markdown(index=False)
        markdown_content += table + "\n\n"

    # Write the content to a Markdown file
    output_path = Path("docs/sae_table.md")
    with open(output_path, "w") as file:
        file.write(markdown_content)


if __name__ == "__main__":
    generate_sae_table()
