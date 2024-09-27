# type: ignore
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from sae_lens import SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    get_connor_rob_hook_z_layer_config,
    get_dictionary_learning_config_1,
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
            repo_id = model_info["repo_id"]
            folder_name = info["path"]
            # can remove this by explicitly overriding config in yaml. Do this later.
            if model_info["conversion_func"] == "connor_rob_hook_z":
                cfg = get_connor_rob_hook_z_layer_config(
                    repo_id, folder_name=folder_name, device=None
                )
                cfg = handle_config_defaulting(cfg)
                cfg = SAEConfig.from_dict(cfg).to_dict()
                info.update(cfg)
            elif model_info["conversion_func"] == "dictionary_learning_1":
                cfg = get_dictionary_learning_config_1(repo_id, folder_name=folder_name)
                cfg = SAEConfig.from_dict(cfg).to_dict()

            elif model_info["conversion_func"] == "gemma_2":
                cfg = get_gemma_2_config(repo_id, folder_name=folder_name)
                cfg = handle_config_defaulting(cfg)
                cfg = SAEConfig.from_dict(cfg).to_dict()
                info.update(cfg)
            else:
                cfg = get_sae_config_from_hf(
                    repo_id,
                    folder_name=folder_name,
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
