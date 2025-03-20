# type: ignore
# ruff: noqa: T201
from pathlib import Path
from textwrap import dedent

import pandas as pd
import yaml
from tqdm import tqdm

from sae_lens import SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    SAEConfigLoadOptions,
    get_sae_config,
    handle_config_defaulting,
)

INCLUDED_CFG = [
    "id",
    "architecture",
    "neuronpedia",
    "hook_name",
    "hook_layer",
    "d_sae",
    "context_size",
    "dataset_path",
    "normalize_activations",
]


def on_pre_build(config):  # noqa: ARG001
    print("Generating SAE table...")
    generate_sae_table()
    print("SAE table generation complete.")


def generate_sae_table():
    # Read the YAML file
    yaml_path = Path("sae_lens/pretrained_saes.yaml")
    with open(yaml_path) as file:
        data = yaml.safe_load(file)

    # Start the Markdown content
    markdown_content = "# Pretrained SAEs\n\n"
    markdown_content += "This is a list of SAEs importable from the SAELens package. Click on each link for more details.\n\n"  # Added newline
    markdown_content += "*This file contains the contents of `sae_lens/pretrained_saes.yaml` in Markdown*\n\n"

    # Generate content for each model
    for release, model_info in tqdm(data.items()):
        repo_link = f"https://huggingface.co/{model_info['repo_id']}"
        markdown_content += f"## [{release}]({repo_link})\n\n"
        markdown_content += f"- **Huggingface Repo**: {model_info['repo_id']}\n"
        markdown_content += f"- **model**: {model_info['model']}\n"

        if "links" in model_info:
            markdown_content += "- **Additional Links**:\n"
            for link_type, url in model_info["links"].items():
                markdown_content += f"    - [{link_type.capitalize()}]({url})\n"

        markdown_content += "\n"

        for info in tqdm(model_info["saes"]):
            # can remove this by explicitly overriding config in yaml. Do this later.
            sae_id = info["id"]
            cfg = get_sae_config(
                release,
                sae_id=sae_id,
                options=SAEConfigLoadOptions(),
            )
            cfg = handle_config_defaulting(cfg)
            cfg = SAEConfig.from_dict(cfg).to_dict()

            if "neuronpedia" not in info:
                info["neuronpedia"] = None

            info.update(cfg)

        df = pd.DataFrame(model_info["saes"])

        # Keep only 'id' and 'path' columns
        df = df[INCLUDED_CFG]

        def style_id(val):
            return f"<div>{val}</div><a class=\"saetable-loadSaeId\" onclick=\"SaeTable.showCode('{release}', '{val}')\">Load this SAE</a>"

        df["id"] = df["id"].apply(style_id)
        table = df.to_markdown(index=False)
        markdown_content += table + "\n\n"
    markdown_content += dedent(
        """
        <div id="codeModal" class="saetable-modal">
            <div class="saetable-modalContent">
                <span class="saetable-close" onclick="SaeTable.closeCode()">&times;</span>
                <pre><code id="codeContent" onclick="SaeTable.selectCode(this)"></code></pre>
                <button onclick="SaeTable.copyCode()" class="saetable-copyButton">Copy Code</button>
            </div>
        </div>
        """
    )
    # Write the content to a Markdown file
    output_path = Path("docs/sae_table.md")
    with open(output_path, "w") as file:
        file.write(markdown_content)


if __name__ == "__main__":
    generate_sae_table()
