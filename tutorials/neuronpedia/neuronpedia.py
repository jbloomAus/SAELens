# we use a script that launches separate python processes to work around OOM issues - this ensures every batch gets the whole system available memory
# better fix is to investigate and fix the memory issues

import json
import math
import os
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests
import torch
import typer
from rich import print
from rich.align import Align
from rich.panel import Panel
from typing_extensions import Annotated

from sae_lens.toolkit.pretrained_saes import load_sparsity
from sae_lens.training.sparse_autoencoder import SparseAutoencoder

OUTPUT_DIR_BASE = Path("../../neuronpedia_outputs")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Tool that generates features (generate) and uploads features (upload) to Neuronpedia.",
)


@app.command()
def generate(
    sae_id: Annotated[
        str,
        typer.Option(
            help="SAE ID to generate features for (must exactly match the one used on Neuronpedia). Example: res-jb",
            prompt="""
What is the SAE ID you want to generate features for?
This was set when you did 'Add SAEs' on Neuronpedia. This must exactly match that ID (including casing).
It's in the format [abbrev hook name]-[abbrev author name], like res-jb.
Enter SAE ID""",
        ),
    ],
    sae_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Absolute local path to the SAE directory (with cfg.json, sae_weights.safetensors, sparsity.safetensors).",
            prompt="""
What is the absolute local path to your SAE's directory (with cfg.json, sae_weights.safetensors, sparsity.safetensors)?
Enter path""",
        ),
    ],
    log_sparsity: Annotated[
        int,
        typer.Option(
            min=-10,
            max=0,
            help="Desired feature log sparsity threshold. Range -10 to 0.",
            prompt="""
What is your desired feature log sparsity threshold?
Enter value from -10 to 0""",
        ),
    ] = -5,
    feat_per_batch: Annotated[
        int,
        typer.Option(
            min=1,
            max=2048,
            help="Features to generate per batch. More requires more memory.",
            prompt="""
How many features do you want to generate per batch? More requires more memory.
Enter value""",
        ),
    ] = 128,
    resume_from_batch: Annotated[
        int,
        typer.Option(
            min=1,
            help="Batch number to resume from.",
            prompt="""
Do you want to resume from a specific batch number?
Enter 1 to start from the beginning""",
        ),
    ] = 1,
    n_batches_to_sample: Annotated[
        int,
        typer.Option(
            min=1,
            help="[Activation Text Generation] Number of batches to sample from.",
            prompt="""
[Activation Text Generation] How many batches do you want to sample from?
Enter value""",
        ),
    ] = 2
    ** 12,
    n_prompts_to_select: Annotated[
        int,
        typer.Option(
            min=1,
            help="[Activation Text Generation] Number of prompts to select from.",
            prompt="""
[Activation Text Generation] How many prompts do you want to select from?
Enter value""",
        ),
    ] = 4096
    * 6,
):
    """
    This will start a batch job that generates features for Neuronpedia for a specific SAE. To upload those features, use the 'upload' command afterwards.
    """

    # Check arguments
    if sae_path.is_dir() is not True:
        print("Error: SAE path must be a directory.")
        raise typer.Abort()
    if sae_path.joinpath("cfg.json").is_file() is not True:
        print("Error: cfg.json file not found in SAE directory.")
        raise typer.Abort()
    if sae_path.joinpath("sae_weights.safetensors").is_file() is not True:
        print("Error: sae_weights.safetensors file not found in SAE directory.")
        raise typer.Abort()
    if sae_path.joinpath("sparsity.safetensors").is_file() is not True:
        print("Error: sparsity.safetensors file not found in SAE directory.")
        raise typer.Abort()

    sae_path_string = sae_path.as_posix()

    # Load SAE
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    sparse_autoencoder = SparseAutoencoder.load_from_pretrained(
        sae_path_string, device=device
    )
    model_id = sparse_autoencoder.cfg.model_name

    outputs_subdir = f"{model_id}_{sae_id}_{sparse_autoencoder.cfg.hook_point}"
    outputs_dir = OUTPUT_DIR_BASE.joinpath(outputs_subdir)
    if outputs_dir.exists() and outputs_dir.is_file():
        print(f"Error: Output directory {outputs_dir.as_posix()} exists and is a file.")
        raise typer.Abort()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    # Check if output_dir has any files starting with "batch_"
    batch_files = list(outputs_dir.glob("batch-*.json"))
    if len(batch_files) > 0 and resume_from_batch == 1:
        print(
            f"Error: Output directory {outputs_dir.as_posix()} has existing batch files. This is only allowed if you are resuming from a batch. Please delete or move the existing batch-*.json files."
        )
        raise typer.Abort()

    sparsity = load_sparsity(sae_path_string)
    # convert sparsity to logged sparsity if it's not
    # TODO: standardize the sparsity file format
    if len(sparsity) > 0 and sparsity[0] >= 0:
        sparsity = torch.log10(sparsity + 1e-10)
    sparsity = sparsity.to(device)
    alive_indexes = (sparsity > log_sparsity).nonzero(as_tuple=True)[0].tolist()
    num_alive = len(alive_indexes)
    num_dead = sparse_autoencoder.d_sae - num_alive

    print("\n")
    print(
        Align.center(
            Panel.fit(
                f"""
[white]SAE Path: [green]{sae_path.as_posix()}
[white]Model ID: [green]{model_id}
[white]Hook Point: [green]{sparse_autoencoder.cfg.hook_point}
[white]Using Device: [green]{device}
""",
                title="SAE Info",
            )
        )
    )
    num_batches = math.ceil(num_alive / feat_per_batch)
    print(
        Align.center(
            Panel.fit(
                f"""
[white]Total Features: [green]{sparse_autoencoder.d_sae}
[white]Log Sparsity Threshold: [green]{log_sparsity}
[white]Alive Features: [green]{num_alive}
[white]Dead Features: [red]{num_dead}
[white]Features per Batch: [green]{feat_per_batch}
[white]Number of Batches: [green]{num_batches}
{resume_from_batch != 1 and f"[white]Resuming from Batch: [green]{resume_from_batch}" or ""}
""",
                title="Number of Features",
            )
        )
    )
    print(
        Align.center(
            Panel.fit(
                f"""
[white]Dataset: [green]{sparse_autoencoder.cfg.dataset_path}
[white]Batches to Sample From: [green]{n_batches_to_sample}
[white]Prompts to Select From: [green]{n_prompts_to_select}
""",
                title="Activation Text Settings",
            )
        )
    )
    print(
        Align.center(
            Panel.fit(
                f"""
[green]{outputs_dir.absolute().as_posix()}
""",
                title="Output Directory",
            )
        )
    )

    print(
        Align.center(
            "\n========== [yellow]Starting batch feature generations...[/yellow] =========="
        )
    )

    # iterate from 1 to num_batches
    for i in range(resume_from_batch, num_batches + 1):
        command = [
            "python",
            "make_batch.py",
            sae_id,
            sae_path.absolute().as_posix(),
            outputs_dir.absolute().as_posix(),
            str(log_sparsity),
            str(n_batches_to_sample),
            str(n_prompts_to_select),
            str(feat_per_batch),
            str(i),
            str(i),
        ]
        print("\n")
        print(
            Align.center(
                Panel.fit(
                    f"""
[yellow]{" ".join(command)}
""",
                    title="Running Command for Batch #" + str(i),
                )
            )
        )
        # make a subprocess call to python make_batch.py
        subprocess.run(
            [
                "python",
                "make_batch.py",
                sae_id,
                sae_path,
                outputs_dir,
                str(log_sparsity),
                str(n_batches_to_sample),
                str(n_prompts_to_select),
                str(feat_per_batch),
                str(i),
                str(i),
            ]
        )

    print(
        Align.center(
            Panel(
                f"""
Your Features Are In: [green]{outputs_dir.absolute().as_posix()}
Use [yellow]'neuronpedia.py upload'[/yellow] to upload your features to Neuronpedia.
""",
                title="Generation Complete",
            )
        )
    )


@app.command()
def upload(
    outputs_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            prompt="What is the absolute, full local file path to the feature outputs directory?",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            prompt="""Host to upload to? (Default: http://localhost:3000)""",
        ),
    ] = "http://localhost:3000",
):
    """
    This will upload features that were generated to Neuronpedia. It currently only works if you have admin access to a Neuronpedia instance via localhost:3000.
    """

    files_to_upload = list(outputs_dir.glob("batch-*.json"))

    # sort files by batch number
    files_to_upload.sort(key=lambda x: int(x.stem.split("-")[1]))

    print("\n")
    # Upload alive features
    for file_path in files_to_upload:
        print("===== Uploading file: " + os.path.basename(file_path))
        f = open(file_path, "r")
        data = json.load(f)

        # Replace NaNs
        data_fixed = json.dumps(data, cls=NanConverter)
        data = json.loads(data_fixed)

        url = host + "/api/local/upload-features"
        requests.post(
            url,
            json=data,
        )

    print(
        Align.center(
            Panel(
                f"""
{len(files_to_upload)} batch files uploaded to Neuronpedia.
""",
                title="Uploads Complete",
            )
        )
    )


@app.command()
def upload_dead_stubs(
    outputs_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            prompt="What is the absolute, full local file path to the feature outputs directory?",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            prompt="""Host to upload to? (Default: http://localhost:3000)""",
        ),
    ] = "http://localhost:3000",
):
    """
    This will create "There are no activations for this feature" stubs for dead features on Neuronpedia.  It currently only works if you have admin access to a Neuronpedia instance via localhost:3000.
    """

    skipped_path = os.path.join(outputs_dir, "skipped_indexes.json")
    f = open(skipped_path, "r")
    data = json.load(f)
    url = host + "/api/local/upload-skipped-features"
    requests.post(
        url,
        json=data,
    )

    print(
        Align.center(
            Panel(
                """
Dead feature stubs created.
""",
                title="Complete",
            )
        )
    )


# Helper utilities that help fix weird NaNs in the feature outputs


def nanToNeg999(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: nanToNeg999(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nanToNeg999(v) for v in obj]
    elif (isinstance(obj, float) or isinstance(obj, Decimal)) and math.isnan(obj):
        return -999
    return obj


class NanConverter(json.JSONEncoder):
    def encode(self, o: Any, *args: Any, **kwargs: Any):
        return super().encode(nanToNeg999(o), *args, **kwargs)


if __name__ == "__main__":
    app()
