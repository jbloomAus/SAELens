# we use a script that launches separate python processes to work around OOM issues - this ensures every batch gets the whole system available memory
# better fix is to investigate and fix the memory issues

import json
import math
import os
import subprocess
from pathlib import Path

import requests
import torch
import typer
from rich import print
from rich.align import Align
from rich.panel import Panel
from typing_extensions import Annotated

from sae_lens.analysis.neuronpedia_integration import NanAndInfReplacer
from sae_lens.toolkit.pretrained_saes import load_sparsity
from sae_lens.training.sparse_autoencoder import SparseAutoencoder

OUTPUT_DIR_BASE = Path("../../neuronpedia_outputs")
RUN_SETTINGS_FILE = "run_settings.json"

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
    resume_from_batch: Annotated[
        int,
        typer.Option(
            min=1,
            help="Batch number to resume from.",
            prompt="""
Do you want to resume from a specific batch number?
Enter 1 to start from the beginning. Existing batch files will not be overwritten.""",
        ),
    ] = 1,
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

    # make the outputs subdirectory if it doesn't exist, ensure it's not a file
    outputs_subdir = f"{model_id}_{sae_id}_{sparse_autoencoder.cfg.hook_point}"
    outputs_dir = OUTPUT_DIR_BASE.joinpath(outputs_subdir)
    if outputs_dir.exists() and outputs_dir.is_file():
        print(f"Error: Output directory {outputs_dir.as_posix()} exists and is a file.")
        raise typer.Abort()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Check if output_dir has a run_settings.json file. If so, load those settings.
    run_settings_path = outputs_dir.joinpath(RUN_SETTINGS_FILE)
    print("\n")
    if run_settings_path.exists() and run_settings_path.is_file():
        # load the json file
        with open(run_settings_path, "r") as f:
            run_settings = json.load(f)
            print(
                f"[yellow]Found existing run_settings.json in {run_settings_path.as_posix()}, checking them."
            )
            if run_settings["log_sparsity"] != log_sparsity:
                print(
                    f"[red]Error: log_sparsity in {run_settings_path.as_posix()} doesn't match the current log_sparsity:\n{run_settings['log_sparsity']} vs {log_sparsity}"
                )
                raise typer.Abort()
            if run_settings["sae_id"] != sae_id:
                print(
                    f"[red]Error: sae_id in {run_settings_path.as_posix()} doesn't match the current sae_id:\n{run_settings['sae_id']} vs {sae_id}"
                )
                raise typer.Abort()
            if run_settings["sae_path"] != sae_path_string:
                print(
                    f"[red]Error: sae_path in {run_settings_path.as_posix()} doesn't match the current sae_path:\n{run_settings['sae_path']} vs {sae_path_string}"
                )
                raise typer.Abort()
            if run_settings["n_batches_to_sample"] != n_batches_to_sample:
                print(
                    f"[red]Error: n_batches_to_sample in {run_settings_path.as_posix()} doesn't match the current n_batches_to_sample:\n{run_settings['n_batches_to_sample']} vs {n_batches_to_sample}"
                )
                raise typer.Abort()
            if run_settings["n_prompts_to_select"] != n_prompts_to_select:
                print(
                    f"[red]Error: n_prompts_to_select in {run_settings_path.as_posix()} doesn't match the current n_prompts_to_select:\n{run_settings['n_prompts_to_select']} vs {n_prompts_to_select}"
                )
                raise typer.Abort()
            if run_settings["feat_per_batch"] != feat_per_batch:
                print(
                    f"[red]Error: feat_per_batch in {run_settings_path.as_posix()} doesn't match the current feat_per_batch:\n{run_settings['feat_per_batch']} vs {feat_per_batch}"
                )
                raise typer.Abort()
            print("[green]All settings match, using existing run_settings.json.")
    else:
        print(f"[green]Creating run_settings.json in {run_settings_path.as_posix()}.")
        run_settings = {
            "sae_id": sae_id,
            "sae_path": sae_path_string,
            "log_sparsity": log_sparsity,
            "n_batches_to_sample": n_batches_to_sample,
            "n_prompts_to_select": n_prompts_to_select,
            "feat_per_batch": feat_per_batch,
        }
        with open(run_settings_path, "w") as f:
            json.dump(run_settings, f, indent=4)

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
            prompt="What is the absolute local file path to the feature outputs directory?",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(
            prompt="""Host to upload to? (Default: http://localhost:3000)""",
        ),
    ] = "http://localhost:3000",
    resume: Annotated[
        int,
        typer.Option(
            prompt="""Resume from batch? (Default: 1)""",
        ),
    ] = 1,
):
    """
    This will upload features that were generated to Neuronpedia. It currently only works if you have admin access to a Neuronpedia instance via localhost:3000.
    """

    files_to_upload = list(outputs_dir.glob("batch-*.json"))

    # filter files where batch-[number].json the number is >= resume
    files_to_upload = [
        file_path
        for file_path in files_to_upload
        if int(file_path.stem.split("-")[1]) >= resume
    ]

    # sort files by batch number
    files_to_upload.sort(key=lambda x: int(x.stem.split("-")[1]))

    print("\n")
    # Upload alive features
    for file_path in files_to_upload:
        print("===== Uploading file: " + os.path.basename(file_path))
        f = open(file_path, "r")
        data = json.load(f, parse_constant=NanAndInfReplacer)

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
            prompt="What is the absolute local file path to the feature outputs directory?",
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


if __name__ == "__main__":
    app()
