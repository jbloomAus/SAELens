import io
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Any, Iterable

from huggingface_hub import HfApi, create_repo, get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from tqdm.autonotebook import tqdm

from sae_lens import logger
from sae_lens.constants import (
    RUNNER_CFG_FILENAME,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
)
from sae_lens.saes.sae import SAE


def upload_saes_to_huggingface(
    saes_dict: dict[str, SAE[Any] | Path | str],
    hf_repo_id: str,
    hf_revision: str = "main",
    show_progress: bool = True,
    add_default_readme: bool = True,
):
    api = HfApi()
    if len(saes_dict) == 0:
        raise ValueError("No SAEs to upload")
    # pre-validate that everything is a SAE or a valid path before starting the upload
    for sae_ref in saes_dict.values():
        if isinstance(sae_ref, SAE):
            continue
        _validate_sae_path(Path(sae_ref))

    if not _repo_exists(api, hf_repo_id):
        create_repo(hf_repo_id)

    for sae_id, sae_ref in tqdm(
        saes_dict.items(), desc="Uploading SAEs", disable=not show_progress
    ):
        with TemporaryDirectory() as tmp_dir:
            sae_path = _build_sae_path(sae_ref, tmp_dir)
            _validate_sae_path(sae_path)
            _upload_sae(
                api,
                sae_path,
                repo_id=hf_repo_id,
                sae_id=sae_id,
                revision=hf_revision,
            )
        if add_default_readme:
            if _repo_file_exists(hf_repo_id, "README.md", hf_revision):
                logger.info("README.md already exists in the repo, skipping upload")
            else:
                readme = _create_default_readme(hf_repo_id, saes_dict)
                readme_io = io.BytesIO()
                readme_io.write(readme.encode("utf-8"))
                readme_io.seek(0)
                api.upload_file(
                    path_or_fileobj=readme_io,
                    path_in_repo="README.md",
                    repo_id=hf_repo_id,
                    revision=hf_revision,
                    commit_message="Add README.md",
                )


def _create_default_readme(repo_id: str, sae_ids: Iterable[str]) -> str:
    readme = dedent(
        """
        ---
        library_name: saelens
        ---

        # SAEs for use with the SAELens library

        This repository contains the following SAEs:
        """
    )
    for sae_id in sae_ids:
        readme += f"- {sae_id}\n"

    readme += dedent(
        f"""
        Load these SAEs using SAELens as below:
        ```python
        from sae_lens import SAE

        sae = SAE.from_pretrained("{repo_id}", "<sae_id>")
        ```
        """
    )
    return readme.strip()


def _repo_file_exists(repo_id: str, filename: str, revision: str) -> bool:
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        get_hf_file_metadata(url)
        return True
    except EntryNotFoundError:
        return False


def _repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id)
        return True
    except RepositoryNotFoundError:
        return False


def _upload_sae(api: HfApi, sae_path: Path, repo_id: str, sae_id: str, revision: str):
    api.upload_folder(
        folder_path=sae_path,
        path_in_repo=sae_id,
        repo_id=repo_id,
        revision=revision,
        repo_type="model",
        commit_message=f"Upload SAE {sae_id}",
        allow_patterns=[
            SAE_CFG_FILENAME,
            SAE_WEIGHTS_FILENAME,
            SPARSITY_FILENAME,
            RUNNER_CFG_FILENAME,
        ],
    )


def _build_sae_path(sae_ref: SAE[Any] | Path | str, tmp_dir: str) -> Path:
    if isinstance(sae_ref, SAE):
        sae_ref.save_model(tmp_dir)
        return Path(tmp_dir)
    if isinstance(sae_ref, Path):
        return sae_ref
    return Path(sae_ref)


def _validate_sae_path(sae_path: Path):
    "Validate that the model files exist in the given path."
    if not (sae_path / SAE_CFG_FILENAME).exists():
        raise FileNotFoundError(
            f"SAE config file not found: {sae_path / SAE_CFG_FILENAME}"
        )
    if not (sae_path / SAE_WEIGHTS_FILENAME).exists():
        raise FileNotFoundError(
            f"SAE weights file not found: {sae_path / SAE_WEIGHTS_FILENAME}"
        )
