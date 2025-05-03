from collections import defaultdict

from huggingface_hub import HfApi

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory


def is_model_file(filepath: str) -> bool:
    """Check if a filepath has a model file extension."""
    model_extensions = [".npz", ".safetensors", ".pt", ".json"]
    return any(filepath.endswith(ext) for ext in model_extensions)


def is_file_path(path: str) -> bool:
    """Determine if a path likely points to a file based on model file extensions or structure."""
    # Check for model file extensions
    if is_model_file(path):
        return True

    # If the path ends at a component that looks like a trainer ID (trainer_X)
    # it's likely a file path even without an extension
    if path.split("/")[-1].startswith("trainer_"):
        return True

    # Check for version tags or other special indicators
    last_component = path.split("/")[-1]
    return ":" in last_component or "." in last_component


def check_path_existence(sae_path: str, huggingface_files: list[str]) -> bool:
    """
    Check if a SAELens path exists in HuggingFace files.

    - For exact matches
    - For paths with different repo prefixes
    - For trainer directories
    """
    # Case 1: Direct match
    if sae_path in huggingface_files:
        return True

    # Case 2: Check for paths that might be subpaths or have added prefixes
    # For example, the HF path might include a main/ or other prefix
    for file in huggingface_files:
        # If the file path ends with our sae_path
        if file.endswith(sae_path):
            return True

        # If our sae_path ends with the file path
        if sae_path.endswith(file):
            return True

        # If our path is in the middle of a longer path
        if "/" + sae_path + "/" in "/" + file + "/":
            return True

    # Case 3: Special handling for trainer paths
    # If the path ends with trainer_X, check if it's part of a file path
    if sae_path.split("/")[-1].startswith("trainer_"):
        trainer_id = sae_path.split("/")[-1]
        parent_dir = "/".join(sae_path.split("/")[:-1])

        # Check if any file is in this trainer directory
        for file in huggingface_files:
            if file.startswith(sae_path + "/") or file.startswith(
                parent_dir + "/" + trainer_id + "/"
            ):
                return True

    # Case 4: Check directories
    if not is_file_path(sae_path):
        directory_prefix = sae_path + "/"
        for file in huggingface_files:
            if file.startswith(directory_prefix):
                return True

    return False


api = HfApi()
saes_directory = get_pretrained_saes_directory()

# Group lookups by repo_id
repo_id_to_lookups = defaultdict(list)
for _, lookup in saes_directory.items():
    repo_id_to_lookups[lookup.repo_id].append(lookup)

any_missing_in_hf = False
output = []

# Process each unique repo_id
for repo_id, lookups in repo_id_to_lookups.items():
    # Get all files in the HuggingFace repo
    huggingface_files = api.list_repo_files(repo_id)

    # Filter for model files
    model_files = [file for file in huggingface_files if is_model_file(file)]

    # Create mapping from directory to files
    huggingface_dir_to_files = defaultdict(list)
    for file in model_files:
        directory = "/".join(file.split("/")[:-1])
        huggingface_dir_to_files[directory].append(file)

    # Combine sae_maps from all lookups for this repo_id
    combined_sae_lens_files = {}
    for lookup in lookups:
        combined_sae_lens_files.update(lookup.saes_map)

    # Check if each sae_lens path exists in HuggingFace
    missing_in_huggingface = []
    for hook_id, sae_path in combined_sae_lens_files.items():
        if not check_path_existence(sae_path, huggingface_files):
            missing_in_huggingface.append((hook_id, sae_path))

    # Find model directories/files in HuggingFace not referenced in sae_lens
    sae_lens_paths = set(combined_sae_lens_files.values())
    missing_in_sae_lens = []

    # Check directories with model files
    for directory, files in huggingface_dir_to_files.items():
        # Skip empty directories or direct file paths
        if not directory or is_file_path(directory):
            continue

        # Check if directory or parent/child relationship exists in sae_lens
        found = False
        for sae_path in sae_lens_paths:
            # Same directory
            if directory == sae_path:
                found = True
                break

            # Check parent-child relationships
            if sae_path.startswith(directory + "/") or directory.startswith(
                sae_path + "/"
            ):
                found = True
                break

        # Include only if it contains model files and not already accounted for
        if not found and any(is_model_file(file) for file in files):
            missing_in_sae_lens.append(directory)

    # Check model files that aren't in any directory we've already checked
    for model_file in model_files:
        directory = "/".join(model_file.split("/")[:-1])

        # Skip files whose directories we've already checked
        if directory in huggingface_dir_to_files:
            continue

        # Check if this specific file is referenced in sae_lens
        found = False
        for sae_path in sae_lens_paths:
            if model_file == sae_path or model_file.startswith(sae_path + "/"):
                found = True
                break

        if not found:
            missing_in_sae_lens.append(model_file)

    if missing_in_huggingface:
        any_missing_in_hf = True
        output.append("\nPaths in sae_lens that don't exist in HuggingFace repo:")
        for hook_id, sae_path in sorted(missing_in_huggingface, key=lambda x: x[1]):
            output.append(f"{repo_id}  {sae_path}")

    if missing_in_sae_lens:
        output.append("\nModel paths in HuggingFace repo that aren't in sae_lens:")
        for path in sorted(missing_in_sae_lens):
            path_type = "file" if is_file_path(path) else "directory"
            output.append(f"{repo_id}  {path}")

if any_missing_in_hf:
    print("\n".join(output))
