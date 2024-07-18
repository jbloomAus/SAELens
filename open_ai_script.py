# type: ignore


import os
import torch
from azure.storage.blob import BlobClient
from azure.core.exceptions import ResourceNotFoundError
from sae_lens import ActivationsStore, run_evals, SAE, SAEConfig
torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer
import pandas as pd 
import json
from sae_lens.evals import get_eval_everything_config
import gc 
from tqdm import tqdm
from sae_lens import SAE, ActivationsStore
from tqdm import tqdm 

model = HookedTransformer.from_pretrained_no_processing("gpt2-small")

def v5_32k(location, layer_index):
    assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]#, "mlp_post_act"]
    assert layer_index in range(12)
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_32k/autoencoders/{layer_index}.pt"

def v5_128k(location, layer_index):
    assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]#, "mlp_post_act"]
    assert layer_index in range(12)
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_128k/autoencoders/{layer_index}.pt"

def download_file(url, filename):
    # Extract container and blob path from the URL
    _, _, account, container, *blob_parts = url.split('/')
    blob_path = '/'.join(blob_parts)

    # Construct the full URL for the BlobClient
    full_url = f"https://{account}.blob.core.windows.net/{container}/{blob_path}"

    # Create a BlobClient
    blob_client = BlobClient.from_blob_url(full_url)

    try:
        # Download the blob
        with open(filename, "wb") as file:
            data = blob_client.download_blob()
            file.write(data.readall())
        print(f"Downloaded: {filename}")
    except ResourceNotFoundError:
        print(f"Failed to download: {filename}. The blob was not found.")
    except Exception as e:
        print(f"An error occurred while downloading {filename}: {str(e)}")


@torch.no_grad()
def get_sparsity_metrics(
    sparse_autoencoder: SAE,
    activation_store: ActivationsStore,
    n_batches: int = 50,
) -> tuple[float, float, torch.Tensor, float]:

    batch_size = 4096

    assert isinstance(sparse_autoencoder.cfg.d_sae, int)
    total_feature_acts = torch.zeros(sparse_autoencoder.cfg.d_sae)
    l0s_list = []
    l1s_list = []
    for _ in tqdm(range(n_batches), total=n_batches, desc="Feature Sparsity"):

        batch_activations = activation_store.next_batch()
        feature_acts = sparse_autoencoder.encode(batch_activations).squeeze()
        l0s = (feature_acts > 0).float().squeeze().sum(dim=1)
        l1s = feature_acts.abs().sum(dim=1)
        total_feature_acts += (feature_acts > 0).squeeze().sum(dim=0).cpu()
        l0s_list.append(l0s)
        l1s_list.append(l1s)

    l0 = torch.concat(l0s_list).mean().item()
    l1 = torch.concat(l1s_list).mean().item()

    sparsity = total_feature_acts / (n_batches * batch_size)
    log_feature_sparsity = torch.log10(sparsity + 1e-10)
    percent_alive = (log_feature_sparsity > -5).float().mean().item()

    return l0, l1, log_feature_sparsity, percent_alive

if __name__ == "__main__":

    # SETTINGS
    locations = ["resid_delta_attn","resid_delta_mlp"]
    layers = range(12)
    version_funcs = [v5_32k, v5_128k]
    
    
    # begin script
    all_metrics = {}

    for location in locations:
        sae_directory = f"open_ai_sae_weights_{location}"
        sae_directory_new = f"open_ai_sae_weights_{location}_reformatted"

        os.makedirs(sae_directory, exist_ok=True)
        os.makedirs(sae_directory_new, exist_ok=True)
        for version_func in version_funcs:
            for layer_index in tqdm(layers):

                filename = os.path.join(sae_directory, f"{version_func.__name__}_layer_{layer_index}.pt")
                if not os.path.exists(filename):
                    url = version_func(location, layer_index)
                    print("Downloading", url)
                    download_file(url, filename)
                    print("Downloaded", url)
                    
                    
                transformer_lens_loc = {
                    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
                    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
                    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
                    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
                    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
                }
            
                reformatted_file_name = f"{version_func.__name__}_layer_{layer_index}"
                
                if not os.path.exists(os.path.join(sae_directory_new, reformatted_file_name)):
                    weights = torch.load(filename)
                    sae_config = SAEConfig.from_dict(
                        config_dict={"architecture": "standard",
                                    "d_in": 768,
                                    "d_sae": weights["latent_bias"].shape[0],
                                    "dtype": "torch.float32",
                                    "device": "cuda",
                                    "model_name": "gpt2-small",
                                    "hook_name": transformer_lens_loc[location],
                                    "hook_layer": layer_index,
                                    "hook_head_index": None,
                                    "activation_fn_str": "topk",  # use string for serialization
                                    "activation_fn_kwargs": {"k":32}, # default to Relu
                                    "apply_b_dec_to_input": True,
                                    "finetuning_scaling_factor": False,
                                    "sae_lens_training_version": None,
                                    "prepend_bos": False,
                                    "dataset_path": "Skylion007/openwebtext",
                                    "dataset_trust_remote_code": True,
                                    "context_size": 64,
                                    "normalize_activations": "layer_norm"
                                }
                    )

                    sae = SAE(sae_config)

                    rename_dict = {
                        "encoder.weight": "W_enc",
                        "latent_bias": "b_enc",
                        "decoder.weight": "W_dec",
                        "pre_bias": "b_dec",
                    }

                    renamed_weights = {}
                    for k, v in weights.items():
                        if k in rename_dict:
                            if k == "encoder.weight":
                                v = v.T
                            elif k == "decoder.weight":
                                v = v.T
                            renamed_weights[rename_dict[k]] = v
                    sae.load_state_dict(renamed_weights)
                    
                    del renamed_weights
                    del weights
                    gc.collect()
                    
                    activations_store = ActivationsStore.from_sae(
                        model=model, 
                        sae=sae,
                        context_size=256,
                        device="cuda",
                        streaming=True,
                        store_batch_size_prompts=64,
                    )
                    
                    # Get L0 and L1
                    n_batches = 300
                    l0, l1, sparsity, percent_alive = get_sparsity_metrics(
                        sparse_autoencoder=sae,
                        activation_store=activations_store,
                        n_batches=n_batches,
                    )
                
                    print(f"{layer_index}/{location} L0: {l0}, L1: {l1}", f"Percent Alive: {percent_alive}")


                    sae.save_model(os.path.join(sae_directory_new, reformatted_file_name), sparsity=sparsity)

                else:
                    print(f"File {reformatted_file_name} already exists")
                    sae = SAE.load_from_pretrained(
                        os.path.join(sae_directory_new, reformatted_file_name),
                        device="cuda",
                    )
                
                
                activations_store = ActivationsStore.from_sae(
                    model=model, 
                    sae=sae,
                    context_size=64,
                    device="cuda",
                    streaming=True,
                )
                    

                cfg = get_eval_everything_config(
                    batch_size_prompts=32,
                    n_eval_reconstruction_batches=3,
                    n_eval_sparsity_variance_batches=3,
                )

                eval_metrics = run_evals(sae, activations_store, model, eval_config=cfg)

                file_name = f"metrics.json"
                with open(os.path.join(sae_directory_new, reformatted_file_name, file_name), "w") as f:
                    json.dump(eval_metrics, f)
                
                del sae 
                gc.collect()
        

    # pd.DataFrame(all_metrics).T.style.format("{:.3f}").background_gradient(cmap="viridis", axis=0) # check metrics are good.