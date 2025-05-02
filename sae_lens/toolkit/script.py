# # %%
# import torch

# from sae_lens import SAE
# from sae_lens.toolkit.pretrained_sae_loaders import (
#     dictionary_learning_huggingface_loader,
# )

# # sae, cfg_dict, sparsity = SAE.from_pretrained(
# #     release="jbloom/GPT2-Small-SAEs-Reformatted",  # see other options in sae_lens/pretrained_saes.yaml
# #     sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
# #     device=device,
# #     converter=sae_lens_huggingface_loader
# # )

# if torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cuda" if torch.cuda.is_available() else "cpu"


# # sae, cfg_dict, sparsity = SAE.from_pretrained(
# #     # release="EleutherAI/sae-llama-3-8b-32x",  # see other options in sae_lens/pretrained_saes.yaml
# #     release="EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k",  # see other options in sae_lens/pretrained_saes.yaml
# #     # sae_id="layers.10",  # won't always be a hook point
# #     sae_id="layers.10.mlp",  # won't always be a hook point
# #     device=device,
# #     converter=sparsify_huggingface_loader,
# # )

# targets = [
#     # ("EleutherAI/sae-llama-3-8b-32x", "layers.10"),
#     # ("EleutherAI/sae-llama-3-8b-32x-v2", "layers.1"),
#     # ("EleutherAI/sae-llama-3.1-8b-32x", "layers.23.mlp"),
#     # ("EleutherAI/sae-llama-3.1-8b-64x", "layers.23"),
#     # ("EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k", "layers.10.mlp"),
#     (
#         "canrager/saebench_gemma-2-2b_width-2pow14_date-0107",
#         "gemma-2-2b_matryoshka_batch_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_0",
#     ),
#     (
#         "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
#         "BatchTopK_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0",
#     ),
#     (
#         "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
#         "TopK_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0",
#     ),
#     (
#         "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
#         "JumpRelu_pythia-160m-deduped__0108/resid_post_layer_8/trainer_5",
#     ),
#     (
#         "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
#         "GatedSAE_pythia-160m-deduped__0108/resid_post_layer_8/trainer_5",
#     ),
# ]
# for repo, layer in targets:
#     sae, cfg, _ = SAE.from_pretrained(
#         # repo, layer, device="cpu", converter=sparsify_huggingface_loader
#         repo,
#         layer,
#         device="cpu",
#         converter=dictionary_learning_huggingface_loader,
#     )
#     acts = sae.encode(torch.randn(1, cfg["d_in"]))
#     # assert (acts != 0).sum().item() == cfg["activation_fn_kwargs"]["k"]
#     # print(repo, "✓", cfg["dataset_path"])
#     print("success")
#     print("\n=== quick inspection ===")
#     print("hook : ", cfg["hook_name"])
#     print("sizes: d_in =", cfg["d_in"], "| d_sae =", cfg["d_sae"])
#     print("dtype:", cfg["dtype"])
#     print("actfn:", cfg["activation_fn_str"], cfg.get("activation_fn_kwargs", {}))
#     print()
#     print("enc weight shape :", sae.W_enc.shape)
#     print("dec weight shape :", sae.W_dec.shape)
#     if hasattr(sae, "b_enc"):
#         print("enc bias shape :", sae.b_enc.shape)
#     if hasattr(sae, "b_dec"):
#         print("dec bias shape :", sae.b_dec.shape)

#     # --- tiny functional test --------------------------------------------
#     batch = torch.randn(4, cfg["d_in"])  # 4 random activations
#     with torch.inference_mode():
#         z = sae.encode(batch)  # sparse code
#         recon = sae.decode(z)  # reconstruction

#     print("\n=== functional check ===")
#     print("sparse code mean‑L0 :", (z != 0).float().sum(-1).mean().item())

#     # check that encode kept sparsity target if Top‑K style
#     k = cfg.get("activation_fn_kwargs", {}).get("k")
#     if k is not None:
#         assert (z != 0).sum(-1).le(k).all()
#         print(f"k‑constraint OK (≤ {k} non‑zeros per example)")

#     # rough reconstruction error (should be < 1 for a sensible SAE)
#     rmse = ((batch - recon) ** 2).mean().sqrt().item()
#     print("RMSE :", rmse)
# # %%
# # from huggingface_hub import hf_hub_download
# # from safetensors.torch import load_file

# # keys = load_file(
# #     hf_hub_download("EleutherAI/sae-llama-3-8b-32x", "layers.10/sae.safetensors"),
# #     device="cpu",
# # ).keys()
# # print(keys)  # %%
# # %%
