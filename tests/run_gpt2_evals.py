import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sae_lens.evals import run_evals
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in saes_directory.items():
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append(
                (release, sae_name, expected_var_explained, expected_l0)
            )

    return all_loadable_saes


def eval_all_loadable_gpt2_saes(
    num_eval_batches: int = 10,
    eval_batch_size_prompts: int = 8,
    datasets: list[str] = ["Skylion007/openwebtext", "lighteval/MATH"],
    ctx_lens: list[int] = [64, 128, 256, 512],
) -> pd.DataFrame:
    all_saes = all_loadable_saes()
    gpt2_saes = [sae for sae in all_saes if "gpt2-small" in sae[0]]

    device = "cuda:0"

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    data = []

    for sae_name, sae_block, _, _ in tqdm(gpt2_saes):

        sae = SAE.from_pretrained(
            release=sae_name,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=sae_block,  # won't always be a hook point
            device=device,
        )[0]

        for ctx_len in ctx_lens:
            for dataset in datasets:
                activation_store = ActivationsStore.from_sae(
                    model, sae, context_size=ctx_len, dataset=dataset
                )
                activation_store.shuffle_input_dataset(seed=42)

                eval_metrics = {}
                eval_metrics["sae_id"] = f"{sae_name}-{sae_block}"
                eval_metrics["context_size"] = ctx_len
                eval_metrics["dataset"] = dataset

                eval_metrics |= run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=model,
                    n_eval_batches=10,
                    eval_batch_size_prompts=8,
                )

                data.append(eval_metrics)

    return pd.DataFrame(data)


# %%

if __name__ == "__main__":
    df = eval_all_loadable_gpt2_saes()
    df.to_csv("gpt2_saes_evals.csv", index=False)
