import torch
from transformer_lens import HookedTransformer

from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.toolkit.sae_attrib import SAEPatchHook

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_sae_patch_hook_does_not_change_output():

    model = HookedTransformer.from_pretrained("gpt2").to(device)
    sae = get_gpt2_res_jb_saes()[0]["blocks.8.hook_resid_pre"].to(device)
    hook_point = sae.cfg.hook_point
    prompt = "Hello world"
    orig_loss = model(prompt, return_type="loss")

    with model.hooks(fwd_hooks=[(hook_point, SAEPatchHook(sae))]):
        patched_loss = model(prompt, return_type="loss")

    assert torch.isclose(orig_loss, patched_loss, atol=1e-6)


def test_sae_patch_hook_fields_have_grad():
    model = HookedTransformer.from_pretrained("gpt2").to(device)
    sae = get_gpt2_res_jb_saes()[0]["blocks.8.hook_resid_pre"].to(device)
    hook_point = sae.cfg.hook_point
    prompt = "Hello world"

    sae_patch_hook = SAEPatchHook(sae)
    assert sae_patch_hook.sae_feature_acts.grad is None
    assert sae_patch_hook.sae_errors.grad is None

    with model.hooks(fwd_hooks=[(hook_point, sae_patch_hook)]):
        patched_loss = model(prompt, return_type="loss")
        patched_loss.backward()

    assert sae_patch_hook.sae_feature_acts.grad is not None
    assert sae_patch_hook.sae_errors.grad is not None
