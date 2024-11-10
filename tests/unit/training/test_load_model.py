import pytest
import torch
from mamba_lens import HookedMamba
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens.load_model import HookedProxyLM, _extract_logits_from_output, load_model


@pytest.fixture
def gpt2_proxy_model():
    return load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )


def test_load_model_works_with_mamba():
    model = load_model(
        model_class_name="HookedMamba",
        model_name="state-spaces/mamba-370m",
        device="cpu",
    )
    assert isinstance(model, HookedMamba)


def test_load_model_works_without_model_kwargs():
    model = load_model(
        model_class_name="HookedTransformer",
        model_name="pythia-14m",
        device="cpu",
    )
    assert isinstance(model, HookedTransformer)
    assert model.cfg.checkpoint_index is None


def test_load_model_works_with_model_kwargs():
    model = load_model(
        model_class_name="HookedTransformer",
        model_name="pythia-14m",
        device="cpu",
        model_from_pretrained_kwargs={"checkpoint_index": 0},
    )
    assert isinstance(model, HookedTransformer)
    assert model.cfg.checkpoint_index == 0


def test_load_model_with_generic_huggingface_lm():
    model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    assert isinstance(model, HookedProxyLM)


def test_HookedProxyLM_gives_same_cached_states_as_original_implementation():
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hooked_model = HookedProxyLM(hf_model, tokenizer)
    input_ids = tokenizer.encode("hi", return_tensors="pt")
    proxy_logits, cache = hooked_model.run_with_cache(input_ids)

    hf_output = hf_model(input_ids, output_hidden_states=True)

    assert torch.allclose(proxy_logits, hf_output.logits)
    for i in range(len(hf_output.hidden_states) - 2):
        assert torch.allclose(
            cache[f"transformer.h.{i}"], hf_output.hidden_states[i + 1]
        )


def test_HookedProxyLM_gives_same_cached_states_as_tlens_implementation(
    gpt2_proxy_model: HookedProxyLM,
):
    tlens_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")

    input_ids = tlens_model.to_tokens("hi")
    hf_cache = gpt2_proxy_model.run_with_cache(input_ids)[1]
    tlens_cache = tlens_model.run_with_cache(input_ids)[1]
    for i in range(12):
        assert torch.allclose(
            hf_cache[f"transformer.h.{i}"],
            tlens_cache[f"blocks.{i}.hook_resid_post"],
            atol=1e-3,
        )


def test_HookedProxyLM_forward_gives_same_output_as_tlens(
    gpt2_proxy_model: HookedProxyLM,
):
    tlens_model = HookedTransformer.from_pretrained("gpt2", device="cpu")

    batch_tokens = tlens_model.to_tokens("hi there")
    tlens_output = tlens_model(batch_tokens, return_type="both", loss_per_token=True)
    hf_output = gpt2_proxy_model(batch_tokens, return_type="both", loss_per_token=True)

    # Seems like tlens removes the means before softmaxing
    hf_logits_normed = hf_output[0] - hf_output[0].mean(dim=-1, keepdim=True)

    assert torch.allclose(tlens_output[0], hf_logits_normed, atol=1e-3)
    assert torch.allclose(tlens_output[1], hf_output[1], atol=1e-3)


def test_extract_logits_from_output_works_with_multiple_return_types():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode("hi there", return_tensors="pt")
    out_dict = model(tokens, return_dict=True)
    out_tuple = model(tokens, return_dict=False)

    logits_dict = _extract_logits_from_output(out_dict)
    logits_tuple = _extract_logits_from_output(out_tuple)

    assert torch.allclose(logits_dict, logits_tuple)


def test_HookedProxyLM_to_tokens_gives_same_output_as_tlens(
    gpt2_proxy_model: HookedProxyLM,
):
    tlens_model = HookedTransformer.from_pretrained("gpt2", device="cpu")

    tl_tokens = tlens_model.to_tokens(
        "hi there", prepend_bos=False, truncate=False, move_to_device=False
    )
    hf_tokens = gpt2_proxy_model.to_tokens(
        "hi there", prepend_bos=False, truncate=False, move_to_device=False
    )

    assert torch.allclose(tl_tokens, hf_tokens)
