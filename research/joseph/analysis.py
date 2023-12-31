import os
import warnings
from functools import partial
from typing import List, Optional, Tuple

import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, utils
from transformer_lens.components import HookPoint

from research.joseph.utils import make_token_df
from sae_training.sparse_autoencoder import SparseAutoencoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


warnings.filterwarnings("ignore")



# HOOKS
# LAYER_IDX, HEAD_IDX = (10, 7)
LAYER_IDX, HEAD_IDX = (1, 11)
# W_U = model.W_U.clone()
# HEAD_HOOK_RESULT_NAME = utils.get_act_name("result", LAYER_IDX)
HEAD_HOOK_RESULT_NAME = utils.get_act_name("z", LAYER_IDX)
HEAD_HOOK_QUERY_NAME = utils.get_act_name("q", LAYER_IDX)
HEAD_HOOK_RESID_NAME = utils.get_act_name("resid_pre", LAYER_IDX)
# BATCH_SIZE = 10




# Get ATTN results

def get_max_attn_token(tokens, cache, model: HookedTransformer, LAYER_IDX, HEAD_IDX):
    tokens = tokens.to("cpu")
    pattern_name = utils.get_act_name("pattern", LAYER_IDX)
    pattern = cache[pattern_name][0,HEAD_IDX].detach().cpu()
    max_idx_pos = pattern.argmax(dim=-1)
    max_idx_token_id = torch.gather(tokens, dim=-1, index=max_idx_pos.unsqueeze(-1).T)
    max_idx_tok = model.to_string(max_idx_token_id.T)
    max_idx_tok_value = pattern.max(dim=1).values
    return max_idx_pos[1:], max_idx_tok[1:], max_idx_tok_value[1:]


def kl_divergence_attention(y_true, y_pred):

    # Compute log probabilities for KL divergence
    log_y_true = torch.log(y_true + 1e-10)
    log_y_pred = torch.log(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)


def eval_prompt(
    prompt: List, 
    model: HookedTransformer, 
    sparse_autoencoder: Optional[SparseAutoencoder] = None,
    head_idx_override: Optional[int] = None,):
    '''
    Takes a list of strings as input.
    '''
    tokens = model.to_tokens(prompt)
    # tokens = tokens[:, :MAX_PROMPT_LEN]
    token_df = make_token_df(model, tokens[:,1:], len_suffix=5, len_prefix=10)
    
    # tokens = t.stack(tokens).to(device)
    layer_idx = sparse_autoencoder.cfg.hook_point_layer
    head_idx = sparse_autoencoder.cfg.hook_point_head_index if head_idx_override is None else head_idx_override
    head_hook_query_name = utils.get_act_name("q", layer_idx)
    head_hook_result_name = utils.get_act_name("z", layer_idx)
    head_hook_resid_name = utils.get_act_name("resid_pre", layer_idx)
    
    # Basic Forward Pass
    (original_logits, original_loss), original_cache = model.run_with_cache(tokens, return_type="both", loss_per_token=True)
    token_df['loss'] = original_loss.flatten().tolist()
    
    ## Collect ATTN Results
    max_idx_pos, max_idx_tok, max_idx_tok_value = get_max_attn_token(tokens, original_cache, model, layer_idx, head_idx)
    token_df['max_idx_pos'] = max_idx_pos.flatten().tolist()
    token_df['max_idx_tok'] = max_idx_tok
    token_df['max_idx_tok_value'] = max_idx_tok_value.flatten().tolist()
    
    
    # Full Head Ablation
    def hook_to_ablate_head(head_output: Float[Tensor, "batch seq_len head_idx d_head"], hook: HookPoint, head = (LAYER_IDX, HEAD_IDX)):
        assert head[0] == hook.layer(), f"{head[0]} != {hook.layer()}"
        assert ("result" in hook.name) or ("q" in hook.name) or ("z" in hook.name)
        head_output[:, :, head[1], :] = 0
        return head_output

    hook_to_ablate_head = partial(hook_to_ablate_head, head=(layer_idx, head_idx))
    ablated_logits, ablated_loss = model.run_with_hooks(tokens, return_type="both", loss_per_token=True, fwd_hooks=[(head_hook_result_name, hook_to_ablate_head)])
    
    logit_diff = original_logits - ablated_logits
    top10_token_suppression_vals, top10_token_suppression_inds = torch.topk(logit_diff, 10, dim=-1, largest=False)
    token_df['top10_token_suppression_diffs'] = top10_token_suppression_vals.flatten(0,1).tolist()[1:]
    decoded_tokens = [[model.tokenizer.decode(tok_id, skip_special_tokens=True) for tok_id in sequence] for sequence in top10_token_suppression_inds[0]]
    token_df['top10_token_suppression_inds'] = decoded_tokens[1:]
    
    top10_token_boosting_vals, top10_token_boosting_inds = torch.topk(logit_diff, 10, dim=-1, largest=True)
    token_df['top10_token_boosting_vals'] = top10_token_boosting_vals.flatten(0,1).tolist()[1:]
    decoded_tokens = [[model.tokenizer.decode(tok_id, skip_special_tokens=True) for tok_id in sequence] for sequence in top10_token_boosting_inds[0]]
    token_df['top10_token_boosting_inds'] = decoded_tokens[1:]
    
    
    token_df['ablated_loss'] = ablated_loss.flatten().tolist()
    token_df["loss_diff"] = token_df["ablated_loss"] - token_df["loss"]
    
    if sparse_autoencoder is not None:
        # Reconstruction of Query with SAE
        if "resid_pre" in sparse_autoencoder.cfg.hook_point:
            original_act = original_cache[sparse_autoencoder.cfg.hook_point]
            # token_df["q_norm"] = torch.norm(original_act, dim=-1)[:,1:].flatten().tolist()
            sae_out, feature_acts, _, mse_loss, _ = sparse_autoencoder(original_act)
            # token_df["rec_q_norm"] = torch.norm(sae_out, dim=-1)[:,1:].flatten().tolist()

            # need to generate query
            def replacement_hook(resid_pre, hook, new_resid_pre=sae_out):
                return new_resid_pre
            
            with model.hooks(fwd_hooks=[(head_hook_resid_name, replacement_hook)]):
                _, resid_pre_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)
                sae_out = resid_pre_cache[head_hook_query_name][:,:,head_idx]
            
            original_act = original_cache[head_hook_query_name][:,:,head_idx]
            per_tok_mse_loss = (sae_out.float() - original_act.float()).pow(2).sum(-1)
            total_variance = original_act.pow(2).sum(-1)
            explained_variance = per_tok_mse_loss/total_variance
            
        else:
            original_act = original_cache[sparse_autoencoder.cfg.hook_point][:,:,head_idx]
            token_df["q_norm"] = torch.norm(original_act, dim=-1)[:,1:].flatten().tolist()
            sae_out, feature_acts, _, mse_loss, _ = sparse_autoencoder(original_cache[sparse_autoencoder.cfg.hook_point][:,:,head_idx])
            token_df["rec_q_norm"] = torch.norm(sae_out, dim=-1)[:,1:].flatten().tolist()
            # norm_ratio = torch.norm(original_act, dim=-1)/ torch.norm(sae_out, dim=-1)
            
            per_tok_mse_loss = (sae_out.float() - original_act.float()).pow(2).sum(-1)
            total_variance = original_act.pow(2).sum(-1)
            explained_variance = per_tok_mse_loss/total_variance
            
        num_active_features = (feature_acts > 0).sum(dim=-1)
        top_feature_acts, top_features = torch.topk(feature_acts, k = 10, dim = -1)
        
                # SAE Metrics
        token_df['mse_loss'] = per_tok_mse_loss.flatten()[1:].tolist()
        token_df['explained_variance'] = explained_variance.flatten()[1:].tolist()
        token_df['num_active_features'] = num_active_features.flatten()[1:].tolist()
        token_df['top_k_feature_acts'] = top_feature_acts.flatten(0,1).tolist()[1:]
        token_df['top_k_features'] = top_features.flatten(0,1).tolist()[1:]
        
        # Reconstruct Query            
        def hook_to_reconstruct_query(
            head_input: Float[Tensor, "batch seq_len head_idx d_head"], 
            hook: HookPoint, 
            head,
            reconstructed_query: Float[Tensor, "batch seq_len d_model"] = None,):
            assert head[0] == hook.layer()
            head_input[:, :, head[1], :] = reconstructed_query[:, :]
            return head_input

                
        hook_fn = partial(hook_to_reconstruct_query, reconstructed_query=sae_out, head = (layer_idx, head_idx))
        with model.hooks(fwd_hooks=[(head_hook_query_name, hook_fn)]):
            _, cache_reconstructed_query = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)
            max_idx_pos, max_idx_tok, max_idx_tok_value = get_max_attn_token(tokens, cache_reconstructed_query, model, layer_idx, head_idx)
            
            # Get the KL Divergence of the attention distributions
            patterns_original = original_cache[utils.get_act_name("pattern", layer_idx)][0,head_idx].detach().cpu()
            patterns_reconstructed = cache_reconstructed_query[utils.get_act_name("pattern", layer_idx)][0,head_idx].detach().cpu()
            kl_result = kl_divergence_attention(patterns_original, patterns_reconstructed)
            kl_result = kl_result.sum(dim=-1)[1:].numpy()
        
        token_df['rec_q_max_idx_pos'] = max_idx_pos.flatten().tolist()
        token_df['rec_q_max_idx_tok'] = max_idx_tok
        token_df['rec_q_max_idx_tok_value'] = max_idx_tok_value.flatten().tolist()
        token_df['kl_divergence'] = kl_result.flatten().tolist()

        # add results to dataframe


        # print(feature_acts.shape)
        # token_df["ids_active_features"] = (feature_acts[0,1:] > 0)
    else:
        cache_reconstructed_query = None
    
    return token_df, original_cache, cache_reconstructed_query, feature_acts.flatten(0,1)[1:]
