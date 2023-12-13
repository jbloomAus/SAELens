from .cautils import *

# from transformer_lens.rs.callum2.utils import get_effective_embedding
# from transformer_lens.rs.callum.keys_fixed import project


def print_best_outputs(
    best_k_indices: List[Tuple[int, int]], # list of (batch_idx, seq_pos) tuples
    best_k_loss_decrease: List[Float], # sorted loss increases from ablating 10.7
    hook: Tuple[str, Callable], # (hook_name, hook_fn) for doing ablation
    model: HookedTransformer,
    data: List[str],
    n: int = 10,
    random: bool = False,
    seed: int = 42,
    window: int = 200,
    return_caches: bool = False,
    names_filter: Callable = lambda x: True
):
    assert len(best_k_indices) == len(best_k_loss_decrease)
    # assert (sorted(best_k_loss_decrease, reverse=True) == best_k_loss_decrease) or (sorted(best_k_loss_decrease, reverse=False) == best_k_loss_decrease)
    caches = []

    if random:
        t.manual_seed(seed)
        indices_to_print = t.randperm(len(best_k_indices))[:n].tolist()
    else:
        indices_to_print = list(range(n))

    for i in indices_to_print:

        # Get the indices of the token where loss was calculated from
        sample_idx, token_idx = best_k_indices[i]
        loss_decrease = best_k_loss_decrease[i]

        # Get the string leading up to that point, and a few other useful things e.g. the correct next token
        new_str = data[sample_idx]
        new_str_tokens = model.to_str_tokens(new_str)
        tokens = model.to_tokens(new_str)[:, :token_idx+1]
        prev_tokens = "".join(new_str_tokens[max(0, token_idx - window): token_idx + 1])
        next_tokens = "".join(new_str_tokens[token_idx + 2: min(token_idx + 10, len(new_str_tokens))])
        correct_str_token = new_str_tokens[token_idx + 1]
        correct_token = model.to_tokens(new_str)[0, token_idx+1]

        # Get logits and stuff for the original (non-ablated) distribution
        if return_caches:
            logits, cache = model.run_with_cache(tokens, return_type="logits", names_filter=names_filter)
            logits = logits[0, -1]
            caches.append((cache, new_str_tokens[max(0, token_idx - window): token_idx + 1]))
        else:
            logits = model(tokens, return_type="logits")[0, -1]
        log_probs = logits.log_softmax(-1)
        sorted_log_probs = t.sort(log_probs, descending=False).values
        top_word_posn = (model.cfg.d_vocab - t.searchsorted(sorted_log_probs, log_probs[correct_token].item())).item()
        topk = log_probs.topk(5, dim=-1)
        top_logits = t.concat([logits[topk.indices], logits[correct_token].unsqueeze(0)])
        top_word_logprobs, top_word_indices = topk
        top_word_logprobs = t.concat([top_word_logprobs.squeeze(), log_probs[correct_token].unsqueeze(0)])
        top_words = model.to_str_tokens(top_word_indices.squeeze()) + [correct_str_token]
        rprint_output = [f"[dark_orange bold]logit = {logit:.3f}[/] | [bright_red bold]prob = {logprob.exp():.3f}[/] | {repr(word)}" for logit, logprob, word in zip(top_logits, top_word_logprobs, top_words)]
        
        # Get logits and stuff for the original ablated distribution
        ablated_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=[hook])[0, -1]
        ablated_log_probs = ablated_logits.log_softmax(-1)
        sorted_log_probs = t.sort(ablated_log_probs, descending=False).values
        top_word_posn_ablated = (model.cfg.d_vocab - t.searchsorted(sorted_log_probs, ablated_log_probs[correct_token].item())).item()
        topk = ablated_log_probs.topk(5, dim=-1)
        top_logits = t.concat([ablated_logits[topk.indices], ablated_logits[correct_token].unsqueeze(0)])
        top_word_logprobs, top_word_indices = topk
        top_word_logprobs = t.concat([top_word_logprobs.squeeze(), ablated_log_probs[correct_token].unsqueeze(0)])
        top_words = model.to_str_tokens(top_word_indices.squeeze()) + [correct_str_token]
        rprint_output_ablated = [f"[dark_orange bold]logit = {logit:.3f}[/] | [bright_red bold]prob = {logprob.exp():.3f}[/] | {repr(word)}" for logit, logprob, word in zip(top_logits, top_word_logprobs, top_words)]

        # Create and display table
        table = Table("Original", "Ablated", title=f"Correct = {repr(correct_str_token)}, Loss decrease from NNMH = {loss_decrease:.3f}")
        table.add_row("Top words:", "Top words:")
        table.add_row("", "")
        for output, output_ablated in zip(rprint_output[:-1], rprint_output_ablated):
            table.add_row(output, output_ablated)
        table.add_row("", "")
        table.add_row(f"Correct word (predicted at posn {top_word_posn}):", f"Correct word (predicted at posn {top_word_posn_ablated}):")
        table.add_row("", "")
        table.add_row(rprint_output[-1], rprint_output_ablated[-1])

        rprint(prev_tokens.replace("\n", "") + f"[dark_orange bold u]{correct_str_token}[/]" + next_tokens.replace("\n", ""))
        rprint(table)

    return caches



def find_best_improvements(
    str_token_list,
    loss_list,
    ablated_loss_list, 
    k = 15,
    print_table = False,
    worst = False, # if True, we take the worst examples (i.e. where 10.7 is least helpful)
):

    best_loss_decrease = []
    best_text = []
    best_indices = []

    for i, (stl, ll, all) in tqdm(list(enumerate(zip(str_token_list, loss_list, ablated_loss_list)))):

        loss_diff = (all - ll).squeeze()
        k_actual = min(k, loss_diff.shape[0])
        max_loss_decrease = loss_diff.topk(k_actual, largest=not(worst))
        
        for value, index in zip(max_loss_decrease.values, max_loss_decrease.indices):
            text = stl[max(0, index - 15): index + 2]
            # ! Why `:idx+2` ? Because loss_diff[idx] is large, meaning we failed to predict the `idx+1`-th element, so this should be the last one in our list. We're highlighting the thing we predicted wrong.
            if text:
                text[-1] = f"[bold dark_orange u]{repr(text[-1])}[/]"
                text = "".join(text)
                if "�" not in text:
                    best_loss_decrease.append(value.item())
                    best_text.append(text + "\n\n")
                    best_indices.append((i, index.item()))

    table = Table("CE-Loss Decrease", "Prompt", title="Prompts & Answers:")

    best_k_indices = []
    best_k_loss_decrease = []

    sorted_lists = sorted(list(zip(best_loss_decrease, best_text, best_indices)), key=lambda x: x[0], reverse=not(worst))
    for loss, text, idx in sorted_lists[:k]:
        table.add_row(f"{loss:.3f}", text)
        best_k_indices.append(idx)
        best_k_loss_decrease.append(loss)

    if print_table: rprint(table)

    return best_k_indices, best_k_loss_decrease


# def clear_plots():
#     for file in Path("/home/ubuntu/Transformerlens/transformer_lens/rs/callum/plots").glob("temp_file_*.html"):
#         file.unlink()




# def attn_scores_as_linear_func_of_queries(
#     batch_idx: Optional[Union[int, List[int], Int[Tensor, "batch"]]],
#     head: Tuple[int, int],
#     model: HookedTransformer,
#     cache: ActivationCache,
#     cache_baseline: ActivationCache,
#     src_indices: Int[Tensor, "batch"],
#     src_baseline_indices: Optional[Int[Tensor, "batch"]],
# ) -> Float[Tensor, "d_model"]:
#     '''
#     If you hold keys fixed, then attention scores are a linear function of the queries.
#     I want to fix the keys of head 10.7, and get a linear function mapping queries -> attention scores.
#     I can then see if (for example) the unembedding vector for the IO token has a really big image in this linear fn.

#     Here, if `src_baseline_indices` is not None, this means we should change the linear map, from key_IO_linear_map to
#     (key_IO_linear_map - key_baseline_linear_map). Same for the bias term.
#     '''
#     layer, head_idx = head
#     if isinstance(batch_idx, int):
#         batch_idx = [batch_idx]
#     if batch_idx is None:
#         batch_idx = range(len(cache["q", 0]))

#     keys_all = cache["k", layer][:, :, head_idx] # shape (all_batch, seq_K, d_head)
#     keys = keys_all[batch_idx, src_indices[batch_idx]]
#     if src_baseline_indices is not None:
#         keys_all_baseline = cache_baseline["k", layer][:, :, head_idx]
#         keys = keys - keys_all_baseline[batch_idx, src_baseline_indices[batch_idx]]
    
#     W_Q = model.W_Q[layer, head_idx].clone() # shape (d_model, d_head)
#     b_Q = model.b_Q[layer, head_idx].clone() # shape (d_head,)

#     linear_map = einops.einsum(W_Q, keys, "d_model d_head, batch d_head -> batch d_model") / (model.cfg.d_head ** 0.5)
#     bias_term = einops.einsum(b_Q, keys, "d_head, batch d_head -> batch") / (model.cfg.d_head ** 0.5)

#     if isinstance(batch_idx, int):
#         linear_map = linear_map[0]
#         bias_term = bias_term[0]

#     return linear_map, bias_term



# def attn_scores_as_linear_func_of_keys(
#     batch_idx: Optional[Union[int, List[int], Int[Tensor, "batch"]]],
#     head: Tuple[int, int],
#     model: HookedTransformer,
#     cache: ActivationCache,
#     dest_indices: Int[Tensor, "batch"],
#     use_baseline: bool,
# ) -> Float[Tensor, "d_model"]:
#     '''
#     If you hold queries fixed, then attention scores are a linear function of the keys.
#     I want to fix the queries of head 10.7, and get a linear function mapping keys -> attention scores.
#     I can then see if (for example) the embedding vector for the IO token has a really big image in this linear fn.

#     Here, if `use_baseline` is True, this implies that we'll be passing (key_IO - key_baseline) to this linear
#     map. So we want to make the bias zero, but not change the linear map.
#     '''
#     layer, head_idx = head
#     if isinstance(batch_idx, int):
#         batch_idx = [batch_idx]
#     if batch_idx is None:
#         batch_idx = range(len(cache["q", 0]))

#     queries = cache["q", layer][:, :, head_idx] # shape (all_batch, seq_K, d_head)
#     queries_at_END = queries[batch_idx, dest_indices[batch_idx]] # shape (batch, d_head)
    
#     W_K = model.W_K[layer, head_idx].clone() # shape (d_model, d_head)
#     b_K = model.b_K[layer, head_idx].clone() # shape (d_head,)

#     linear_map = einops.einsum(W_K, queries_at_END, "d_model d_head, batch d_head -> batch d_model") / (model.cfg.d_head ** 0.5)
#     bias_term = einops.einsum(b_K, queries_at_END, "d_head, batch d_head -> batch") / (model.cfg.d_head ** 0.5)

#     if isinstance(batch_idx, int):
#         linear_map = linear_map[0]
#         bias_term = bias_term[0]

#     if use_baseline:
#         # In this case, we assume the key-side vector supplied will be the difference between the key vectors for the IO and S1 tokens.
#         # We don't change the linear map, but we do change the bias term (because it'll be added then subtracted, i.e. it should be zero!)
#         bias_term *= 0

#     return linear_map, bias_term



# def decompose_attn_scores(
#     toks: Int[Tensor, "batch seq"],
#     dest_indices: Int[Tensor, "batch"],
#     src_indices: Int[Tensor, "batch"],
#     nnmh: Tuple[int, int],
#     model: HookedTransformer,
#     decompose_by: Literal["keys", "queries"],
#     src_baseline_indices: Optional[Int[Tensor, "batch"]] = None,
#     toks_baseline: Optional[Int[Tensor, "batch seq"]] = None,
#     intervene_on_query: Literal["sub_W_U_IO", "project_to_W_U_IO", None] = None,
#     intervene_on_key: Literal["sub_MLP0", "project_to_MLP0", None] = None,
#     use_effective_embedding: bool = False,
#     use_layer0_heads: bool = False,
# ):

#     t.cuda.empty_cache()
#     _, cache = model.run_with_cache(toks)
#     toks_baseline = toks if (toks_baseline is None) else toks_baseline
#     _, cache_baseline = model.run_with_cache(toks_baseline)

#     batch_size, seq_len = toks.shape

#     src_toks = toks[range(batch_size), src_indices]

#     use_baseline = (src_baseline_indices is not None)
#     if use_baseline:
#         src_baseline_toks = toks_baseline[range(batch_size), src_baseline_indices]

#     # * Get the MLP0 output (note that we need to be careful here if we're subtracting the S1 baseline, because we actually need the 2 different MLP0s)
#     if use_effective_embedding:
#         W_EE_dict = get_effective_embedding(model, use_codys_without_attention_changes=False)
#         W_EE = (W_EE_dict["W_E (including MLPs)"] - W_EE_dict["W_E (no MLPs)"]) if use_layer0_heads else W_EE_dict["W_E (only MLPs)"]
#         MLP0_output = W_EE[src_toks]
#         if use_baseline: MLP0_output_baseline = W_EE[src_baseline_toks]
#     else:
#         if use_layer0_heads:
#             MLP0_output = cache["mlp_out", 0][range(batch_size), src_indices] + cache["attn_out", 0][range(batch_size), src_indices]
#             MLP0_output_baseline = cache_baseline["mlp_out", 0][range(batch_size), src_baseline_indices] + cache_baseline["attn_out", 0][range(batch_size), src_baseline_indices]
#         else:
#             MLP0_output = cache["mlp_out", 0][range(batch_size), src_indices]
#             MLP0_output_baseline = cache_baseline["mlp_out", 0][range(batch_size), src_baseline_indices]
#     MLP0_output_scaled = (MLP0_output - MLP0_output.mean(-1, keepdim=True)) / MLP0_output.var(dim=-1, keepdim=True).pow(0.5)


#     if decompose_by == "keys":

#         assert intervene_on_query in [None, "sub_W_U_IO", "project_to_W_U_IO"]

#         decomp_seq_pos_indices = src_indices
#         lin_map_seq_pos_indices = dest_indices

#         unembeddings = model.W_U.T[src_toks]
#         unembeddings_scaled = (unembeddings - unembeddings.mean(-1, keepdim=True)) / unembeddings.var(dim=-1, keepdim=True).pow(0.5)

#         resid_pre = cache["resid_pre", nnmh[0]]
#         resid_pre_normalised = (resid_pre - resid_pre.mean(-1, keepdim=True)) / resid_pre.var(dim=-1, keepdim=True).pow(0.5)
#         resid_pre_normalised_slice = resid_pre_normalised[range(batch_size), lin_map_seq_pos_indices]

#         W_Q = model.W_Q[nnmh[0], nnmh[1]]
#         b_Q = model.b_Q[nnmh[0], nnmh[1]]
#         q_name = utils.get_act_name("q", nnmh[0])
#         q_raw = cache[q_name].clone()
        
#         # ! (1A)
#         # * Get 2 linear functions from keys -> attn scores, corresponding to the 2 different components of query vectors: (∥ / ⟂) to W_U[IO]
#         if intervene_on_query == "project_to_W_U_IO":
#             resid_pre_in_io_dir, resid_pre_in_io_perpdir = project(resid_pre_normalised_slice, unembeddings)

#             # Overwrite the query-side vector in the cache with the projection in the unembedding direction
#             q_new = einops.einsum(resid_pre_in_io_dir, W_Q, "batch d_model, d_model d_head -> batch d_head")
#             q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
#             cache_dict_io_dir = {**cache.cache_dict, **{q_name: q_raw.clone()}}
#             cache_io_dir = ActivationCache(cache_dict=cache_dict_io_dir, model=model)
#             linear_map_io_dir, bias_term_io_dir = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, cache=cache_io_dir, dest_indices=dest_indices, use_baseline=use_baseline)

#             # Overwrite the query-side vector with the bit that's perpendicular to the IO unembedding (plus the bias term)
#             q_new = einops.einsum(resid_pre_in_io_perpdir, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
#             q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
#             cache_dict_io_perpdir = {**cache.cache_dict, **{q_name: q_raw.clone()}}
#             cache_io_perpdir = ActivationCache(cache_dict=cache_dict_io_perpdir, model=model)
#             linear_map_io_perpdir, bias_term_io_perpdir = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, cache=cache_io_perpdir, dest_indices=dest_indices, use_baseline=use_baseline)
            
#             linear_map_dict = {"IO_dir": (linear_map_io_dir, bias_term_io_dir), "IO_perp": (linear_map_io_perpdir, bias_term_io_perpdir)}

#         # ! (1A)
#         # * Get new linear function from keys -> attn scores, corresponding to subbing in W_U[IO] as queryside vector
#         # * TODO - replace `sub`, because it implies `subtract` rather than `substitute`
#         elif intervene_on_query == "sub_W_U_IO":
#             # Overwrite the query-side vector by replacing it with the (normalized) W_U[IO] unembeddings
#             q_new = einops.einsum(unembeddings_scaled, W_Q, "batch d_model, d_model d_head -> batch d_head") + b_Q
#             q_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = q_new
#             cache_dict_io_subbed = {**cache.cache_dict, **{q_name: q_raw.clone()}}
#             cache_io_subbed = ActivationCache(cache_dict=cache_dict_io_subbed, model=model)
#             linear_map_io_subbed, bias_term_io_subbed = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, cache=cache_io_subbed, dest_indices=dest_indices, use_baseline=use_baseline)
            
#             linear_map_dict = {"IO_sub": (linear_map_io_subbed, bias_term_io_subbed)}

#         # * Get linear function from keys -> attn scores (no intervention on query)
#         else:
#             linear_map, bias_term = attn_scores_as_linear_func_of_keys(batch_idx=None, head=nnmh, model=model, cache=cache, dest_indices=dest_indices, use_baseline=use_baseline)
#             linear_map_dict = {"unchanged": (linear_map, bias_term)}


    
#     elif decompose_by == "queries":

#         assert intervene_on_key in [None, "sub_MLP0", "project_to_MLP0"]

#         decomp_seq_pos_indices = dest_indices
#         lin_map_seq_pos_indices = src_indices

#         resid_pre = cache["resid_pre", nnmh[0]]
#         resid_pre_normalised = (resid_pre - resid_pre.mean(-1, keepdim=True)) / resid_pre.var(dim=-1, keepdim=True).pow(0.5)
#         resid_pre_normalised_slice = resid_pre_normalised[range(batch_size), lin_map_seq_pos_indices]
#         resid_pre_normalised_slice_baseline = resid_pre_normalised[range(batch_size), src_baseline_indices]

#         W_K = model.W_K[nnmh[0], nnmh[1]]
#         b_K = model.b_K[nnmh[0], nnmh[1]]
#         k_name = utils.get_act_name("k", nnmh[0])
#         k_raw = cache[k_name].clone()
        
#         # ! (2B)
#         # * Get 2 linear functions from queries -> attn scores, corresponding to the 2 different components of key vectors: (∥ / ⟂) to MLP0_out
#         if intervene_on_key == "project_to_MLP0":
#             resid_pre_in_mlp0_dir, resid_pre_in_mlp0_perpdir = project(resid_pre_normalised_slice, MLP0_output)
#             resid_pre_in_mlp0_dir_baseline, resid_pre_in_mlp0_perpdir_baseline = project(resid_pre_normalised_slice_baseline, MLP0_output_baseline)

#             # Overwrite the key-side vector in the cache with the projection in the MLP0_output direction
#             # Do the same with the S1 baseline (note that we might not actually use it, but it's good to have it there)
#             k_new = einops.einsum(resid_pre_in_mlp0_dir, W_K, "batch d_model, d_model d_head -> batch d_head")
#             k_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = k_new
#             cache_dict_mlp0_dir = {**cache.cache_dict, **{k_name: k_raw.clone()}}
#             cache_mlp0_dir = ActivationCache(cache_dict=cache_dict_mlp0_dir, model=model)
#             if use_baseline:
#                 k_new_baseline = einops.einsum(resid_pre_in_mlp0_dir_baseline, W_K, "batch d_model, d_model d_head -> batch d_head")
#                 k_raw[range(batch_size), src_baseline_indices, nnmh[1]] = k_new_baseline
#                 cache_baseline_dict_mlp0_dir = {**cache.cache_dict, **{k_name: k_raw.clone()}}
#                 cache_baseline_mlp0_dir = ActivationCache(cache_baseline_dict=cache_baseline_dict_mlp0_dir, model=model)
#             # ! (3B)
#             # * This function (the `use_baseline` argument) is where we subtract the baseline from the linear map from queries -> attn scores
#             # * Obviously the same is true for the other 3 instances of the `attn_scores_as_linear_func_of_querieS` function below
#             linear_map_mlp0_dir, bias_term_mlp0_dir = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, cache=cache_mlp0_dir, cache_baseline=cache_baseline_mlp0_dir, src_indices=src_indices, src_baseline_indices=src_baseline_indices)

#             # Overwrite the key-side vector with the bit that's perpendicular to the MLP0_output (plus the bias term)
#             k_new = einops.einsum(resid_pre_in_mlp0_perpdir, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
#             k_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = k_new
#             cache_dict_mlp0_perpdir = {**cache.cache_dict, **{k_name: k_raw.clone()}}
#             cache_mlp0_perpdir = ActivationCache(cache_dict=cache_dict_mlp0_perpdir, model=model)
#             if use_baseline:
#                 k_new_baseline = einops.einsum(resid_pre_in_mlp0_perpdir_baseline, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
#                 k_raw[range(batch_size), src_baseline_indices, nnmh[1]] = k_new_baseline
#                 cache_baseline_dict_mlp0_perpdir = {**cache.cache_dict, **{k_name: k_raw.clone()}}
#                 cache_baseline_mlp0_perpdir = ActivationCache(cache_baseline_dict=cache_baseline_dict_mlp0_perpdir, model=model)
#             linear_map_mlp0_perpdir, bias_term_mlp0_perpdir = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, cache=cache_mlp0_perpdir, cache_baseline=cache_baseline_mlp0_perpdir, src_indices=src_indices, src_baseline_indices=src_baseline_indices)
            
#             linear_map_dict = {"MLP0_dir": (linear_map_mlp0_dir, bias_term_mlp0_dir), "MLP0_perp": (linear_map_mlp0_perpdir, bias_term_mlp0_perpdir)}
        
#         # ! (2B)
#         # * Get new linear function from queries -> attn scores, corresponding to subbing in MLP0_output as keyside vector
#         elif intervene_on_key == "sub_MLP0":
#             # Overwrite the key-side vector by replacing it with the (normalized) MLP0_output
#             assert not(use_baseline), "This will behave weirdly right now."
#             k_new = einops.einsum(MLP0_output_scaled, W_K, "batch d_model, d_model d_head -> batch d_head") + b_K
#             k_raw[range(batch_size), lin_map_seq_pos_indices, nnmh[1]] = k_new
#             cache_dict_mlp0_subbed = {**cache.cache_dict, **{k_name: k_raw.clone()}}
#             cache_mlp0_subbed = ActivationCache(cache_dict=cache_dict_mlp0_subbed, model=model)
#             linear_map_mlp0_subbed, bias_term_mlp0_subbed = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, cache=cache_mlp0_subbed, cache_baseline=cache_mlp0_subbed, src_indices=src_indices, src_baseline_indices=src_baseline_indices)
            
#             linear_map_dict = {"MLP0_sub": (linear_map_mlp0_subbed, bias_term_mlp0_subbed)}

#         # * Get linear function from queries -> attn scores (no intervention on key)
#         else:
#             linear_map, bias_term = attn_scores_as_linear_func_of_queries(batch_idx=None, head=nnmh, model=model, cache=cache, cache_baseline=cache, src_indices=src_indices, src_baseline_indices=src_baseline_indices)
#             linear_map_dict = {"unchanged": (linear_map, bias_term)}


#     t.cuda.empty_cache()

#     contribution_to_attn_scores_list = []

#     # * This is where we get the thing we're projecting keys onto if required (i.e. if we're decomposing by keys, and want to split into ||MLP0 and ⟂MLP0)
#     if (intervene_on_key is not None) and (decompose_by == "keys"):
#         assert intervene_on_key == "project_to_MLP0", "If you're decomposing by key component, then 'intervene_on_key' must be 'project_to_MLP0' or None."
#         contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)
#         contribution_to_attn_scores_names = ["MLP0_dir", "MLP0_perp"]

#     # * This is where we get the thing we're projecting queries onto if required (i.e. if we're decomposing by queries, and want to split into ||W_U[IO] and ⟂W_U[IO])
#     elif (intervene_on_query is not None) and (decompose_by == "queries"):
#         assert intervene_on_query == "project_to_W_U_IO", "If you're decomposing by key component, then 'intervene_on_query' must be 'project_to_W_U_IO' or None."
#         unembeddings = model.W_U.T[src_toks]
#         contribution_to_attn_scores_shape = (2, 1 + nnmh[0], model.cfg.n_heads + 1)
#         contribution_to_attn_scores_names = ["IO_dir", "IO_perp"]

#     # * We're not projecting by anything when we get the decomposed bits
#     else:
#         contribution_to_attn_scores_shape = (1, 1 + nnmh[0], model.cfg.n_heads + 1)
#         contribution_to_attn_scores_names = ["unchanged"]



#     def get_decomposed_components(component_name, layer=None):
#         '''
#         This function does the following:
#             > Get the value we want from the cache (at the appopriate sequence positions for the decomposition: either "IO" or "end")
#             > If we need to project it in a direction, then apply that projection (this gives it an extra dim at the start)
#             > If we need to subtract the mean of S1, do that too.
#         '''
#         assert component_name in ["result", "mlp_out", "embed", "pos_embed"]
        
#         # Index from ioi cache
#         component_output: Float[Tensor, "batch *n_heads d_model"] = cache[component_name, layer][range(batch_size), decomp_seq_pos_indices]

#         # Apply scaling
#         component_output_scaled = component_output / (ln_scale.unsqueeze(1) if (component_name == "result") else ln_scale)

#         # Apply projections
#         # ! (2A)
#         # * This is where we decompose the query-side output of each component, by possibly projecting it onto the ||W_U[IO] and ⟂W_U[IO] directions
#         if (decompose_by == "queries") and (intervene_on_query == "project_to_W_U_IO"):
#             projection_dir = einops.repeat(unembeddings, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else unembeddings
#             component_output_scaled = t.stack(project(component_output_scaled, projection_dir))
#         # ! (1B)
#         # * This is where we decompose the key-side output of each component, by possibly projecting it onto the ||MLP0 and ⟂MLP0 directions
#         elif (decompose_by == "keys") and (intervene_on_key == "project_to_MLP0"):
#             projection_dir = einops.repeat(MLP0_output, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else MLP0_output
#             component_output_scaled = t.stack(project(component_output_scaled, projection_dir))

#         # ! (3A)
#         # * This is where we subtract the keyside component baseline of S2 (if our decomposition is by-keys)
#         # * This involves going through exactly the same process as above, except with S2 (I'll make the code shorter)
#         if (decompose_by == "keys") and use_baseline:
#             # Calculate scaled baseline
#             component_output_baseline = cache_baseline[component_name, layer][range(batch_size), src_baseline_indices]
#             component_output_scaled_baseline = component_output_baseline / (ln_scale_baseline.unsqueeze(1) if (component_name == "result") else ln_scale_baseline)
#             # Apply projections, if required
#             if (intervene_on_key == "project_to_MLP0"):
#                 projection_dir_baseline = einops.repeat(MLP0_output_baseline, "b d_m -> b heads d_m", heads=model.cfg.n_heads) if (component_name == "result") else MLP0_output_baseline
#                 component_output_scaled_baseline = t.stack(project(component_output_scaled_baseline, projection_dir_baseline))
#             # Subtract baseline
#             component_output_scaled = component_output_scaled - component_output_scaled_baseline

#         return component_output_scaled


#     results_dict = {}

#     for name, (linear_map, bias_term) in linear_map_dict.items():
        
#         # Check linear map is valid
#         assert linear_map.shape == (batch_size, model.cfg.d_model)
#         assert bias_term.shape == (batch_size,)

#         # Create tensor to store all the values for this facet plot (possibly 2 facet plots, if we're splitting by projecting our decomposed components)
#         contribution_to_attn_scores = t.zeros(contribution_to_attn_scores_shape)

#         # Get scale factor we'll be dividing all our components by
#         ln_scale = cache["scale", nnmh[0], "ln1"][range(batch_size), decomp_seq_pos_indices, nnmh[1]]
#         if use_baseline:
#             ln_scale_baseline = cache_baseline["scale", nnmh[0], "ln1"][range(batch_size), src_baseline_indices, nnmh[1]]

#         # We start with all the things before attn heads and MLPs
#         embed_scaled = get_decomposed_components("embed")
#         pos_embed_scaled = get_decomposed_components("pos_embed")
#         # Add these to the results tensor. Note we use `:` because this covers cases where the first dim is 1 (no projection split) or 2 (projection split)
#         contribution_to_attn_scores[:, 0, 0] = einops.einsum(embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
#         contribution_to_attn_scores[:, 0, 1] = einops.einsum(pos_embed_scaled, linear_map, "... batch d_model, batch d_model -> ... batch").mean(-1)
#         # Add the bias term (this is only ever added to the last term, because it's the perpendicular one)
#         contribution_to_attn_scores[-1, 0, 2] = bias_term.mean()

#         for layer in range(nnmh[0]):

#             # Calculate output of each attention head, split by projecting onto MLP0 output if necessary, then add to our results tensor
#             # z = cache["z", layer][range(batch_size), decomp_seq_pos_indices]
#             # result = einops.einsum(z, model.W_O[layer], "batch n_heads d_head, n_heads d_head d_model -> batch n_heads d_model")
#             results_scaled = get_decomposed_components("result", layer)
#             contribution_to_attn_scores[:, 1 + layer, :model.cfg.n_heads] = einops.einsum(
#                 results_scaled, linear_map, 
#                 "... batch n_heads d_model, batch d_model -> ... n_heads batch"
#             ).mean(-1)

#             # Calculate output of the MLPs, split by projecting onto MLP0 output if necessary, then add to our results tensor
#             mlp_out_scaled = get_decomposed_components("mlp_out", layer)
#             contribution_to_attn_scores[:, 1 + layer, -1] = einops.einsum(
#                 mlp_out_scaled, linear_map,
#                 "... batch d_model, batch d_model -> ... batch"
#             ).mean(-1)
        
#         contribution_to_attn_scores_list.append(contribution_to_attn_scores.squeeze())

#         for name2, contribution_to_attn_scores_slice in zip(contribution_to_attn_scores_names, contribution_to_attn_scores):
#             names = tuple(sorted([name, name2]))
#             results_dict[names] = contribution_to_attn_scores_slice.squeeze()

#     if len(contribution_to_attn_scores_list) == 1:
#         contribution_to_attn_scores = contribution_to_attn_scores_list[0]
#     else:
#         contribution_to_attn_scores = t.stack(contribution_to_attn_scores_list)

#     if len(results_dict) == 1:
#         return results_dict[list(results_dict.keys())[0]]
#     else:
#         return results_dict

    



# def decompose_attn_scores_full(
#     toks: Int[Tensor, "batch seq"],
#     dest_indices: Int[Tensor, "batch"],
#     src_indices: Int[Tensor, "batch"],
#     src_baseline_indices: Optional[Int[Tensor, "batch"]],
#     nnmh: Tuple[int, int],
#     model: HookedTransformer,
#     use_effective_embedding: bool = False,
#     use_layer0_heads: bool = False,
# ):
#     t.cuda.empty_cache()
#     _, cache = model.run_with_cache(toks)

#     batch_size, seq_len = toks.shape

#     src_toks = toks[range(batch_size), src_indices]

#     ln_scale_src = cache["scale", nnmh[0], "ln1"][range(batch_size), src_indices, nnmh[1]]
#     ln_scale_dest = cache["scale", nnmh[0], "ln1"][range(batch_size), dest_indices, nnmh[1]]

#     use_baseline = (src_baseline_indices is not None)
#     if use_baseline:
#         src_baseline_toks = toks[range(batch_size), src_baseline_indices]
#         ln_scale_src_baseline = cache["scale", nnmh[0], "ln1"][range(batch_size), src_baseline_indices, nnmh[1]]


#     # * Get the MLP0 output (note that we need to be careful here if we're subtracting the S1 baseline, because we actually need the 2 different MLP0s)
#     if use_effective_embedding:
#         W_EE_dict = get_effective_embedding(model, use_codys_without_attention_changes=False)
#         W_EE = (W_EE_dict["W_E (including MLPs)"] - W_EE_dict["W_E (no MLPs)"]) if use_layer0_heads else W_EE_dict["W_E (only MLPs)"]
#         MLP0_output = W_EE[src_toks]
#         if use_baseline: MLP0_output_baseline = W_EE[src_baseline_toks]
#     else:
#         if use_layer0_heads:
#             MLP0_output = cache["mlp_out", 0][range(batch_size), src_indices] + cache["attn_out", 0][range(batch_size), src_indices]
#             MLP0_output_baseline = cache["mlp_out", 0][range(batch_size), src_indices] + cache["attn_out", 0][range(batch_size), src_indices]
#         else:
#             MLP0_output = cache["mlp_out", 0][range(batch_size), src_indices]
#             MLP0_output_baseline = cache["mlp_out", 0][range(batch_size), src_indices]

#     # * Get the unembeddings
#     unembeddings = model.W_U.T[src_toks]

#     t.cuda.empty_cache()

#     contribution_to_attn_scores = t.zeros(
#         4, # this is for the 4 options: (∥ / ⟂) to (unembed of IO on query side / MLP0 on key side)
#         3 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the query-side
#         3 + (nnmh[0] * (1 + model.cfg.n_heads)), # this is for the key-side
#     )

#     keyside_components = []
#     queryside_components = []

#     keys_decomposed = t.zeros(2, 3 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)
#     queries_decomposed = t.zeros(2, 3 + (nnmh[0] * (1 + model.cfg.n_heads)), batch_size, model.cfg.d_head)

#     def get_component(component_name, layer=None, keyside=False):
#         '''
#         Gets component (key or query side).

#         If we need to subtract the baseline, it returns both the component for IO and the component for S1 (so we can project then subtract scores from each other).
#         '''
#         full_component = cache[component_name, layer]
#         if keyside:
#             component_src = full_component[range(batch_size), src_indices] / (ln_scale_src.unsqueeze(1) if (component_name == "result") else ln_scale_src)
#             if use_baseline:
#                 component_src_baseline = full_component[range(batch_size), src_baseline_indices] / (ln_scale_src_baseline.unsqueeze(1) if (component_name == "result") else ln_scale_src_baseline)
#             return (component_src, component_src_baseline) if use_baseline else component_src
#         else:
#             component_END = full_component[range(batch_size), dest_indices] / (ln_scale_dest.unsqueeze(1) if (component_name == "result") else ln_scale_dest)
#             return component_END

#     b_K = model.b_K[nnmh[0], nnmh[1]]
#     if use_baseline: b_K *= 0
#     b_K = einops.repeat(b_K, "d_head -> batch d_head", batch=batch_size)
#     b_Q = einops.repeat(model.b_Q[nnmh[0], nnmh[1]], "d_head -> batch d_head", batch=batch_size)

#     # First, get the biases and direct terms
#     keyside_components.extend([
#         b_K,
#         get_component("embed", keyside=True),
#         get_component("pos_embed", keyside=True),
#     ])
#     queryside_components.extend([
#         b_Q,
#         get_component("embed", keyside=False),
#         get_component("pos_embed", keyside=False),
#     ])

#     # Next, get all the MLP terms
#     for layer in range(nnmh[0]):
#         keyside_components.append(get_component("mlp_out", layer=layer, keyside=True))
#         queryside_components.append(get_component("mlp_out", layer=layer, keyside=False))

#     # Lastly, all the heads
#     for layer in range(nnmh[0]):
#         keyside_heads = get_component("result", layer=layer, keyside=True)
#         queryside_heads = get_component("result", layer=layer, keyside=False)
#         for head in range(model.cfg.n_heads):
#             if use_baseline:
#                 keyside_components.append((keyside_heads[0][:, head, :], keyside_heads[1][:, head, :]))
#             else:
#                 keyside_components.append(keyside_heads[:, head, :])
#             queryside_components.append(queryside_heads[:, head, :])

#     # Now, we do the projection thing...
#     # ... for keys ....
#     keys_decomposed[1, 0] = keyside_components[0]
#     for i, keyside_component in enumerate(keyside_components[1:], 1):
#         if use_baseline:
#             keyside_component_src, keyside_component_src_baseline = keyside_component
#             projections = project(keyside_component_src, MLP0_output), project(keyside_component_src_baseline, MLP0_output_baseline)
#             projections = t.stack([projections[0][0] - projections[1][0], projections[0][1] - projections[1][1]])
#         else:
#             projections = t.stack(list(project(keyside_component, MLP0_output)))
#         keys_decomposed[:, i] = einops.einsum(projections.cpu(), model.W_K[nnmh[0], nnmh[1]].cpu(), "projection batch d_model, d_model d_head -> projection batch d_head")
#     # ... and for queries ...
#     queries_decomposed[1, 0] = queryside_components[0]
#     for i, queryside_component in enumerate(queryside_components[1:], 1):
#         projections = t.stack(project(queryside_component, unembeddings))
#         queries_decomposed[:, i] = einops.einsum(projections.cpu(), model.W_Q[nnmh[0], nnmh[1]].cpu(), "projection batch d_model, d_model d_head -> projection batch d_head")
    

#     # Finally, we do the outer product thing
#     for (key_idx, keyside_component) in enumerate(keys_decomposed.unbind(dim=1)):
#         for (query_idx, queryside_component) in enumerate(queries_decomposed.unbind(dim=1)):
#             contribution_to_attn_scores[:, query_idx, key_idx] = einops.einsum(
#                 queryside_component,
#                 keyside_component,
#                 "q_projection batch d_head, k_projection batch d_head -> q_projection k_projection batch"
#             ).mean(-1).flatten() / (model.cfg.d_head ** 0.5)


#     return contribution_to_attn_scores