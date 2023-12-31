
import numpy as np
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer

from sae_training.sparse_autoencoder import SparseAutoencoder


def plot_attn(patterns, token_df, title="", facet_col_labels = ["Original", "Reconstructed"]):
    '''
    # patterns_original = cache[utils.get_act_name("pattern", LAYER_IDX)][0,HEAD_IDX].detach().cpu()
    # patterns_reconstructed = cache_reconstructed_query[utils.get_act_name("pattern", LAYER_IDX)][0,HEAD_IDX].detach().cpu()
    patterns_original = cache[utils.get_act_name("attn_scores", LAYER_IDX)][0,HEAD_IDX].detach().cpu()
    patterns_reconstructed = cache_reconstructed_query[utils.get_act_name("attn_scores", LAYER_IDX)][0,HEAD_IDX].detach().cpu()
    both_patterns = torch.stack([patterns_original, patterns_reconstructed])
    plot_attn(both_patterns.detach().cpu(), token_df, title="Original and Reconstructed Attention Distribution")
    
    '''
    fig = px.imshow(patterns, text_auto=".2f", title=title,
                    facet_col=0,
                    color_continuous_midpoint=0,
                    color_continuous_scale="RdBu",
                    )
    
    tickvals = 1+ np.arange(patterns.shape[2])
    ticktext = token_df["unique_token"].tolist()
    
    # add tokens as x-ticks and y-ticks, for each facet
    # Update x-ticks and y-ticks for each facet
    for i in range(len(facet_col_labels)):
        fig.update_xaxes(
            dict(tickmode='array', tickvals=tickvals, ticktext=ticktext),
            row=1, col=i+1
        )
        fig.update_yaxes(
            dict(tickmode='array', tickvals=tickvals, ticktext=ticktext),
            row=1, col=i+1
        )
    
    
    # add facet col labels:
    for i, label in enumerate(facet_col_labels):
        fig.layout.annotations[i].text = label
        fig.layout.annotations[i].font.size = 20
        
    fig.update_layout(
        width=1200,
        height=800,
    )
    fig.show()

def plot_line_with_top_10_labels(tensor, title="", n = 10):
    """
    Plots a line chart of the given tensor with the top 10 values annotated.

    :param tensor: A PyTorch tensor to be plotted.
    """
    # Convert the tensor to a Pandas series for easier processing
    data = pd.Series(tensor.detach().cpu().numpy())

    # Create the line chart
    fig = px.line(data, labels={'value': 'Activation Value', 'index': 'Feature Id'},
                    title=title)

    # Identify the top 10 values
    top_10_indices = data.nlargest(n).index

    # Annotate the top 10 values
    for idx in top_10_indices:
        fig.add_annotation(x=idx, y=data[idx],
                           text=f"{idx}",
                           showarrow=False, arrowhead=1,
                           ax=0, ay=-40)  # Adjust ax, ay as needed
    # Remove the legend
    fig.update_layout(showlegend=False)

    # Show the plot
    fig.show()


def plot_attn_score_by_feature(model: HookedTransformer, sparse_autoencoder: SparseAutoencoder, feature_ids, cache, token_df, pos_interest, vals = None):
    '''
    '''
    
    layer_index = sparse_autoencoder.cfg.hook_point_layer
    head_index = sparse_autoencoder.cfg.hook_point_head_index

    k = cache[f"blocks.{layer_index}.attn.hook_k"][0,:(1+pos_interest),head_index]
    # score_contributions = sparse_autoencoder.W_enc[:,inds].T @ k.T
    score_contributions = sparse_autoencoder.W_dec[feature_ids] @ k.T
    print(score_contributions.sum(dim=0))
    if vals is not None:
        score_contributions = score_contributions * vals.unsqueeze(1)
    fig = px.imshow(score_contributions.detach().cpu(), 
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    labels = dict(y="Feature", x="Token"),
                    text_auto=".2f", title="")
    # add xticks and y ticks
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=1+np.arange(score_contributions.shape[1]),
            ticktext=token_df["str_tokens"].tolist(),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(score_contributions.shape[0]),
            ticktext=list(feature_ids.detach().cpu().numpy()),
        ),
    )
    return fig
    
def plot_unembed_score_by_feature(model: HookedTransformer, sparse_autoencoder: SparseAutoencoder, feature_ids, token_df, vals = None):

    layer_index = sparse_autoencoder.cfg.hook_point_layer
    head_index = sparse_autoencoder.cfg.hook_point_head_index

    token_ids = model.to_tokens(token_df.str_tokens.to_list(), prepend_bos=False).squeeze()
    # W_U_normed = W_U / W_U.norm(dim=-1).unsqueeze(-1)
    resid_stream_projection =  sparse_autoencoder.W_dec[feature_ids] @ model.W_Q[layer_index, head_index].T @ model.W_U[:,token_ids]
    
    if vals is not None:
        resid_stream_projection = resid_stream_projection * vals.unsqueeze(1)
    fig = px.imshow(resid_stream_projection.detach().cpu(), 
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    labels = dict(y="Feature", x="Token"),
                    text_auto=".2f", title="")
    # add xticks and y ticks
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(resid_stream_projection.shape[1]),
            ticktext=token_df["str_tokens"].tolist(),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(resid_stream_projection.shape[0]),
            ticktext=list(feature_ids.detach().cpu().numpy()),
        ),
    )
    return fig

def plot_feature_unembed_bar(model: HookedTransformer, feature_id, sparse_autoencoder: SparseAutoencoder, feature_name = ""):
    
    layer_index = sparse_autoencoder.cfg.hook_point_layer
    head_index = sparse_autoencoder.cfg.hook_point_head_index
    
    # norm_unembed = model.W_U / model.W_U.norm(dim=0)[None: None]
    # feature_unembed = sparse_autoencoder.W_dec[feature_id] @ norm_unembed
    feature_unembed = sparse_autoencoder.W_dec[feature_id] @ model.W_Q[layer_index, head_index].T @  model.W_U
    # feature_unembed = sparse_autoencoder.W_dec[feature_id] @  model.W_U
    # torch.topk(unembed_4795,10)

    feature_unembed_df = pd.DataFrame(
        feature_unembed.detach().cpu().numpy(),
        columns = [feature_name],
        index = [model.tokenizer.decode(i) for i in list(range(50257))]
    )

    feature_unembed_df = feature_unembed_df.sort_values(feature_name, ascending=False).reset_index().rename(columns={'index': 'token'})
    fig = px.bar(feature_unembed_df.head(20).sort_values(feature_name, ascending=True),
                 color_continuous_midpoint=0,
                 color_continuous_scale="RdBu",
            y = 'token', x = feature_name, orientation='h', color = feature_name, hover_data=[feature_name])

    fig.update_layout(
        width=800,
        height=600,
    )

    # fig.write_image(f"figures/{str(feature_id)}_{feature_name}.png")
    fig.show()

def plot_qk_via_feature(model: HookedTransformer, feature_id, sparse_autoencoder: SparseAutoencoder, feature_name = "", highlight_tokens = []):
    
    layer_index = sparse_autoencoder.cfg.hook_point_layer
    head_index = sparse_autoencoder.cfg.hook_point_head_index
    
    eff_embed = model.W_E + model.blocks[0].mlp(model.blocks[0].ln2(model.W_E[None] + model.blocks[0].attn.b_O))
    eff_emb_in_key_space =  eff_embed @ model.W_K[layer_index,head_index] @ sparse_autoencoder.W_dec[feature_id]
    # feature_unembed = sparse_autoencoder.W_dec[feature_id] @ model.W_Q[LAYER_IDX,HEAD_IDX].T @  model.W_U
    feature_unembed = sparse_autoencoder.W_enc[:,feature_id] @ model.W_Q[layer_index,head_index].T @  model.W_U
    
    df = pd.DataFrame(dict(
        eff_emb_in_key_space=eff_emb_in_key_space[0].detach().cpu().numpy(),
        feature_unembed = feature_unembed.detach().cpu().numpy(),
        token = [model.tokenizer.decode(i) for i in range(50257)],
    ))
    
    df["token_of_interest"] = df["token"].isin(highlight_tokens)
    
    # add a column to df with text for the largest 10 values (positive and negative) 
    # that we can use to label these points
    top_10_key = df.sort_values("eff_emb_in_key_space", ascending=False).head(6)
    top_10_proj = df.sort_values("feature_unembed", ascending=False).head(6)
    
    top_10_key["text"] = top_10_key.apply(lambda x: f"{x['token']}", axis=1)
    top_10_proj["text"] = top_10_proj.apply(lambda x: f"{x['token']}", axis=1)
    
    # Merging the top and bottom points for annotation
    points_to_annotate = pd.concat([top_10_key, top_10_proj])

    fig = px.scatter(
        df,
        x="eff_emb_in_key_space",
        y = "feature_unembed",
        color="token_of_interest",
        color_continuous_scale="RdBu",
        # color="score_contributions",
        # text="text",
        # opacity=0.3,
        hover_data=["token"],
        labels=dict(eff_emb_in_key_space="Token to Feature Virtual Weight", feature_unembed="Unembed to Feature Virtual Weight"),
        title=f"Feature {feature_id} {feature_name}",
        template="plotly",
        marginal_x="histogram",
        marginal_y="histogram",
    )
    

    for _, row in points_to_annotate.iterrows():
        fig.add_annotation(x=row['eff_emb_in_key_space'], y=row['feature_unembed'],
                           text=row['text'], showarrow=False, arrowhead=1,
                           ax=20, ay=-40)

    
    fig.update_layout(
        width=1200,
        height=1200,
    )
    fig.show()