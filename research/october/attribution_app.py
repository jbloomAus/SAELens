import plotly.express as px
import streamlit as st
import torch
from attribution_app_functions import (
    calculate_feature_attribution,
    convert_sparse_feature_to_long_df,
    get_prediction_df,
    make_token_df,
)
from streamlit.components.v1 import iframe
from transformer_lens import HookedTransformer

from sae_lens import SAE

# Set up page config
st.set_page_config(layout="wide")

model_name = "gemma-2-2b"
sae_release = "gemma-scope-2b-pt-res-canonical"
sae_id = "layer_20/width_16k/canonical"


st.write(f"Model: {model_name}, SAE: {sae_release}/{sae_id}")


# Initialize device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and SAE in session state
if "model" not in st.session_state:

    @st.cache_resource
    def load_model():
        return HookedTransformer.from_pretrained(model_name, device=device)

    st.session_state.model = load_model()

if "sae" not in st.session_state:

    @st.cache_resource
    def load_sae():
        sae, _, _ = SAE.from_pretrained(
            sae_release,
            sae_id,
            device=device,
        )
        sae.fold_W_dec_norm()
        return sae

    st.session_state.sae = load_sae()


model = st.session_state.model
sae = st.session_state.sae
# Title and description
st.title("Token Attribution Demo")
st.markdown("Enter text below to see how it's broken into tokens.")

# Custom CSS for token display
st.markdown(
    """
    <style>
        div[data-testid="column"] {
            width: fit-content !important;
            flex: unset;
        }
        div[data-testid="column"] * {
            width: fit-content !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Text input
text = st.text_input("Enter text:", value="Tiger Woods plays the sport of")

st.write(text)


if not text:
    pass

tokens = model.to_str_tokens(text)
token_ids = model.to_tokens(text)
token_df = make_token_df(token_ids, model=model)
prediction_df = get_prediction_df(text, model, top_k=100)

assert len(tokens) == len(token_df)

st.write("### Tokenized Text")
# Create columns for each token
cols = st.columns(len(tokens))
for i, token in enumerate(tokens):
    if cols[i].button(token, key=f"token_{i}"):
        st.session_state.selected_token = i
        st.session_state.selected_position = i

        st.write(st.session_state.selected_position)
# # Display top predictions
st.write("### Most Likely Next Tokens")

k = st.slider("Top k", min_value=1, max_value=prediction_df.shape[0], value=10, step=1)
fig = px.bar(
    prediction_df.sort_values(by="probability", ascending=True).iloc[-k:],
    y="token",
    x="probability",
    text="probability",
    text_auto=".2f",  # type: ignore
    range_x=[0, 100],
    orientation="h",
)
st.plotly_chart(fig, use_container_width=True)

st.write("### Attribution")

col1, col2, col3 = st.columns(3)
pos_token_str = col1.selectbox(
    "Positive Predicted Token", prediction_df["token"].values
)
neg_token_str = col2.selectbox(
    "Negative Predicted Token", prediction_df["token"].values, index=1
)
attribution_token = col3.selectbox(
    "Position",
    options=token_df.unique_token.tolist(),
    index=st.session_state.selected_position,
)
attribution_token_idx = token_df[
    token_df["unique_token"] == attribution_token
].index.item()

pos_token = model.to_single_token(pos_token_str)
neg_token = model.to_single_token(neg_token_str)


def metric_fn(
    logits: torch.tensor,
    pos_token: torch.tensor = pos_token,
    neg_token: torch.Tensor = neg_token,
    position: int = attribution_token_idx,
) -> torch.Tensor:
    return logits[0, position, pos_token] - logits[0, position, neg_token]


attribution_output = calculate_feature_attribution(
    input=text,
    model=model,
    metric_fn=metric_fn,
    include_saes={sae.cfg.hook_name: sae},
    include_error_term=True,
    return_logits=True,
)

feature_attribution_df = attribution_output.sae_feature_attributions[sae.cfg.hook_name]
attribution_df_long = convert_sparse_feature_to_long_df(
    attribution_output.sae_feature_attributions[sae.cfg.hook_name][0]
)
# st.write(attribution_df_long)

attribution_df_long = attribution_df_long.sort_values(by="attribution", ascending=False)
attribution_df_long = attribution_df_long.iloc[:10]
attribution_df_long["feature"] = attribution_df_long["feature"].apply(
    lambda x: f"Feature {x}"
)
fig = px.bar(
    attribution_df_long.sort_values(by="attribution", ascending=True).iloc[:10],
    y="feature",
    x="attribution",
    orientation="h",
)
st.plotly_chart(fig, use_container_width=True)


def get_dashboard_html(neuronpedia_id: str, feature_idx: int) -> str:

    html_template = "https://neuronpedia.org/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=false&width=600&height=600"
    return html_template.format(neuronpedia_id, feature_idx)


tabs = st.tabs(attribution_df_long["feature"].values.tolist())
for i, feature in enumerate(attribution_df_long["feature"].values):
    with tabs[i]:
        feature_idx = int(feature.split(" ")[1])
        html = get_dashboard_html(
            neuronpedia_id=sae.cfg.neuronpedia_id,  # type: ignore
            feature_idx=feature_idx,  # type: ignore
        )
        iframe(html, width=600, height=600)
