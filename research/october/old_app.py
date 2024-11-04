import html
import json
import os
from copy import deepcopy
from datetime import datetime
from functools import partial

import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sae_lens.analysis.neuronpedia_integration import (
    get_neuronpedia_feature,
    get_neuronpedia_quick_list,
)

# set streamlit to wide mode
st.set_page_config(layout="wide")


torch.set_grad_enabled(False)


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


# Set up the Streamlit app
st.title("Gemma 2B Chat Interface")


# Initialize session state for the model
if "model" not in st.session_state:

    @st.cache_resource
    def load_model():
        return HookedTransformer.from_pretrained("gemma-2b-it", device=device)

    st.session_state.model = load_model()


if "sae" not in st.session_state:

    @st.cache_resource
    def load_sae():
        sae, _, _ = SAE.from_pretrained(
            "gemma-2b-it-res-jb",
            "blocks.12.hook_resid_post",
            device=device,
        )
        sae.fold_W_dec_norm()
        return sae

    st.session_state.sae = load_sae()


model = st.session_state.model
sae = st.session_state.sae


# Sidebar for sampling hyperparameters, refresh button, and debug mode
st.sidebar.header("Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
max_new_tokens = st.sidebar.slider("Max New Tokens", 10, 500, 10, 10)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.05)


# Debug mode checkbox
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
debug_prompt = "What's on your mind?"


# Tokenization visualization toggle
show_tokens = st.sidebar.checkbox("Show Tokenization", value=False)


# Steering options
st.sidebar.header("Steering")
enable_steering = st.sidebar.checkbox("Enable Steering", value=False)
steering_feature = st.sidebar.number_input(
    "Steering Feature", min_value=0, max_value=sae.cfg.d_sae - 1, value=0
)
steering_strength = st.sidebar.slider("Steering Strength", -10.0, 10.0, 1.0, 0.1)
steering_feature_data = get_neuronpedia_feature(
    steering_feature, layer=12, model="gemma-2b-it", dataset="res-jb"
)
steering_feature_max_act = steering_feature_data["maxActApprox"]


# Refresh button
if st.sidebar.button("Refresh Conversation"):
    st.session_state.messages = []
    st.session_state.debug_initialized = False
    if "selected_token" in st.session_state:
        del st.session_state.selected_token
    st.rerun()


# Save conversation button
if st.sidebar.button("Save Conversation"):
    # Ensure the history directory exists
    os.makedirs("./history", exist_ok=True)

    # Ensure the messages are JSON-serializable
    messages = deepcopy(st.session_state.messages)
    for message in messages:
        message.pop("sae_activations")
        message.pop("tokens")

    # Prepare the data to save
    save_data = {
        "messages": messages,
        "config": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "debug_mode": debug_mode,
            "enable_steering": enable_steering,
            "steering_feature": steering_feature,
            "steering_strength": steering_strength,
        },
    }

    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./history/conversation_{timestamp}.json"

    # Save the data to a JSON file
    with open(filename, "w") as f:
        json.dump(save_data, f, indent=2)

    st.sidebar.success(f"Conversation saved to {filename}")


# Initialize chat history and starter prompt state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_initialized" not in st.session_state:
    st.session_state.debug_initialized = False


# Steering function
def steering(
    activations,
    hook,
    steering_strength=1.0,
    steering_vector=None,
    max_act=steering_feature_max_act,
):
    activations = activations + max_act * steering_strength * steering_vector
    return activations


# Function to generate response and store SAE activations
def generate_response(messages):
    full_prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = model.to_tokens(full_prompt, prepend_bos=False)
    input_length = input_ids.shape[1]

    if enable_steering:
        steering_vector = sae.W_dec[steering_feature].to(device)
        with model.hooks(
            fwd_hooks=[
                (
                    sae.cfg.hook_name,
                    partial(
                        steering,
                        steering_vector=steering_vector,
                        steering_strength=steering_strength,
                    ),
                )
            ]
        ):
            response = model.generate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                stop_at_eos=False,
            )
    else:
        response = model.generate(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            stop_at_eos=False,
        )

    eos_token_id = model.tokenizer.eos_token_id
    response_list = response.tolist()[0]

    if eos_token_id in response_list[input_length:]:
        eos_token_position = response_list.index(eos_token_id, input_length)
        response = response[:, : eos_token_position + 1]
    else:
        response = response[:, : input_length + max_new_tokens]

    _, cache = model.run_with_cache(response)
    sae_activations = sae.encode(cache[sae.cfg.hook_name])

    new_tokens = response[:, (input_length - 1) :]
    new_activations = sae_activations[:, (input_length - 1) :]
    assistant_response = model.tokenizer.decode(new_tokens[0]).strip()

    if assistant_response.endswith("<eos>"):
        assistant_response = assistant_response[:-5].strip()

    return assistant_response, new_activations


# Function to process a prompt
def process_prompt(prompt, edit_index=None):

    full_prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    _, cache = model.run_with_cache(full_prompt, prepend_bos=False)
    sae_activations = sae.encode(cache[sae.cfg.hook_name])

    if edit_index is not None:
        st.session_state.messages[edit_index] = {
            "role": "user",
            "content": prompt,
            "tokens": model.to_tokens(full_prompt, prepend_bos=False),
            "sae_activations": sae_activations,
        }
        st.session_state.messages = st.session_state.messages[: edit_index + 1]
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "tokens": model.to_tokens(full_prompt, prepend_bos=False),
                "sae_activations": sae_activations,
            }
        )

    response, sae_activations = generate_response(st.session_state.messages)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sae_activations": sae_activations,
            "tokens": model.to_tokens(response, prepend_bos=False),
        }
    )


# Function to display message with optional tokenization and token selection
def display_message(message, role, index):
    with st.chat_message(role):
        if show_tokens:
            tokens = model.to_str_tokens(message["tokens"])

            # # for each token get the length in pixels
            # token_lengths = [len(token) for token in tokens]
            # # get the total length of the tokens
            # total_length = sum(token_lengths)
            # # get the proportion of the total length for each token
            # token_proportions = [length / total_length for length in token_lengths]

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

            cols = st.columns(len(tokens))
            for i, token in enumerate(tokens):
                if cols[i].button(token, key=f"{role}_{index}_{i}"):
                    st.write(index, i)
                    st.session_state.selected_token = (index, i)
        else:
            st.markdown(message["content"])

        if role == "user":
            if st.button("✏️", key=f"edit_{index}", help="Edit and regenerate"):
                st.session_state.editing = index


# Function to plot feature activations
def plot_feature_activations(activations):
    fig = px.line(activations.cpu().numpy().T, height=200)
    st.plotly_chart(fig)


# Function to display top features and Neuronpedia quicklist
def display_top_features(activations):
    vals, indices = torch.topk(activations, 10)
    # st.write("Top 10 Features:")
    # for val, idx in zip(vals.tolist(), indices.tolist()):
    #     st.write(f"Feature {idx}: {val:.4f}")

    if st.button("Open feature list on neuronpedia"):
        get_neuronpedia_quick_list(
            indices.tolist(), layer=12, model="gemma-2b-it", dataset="res-jb"
        )

    feature_tabs = st.tabs(
        [f"Feature_{indices}: {vals:.4f}" for vals, indices in zip(vals, indices)]
    )
    for i, tab in enumerate(feature_tabs):
        with tab:
            st.write(f"Feature {indices[i]}: {vals[i]:.4f}")
            # embed iframe
            iframe_template = (
                """https://neuronpedia.org/gemma-2b-it/12-res-jb/{}?embed=true"""
            )
            components.iframe(
                iframe_template.format(indices[i]),
                width=1100,
                height=1200,
                scrolling=False,
            )

    # for item in quicklist:
    #     st.write(f"[Feature {item['id']}]({item['url']}): {item['label']}")


# Display Neuronpedia iframe for steering feature
if enable_steering:
    st.sidebar.header("Steering Feature Info")
    with st.sidebar:
        iframe_template = "https://neuronpedia.org/gemma-2b-it/12-res-jb/{}?embed=true"
        components.iframe(
            iframe_template.format(steering_feature),
            width=300,
            height=700,
            scrolling=True,
        )


# Display chat messages and handle token selection
for i, message in enumerate(st.session_state.messages):
    display_message(message, message["role"], i)


# Handle token selection and feature analysis
if "selected_token" in st.session_state:
    message_idx, token_idx = st.session_state.selected_token
    selected_message = st.session_state.messages[message_idx]

    if "sae_activations" in selected_message:
        with st.expander(
            f"Analyzing features for token: {model.to_str_tokens(selected_message['tokens'])[token_idx]}"
        ):

            activations = selected_message["sae_activations"][0, token_idx]
            plot_feature_activations(activations)
            display_top_features(activations)


# Handle editing
if "editing" in st.session_state:
    edit_index = st.session_state.editing
    edited_prompt = st.text_input(
        "Edit your prompt:",
        st.session_state.messages[edit_index]["content"],
        key=f"edit_input_{edit_index}",
    )
    if st.button("Regenerate", key=f"regenerate_{edit_index}"):
        process_prompt(edited_prompt, edit_index=edit_index)
        del st.session_state.editing
        st.rerun()


# Starter prompts
starter_prompts = [
    "Tell me about the latest advancements in AI",
    "What's on your mind?",
    "Explain the concept of quantum computing",
    "How can we address climate change?",
]


# Display starter prompt buttons if no prompt has been selected and not in debug mode
if not st.session_state.messages and not debug_mode:
    st.write("Choose a starter prompt or type your own:")
    cols = st.columns(2)
    for i, prompt in enumerate(starter_prompts):
        if cols[i % 2].button(prompt):
            process_prompt(prompt)
            st.rerun()


# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    process_prompt(prompt)
    st.rerun()
