#!/bin/bash

# # Set these variables
# LOCAL_FOLDER="OAI_GPT2Small_v5_32k_resid_delta_attn"
# REPO_NAME="GPT2-Small-OAI-v5-32k-attn-out-SAEs"
# USERNAME="jbloom"


# It's actually really easy to upload folders
huggingface-cli repo create GPT2-Small-OAI-v5-128k-attn-out-SAEs
cd OAI_GPT2Small_v5_128k_resid_delta_attn
huggingface-cli upload jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs .

huggingface-cli repo create GPT2-Small-OAI-v5-128k-resid-mid-SAEs
cd OAI_GPT2Small_v5_128k_resid_post_attn
huggingface-cli upload jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs .