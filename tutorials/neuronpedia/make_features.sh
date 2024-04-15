#!/bin/bash

# we use a script around python to work around OOM issues - this ensures every batch gets the whole system available memory
# better fix is to investigate and fix the memory issues

echo "===== This will start a batch job that generates features to upload to Neuronpedia."
echo "===== This takes input of one SAE directory at a time."
echo "===== Features will be output into ./neuronpedia_outputs/{model}_{hook_point}_{d_sae}/batch-{batch_num}.json"

echo ""
echo "(Step 1 of 10)"
echo "What is the absolute, full local file path to your SAE's directory (with cfg.json, sae_weights.safetensors, sparsity.safetensors)?"
read saepath
# TODO: support huggingface directories

echo ""
echo "(Step 2 of 10)"
echo "What's the model ID? This must exactly match (including casing) the model ID you created on Neuronpedia."
read modelid

echo ""
echo "(Step 3 of 10)"
echo "What's the SAE ID?"
echo "This was set when you did 'Add SAEs' on Neuronpedia. This must exactly match that ID (including casing). It's in the format [abbrev hook name]-[abbrev author name], like res-jb."
read saeid

echo ""
echo "(Step 4 of 10)"
echo "How many features are in this SAE?"
read numfeatures

echo ""
echo "(Step 5 of 10)"
read -p "How many features do you want generate per batch file? More requires more RAM. (default: 128): " perbatch
[ -z "${perbatch}" ] && perbatch='128'

echo ""
echo "(Step 6 of 10)"
echo "For each activating text sequence, how many tokens to the LEFT of the top activating token do you want?"
echo "If your text sequences are 128 tokens long, then you might put 64. (default: 64)"
read leftbuffer
[ -z "${leftbuffer}" ] && leftbuffer='64'

echo ""
echo "(Step 7 of 10)"
echo "For each activating text sequence, how many tokens to the RIGHT of the top activating token do you want?"
echo "Left Buffer + Right Buffer must be < Total Text Length - 1"
echo "For example, text sequences of 128 can have at most buffers of 64 + 62 = 126"
echo "If your text sequences are 128 tokens long, then you might put 62. (default: 62)"
read rightbuffer
[ -z "${rightbuffer}" ] && rightbuffer='62'

echo ""
echo "(Step 8 of 10)"
read -p "Enter number of batches to sample from (default: 4096): " batches
[ -z "${batches}" ] && batches='4096'

echo ""
echo "(Step 9 of 10)"
read -p "Enter number of prompts to select from (default: 24576): " prompts
[ -z "${prompts}" ] && prompts='24576'

echo ""
numbatches=$(expr $numfeatures / $perbatch)
echo "===== INFO: We'll generate $numbatches batches of $perbatch features per batch = $numfeatures total features"

echo ""
echo "(Step 10 of 10)"
read -p "Do you want to resume from a specific batch number? Enter 1 to start from the beginning (default: 1): " startbatch
[ -z "${startbatch}" ] && startbatch='1'

endbatch=$(expr $numbatches)


echo ""
echo "===== Features will be output into [repo_dir]/neuronpedia_outputs/{modelId}_{saeId}_{hook_point}/batch-{batch_num}.json"
read -p "===== Hit ENTER to start!" start

for j in $(seq $startbatch $endbatch)
    do
    echo ""
    echo "===== BATCH: $j"
    echo "RUNNING: python make_batch.py $saepath $modelid $saeid $leftbuffer $rightbuffer $batches $prompts $perbatch $j $j"
    python make_batch.py $saepath $modelid $saeid $leftbuffer $rightbuffer $batches $prompts $perbatch $j $j
done

echo ""
echo "===== ALL DONE."
echo "===== Your features are under: [repo_dir]/neuronpedia_outputs/{model}_{hook_point}_{d_sae}"
echo "===== Use upload_features.sh to upload your features. Be sure to have the localhost server running first."