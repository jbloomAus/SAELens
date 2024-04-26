#!/bin/bash
#$ -N train_gated_sae             # Specify the job name
#$ -l h_rt=72:00:00
#$ -l h_vmem=64G           # Request 64GB of memory per job
#$ -l gpu=1               # Request 1 GPU
#$ -ac allow=L        # Specify the type of GPU
#$ -o $HOME/logs
#$ -e $HOME/logs

# Add locally installed executables to PATH
source /home/$USER/.bash_profile

# Start the job
# NOTE: Assume you have imported cluster utils
# See: https://github.com/90HH/cluster-utils/tree/main
send_slack_notification "Job $JOB_NAME:$JOB_ID started"

# NOTE: Assume you have already cloned the repository and checked out correct branch
# Navigate to the project directory
cd /home/$USER/Scratch/SAELens 

# NOTE: Assume you have already installed pdm
poetry install

# Run the script
poetry run python -m sae_lens.examples.train_sae --config.sae_class_name GatedSparseAutoencoder
poetry run python -m sae_lens.examples.train_sae --config.sae_class_name SparseAutoencoder

# End the job
send_slack_notification "Job $JOB_NAME:$JOB_ID ended"
