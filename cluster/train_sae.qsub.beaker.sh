# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=32G
#$ -l h_rt=48:0:0

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N train_sae
#$ -l gpu=true
#$ -l gpu_type=(a100|rtx8000|a100_80|a100_dgx)
#$ -pe gpu 1
#$ -o $HOME/logs
#$ -e $HOME/logs

# NOTE: Set env variables in this script
# e.g. export MYENV=MYVALUE
source $HOME/setup/env.sh
# NOTE: Install cluster utils here: https://github.com/90HH/cluster-utils/
source $HOME/setup/cluster-utils/import.sh

send_slack_notification "Job $JOB_NAME:$JOB_ID started" 

assert_env_var_set "SINGULARITYENV_WANDB_API_KEY"

# activate python environment
source /share/apps/source_files/cuda/cuda-11.8.source
source /share/apps/source_files/python/python-3.11.9.source

# NOTE: Assume you have uploaded a singularity image
CONTAINER=$HOME/Scratch/SAELens/cluster/sae_lens.sif
singularity exec --fakeroot --nv $CONTAINER python -m sae_lens.examples.train_sae --sae_class_name GatedSparseAutoencoder
singularity exec --fakeroot --nv $CONTAINER python -m sae_lens.examples.train_sae --sae_class_name SparseAutoencoder

send_slack_notification "Job $JOB_NAME:$JOB_ID ended"


                                                  
