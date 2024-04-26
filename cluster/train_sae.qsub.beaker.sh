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

# activate python environment
source /share/apps/source_files/cuda/cuda-11.8.source
source /share/apps/source_files/python/python-3.11.9.source

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/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
SLACK_USER_ID=U070NUR0S04

send_slack_notification() {
    local message=$1
    local url=$SLACK_WEBHOOK_URL
    local text="<@$SLACK_USER_ID> $message"
    curl -X POST -H 'Content-type: application/json' --data "{'text':'${text}'}" $url
}

send_slack_notification "Job $JOB_NAME:$JOB_ID started" 

# NOTE: Assume you have uploaded a singularity image
CONTAINER=$HOME/Scratch/sae_lens.sif
singularity exec --nv $CONTAINER --fakeroot python -m sae_lens.examples.train_sae --config.sae_class_name GatedSparseAutoencoder
singularity exec --nv $CONTAINER --fakeroot python -m sae_lens.examples.train_sae --config.sae_class_name SparseAutoencoder

send_slack_notification "Job $JOB_NAME:$JOB_ID ended"


                                                  
