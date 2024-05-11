### WIP - Very rough draft

This is an Ansible playbook that runs `Cache Activations` and and `Train SAE` in AWS.

Jobs are configured in the `configs` directory.

### Prerequisites
- AWS Account
- AWS ability to launch G instance types - you need to submit a request to enable this.
  - [Link to submit request for G. Click "Request increase at account level".](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-3819A6DF)
  - [Link to increase other quotas (like P instances)](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas)
  - G and P instances are not enabled by default [docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html)
  - What GPUs/specs are G and P instance types? [docs](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html)


### Local setup

#### Save AWS Credentials locally
1) Generate a set of AWS access keys 
   1) Sign into AWS
   2) [Click here to generate keys](
	https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-1#/security_credentials/access-key-wizard)

2) Save the following file into `~/.aws/credentials`, replacing the values with the ones you generated.
   - Don't change the region - keep it as `us-east-1`. Since all data transfer is in the same data center, it doesn't matter where you physically reside. If you change this, you will need to update `aws_ec2.yml` and the AMI ID in `cache_acts.yml`, and some services may not be available in other regions.

```
# ~/.aws/credentials
[default]
aws_access_key_id=AWS_ACCESS_KEY_ID
aws_secret_access_key=AWS_SECRET_ACCESS_KEY
region=us-east-1
```

#### Install Ansible

```
pip install ansible
ansible --version

ansible-galaxy collection install -r util/requirements.yml
```

#### Configure a Job to Run
```
cd scripts/ansible
cp -r configs_example configs
```
Modify `configs/shared.yml` and set `s3_bucket_name` to something unique. Bucket names are global so they must be unique (think of them as a username on a platform).

You don't need to modify anything else to run the example job.

This example job runs:
`cache_acts.yml` - Cache Activations with a `training_steps` of 2000.
`train_sae` - Contains `sweep.yml` which just has the name of the sweep, and two `jobs`, different only in the `l1_coefficient`.

#### Run the Example Job

You'll want to have the [AWS EC2 Console](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1) open so you can watch instances be launched, and terminate them manually in case it has any problems. By default if you exit Ansible prematurely it will not stop your EC2 instance, so you'll keep getting billed for it.

```
cd scripts/ansible
export WANDB_API_KEY=[WANDB API key here]
ansible-playbook run-configs.yml
```

Briefly, this will (time estimates are for this example):
1) Make your S3 bucket and other prerequisites. (~3 minutes)
2) Run the Cache Activations job (~10 minutes)
   1) Launch EC2 instance
   2) Run `util/cache_acts.py`, saving output to your S3 bucket
   3) Terminate the EC2 instance
3) Run the Train SAE jobs in parallel (~10 minutes)
   1) Launch EC2 instance
   2) Run `util/train_sae.py`, loading the cached activations from your S3 bucket.
   3) You can monitor progress of this by going to WANDB

It currently shows an error at the end but it should still output the artifacts to WANDB correctly.

### TODO
   - better integration with wandb ("sweep param")
    - use containers to simplify instance configuration
    - use 'typer' on `cache_acts.py` and `train_sae.py` 
    - split into different playbooks, roles
    - don't use 777 permissions
	- AWX server for GUI monitoring jobs
    - Automatically pull the latest AMI using Ansible