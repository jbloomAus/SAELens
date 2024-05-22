This is an Ansible playbook that runs `Cache Activations` and and `Train SAE` in AWS.

- The playbook looks in the `configs` directory for what jobs to run, and runs them.
- It makes a copy of previously run jobs in the `jobs` directory.
- Check out the `configs_example` directory and read the comments in the YAML files.

### Prerequisites
- AWS Account
- AWS ability to launch G instance types - you need to submit a request to enable this.
  - [Submit request for G. Click "Request increase at account level".](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA)
  - [Increase other quotas (like P instances) - Be sure to request On-Demand, not Spot"](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas)
  - G and P instances are not enabled by default [docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html)
  - What GPUs/specs are G and P instance types? [docs](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html)
- Wandb Account (wandb.ai)

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

#### Load Wandb API Key automatically
1) Get your Wandb API key here: https://wandb.ai/settings#api
2) Add the following to your `~/.bash_profile` (or your equivalent shell defaults file)
```
export WANDB_API_KEY=[Paste Wandb API Key]
```

#### Install Ansible

```
pip install ansible
ansible --version

cd scripts/ansible
ansible-galaxy collection install -r util/requirements.yml
```

#### Configure a Job to Run
```
cd scripts/ansible
cp -r configs_example configs
```
Modify `configs/shared.yml` and set `s3_bucket_name` to something unique. Bucket names are global so they must be unique (think of them as a username on a platform).

You don't need to modify anything else to run the example job.

Explanation of the config files under `configs_example`:
- `shared.yml` - Shared values for all jobs.
- `cache_acts.yml` - Cache Activations with a `training_steps` of 2000.
- `train_sae` - Contains `sweep_common.yml` which has the name of the sweep, plus all of the common config values between the sweep's jobs. There are two jobs in the `jobs` subdirectory, which defines only the values that are different, which in this case is the `l1_coefficient`.
- It's only 2000 training steps for Cache Activations and 500 training steps for Train SAEs so the jobs themselves in the example are fast - most of the time is spend launching instances, configuring them, etc.

#### Run the Example Job

You should have the [AWS EC2 Console](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1) open so you can watch instances be launched, and terminate them manually in case it has any problems. By default if you exit Ansible prematurely it will not stop your EC2 instance, so you'll keep getting billed for it.

```
cd scripts/ansible
export WANDB_API_KEY=[WANDB API key here]
ansible-playbook run-configs.yml
```

Briefly, this example job will (time estimates are for the example above):
1) Create your S3 bucket and other prerequisites. (~3 minutes)
2) Run the Cache Activations job (~15 minutes)
   1) Launch EC2 instance
   2) Run `util/cache_acts.py`, saving output to your S3 bucket
   3) Terminate the EC2 instance
3) Run the Train SAE jobs in parallel (~15 minutes)
   1) Launch EC2 instance
   2) Run `util/train_sae.py`, loading the cached activations from your S3 bucket.
   3) You can monitor progress of this by going to WANDB, where it should also have your artifacts.


#### Run Cache Acts or Train SAEs Job Separately

Cache Activations only
```
ansible-playbook playbooks/setup.yml
ansible-playbook playbooks/cache_acts.yml
```

Train SAE only
```
ansible-playbook playbooks/setup.yml
ansible-playbook playbooks/train_sae.yml
```

#### Run Instance for Development

This brings up an instance with SAELens on it that has everything configured to run a job, including mounted S3. You will need to shut down the instance yourself when you're done with it.
1) Make sure you've copied the latest `scripts/ansible/configs-example` to `scripts/ansible/config`.
2) Modify `scripts/ansible/config/dev.yml` with the instance type you wish to launch, then save.

```
cd scripts/ansible
ansible-playbook run-dev.yml

# wait for run-dev.yml to complete (~7 minutes)

# get the IP address to ssh into
ansible-inventory --list --yaml tag_service__dev | grep public_ip_address

ssh -i ~/.ssh/saelens_ansible ubuntu@[PASTE_PUBLIC_IP_ADDRESS]
```

Once you're SSH'ed into the instance, the directories are:
```
# s3 mounted bucket directory, which should contain your bucket as its sole subdirectory
cd /mnt/s3
ls

# SAELens git repo - main branch
cd /home/ubuntu/SAELens
```

**Remember to terminate your instance when you're done.**
You can terminate the instance from the EC2 console. Alternatively, the instance has been configured to terminate on shutdown, so from SSH you can just run:
```
sudo shutdown -h now
```
You should still double check that the instance does indeed terminate from EC2 Console, just in case shutdown failed for some reason.

### TODO
   - make config scripts that makes the config sweep files automatically
   - should do async so that canceling ansible doesnt cancel the job
   - document how to monitor running jobs
   - better integration with wandb ("sweep param")
     - should we just use/repurpose wandb stuff instead of manually doing all this?
   - use containers, possilby cloudformation to simplify instance configuration
   - use 'typer' on `cache_acts.py` and `train_sae.py` 
   - ansible "best practices", better use of ansible features
   - don't use 777 permissions
	- AWX server for GUI monitoring jobs
   - Automatically pull the latest AMI using Ansible