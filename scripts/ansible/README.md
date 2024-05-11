### WIP - Very rough draft

This is an Ansible playbook that runs `caching_replication_how_train_saes.py` and `replication_how_train_saes.py` in AWS.

Jobs are configured by the `cache_acts.yml` file.

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
cp examples/config.cache_acts.yml config.cache_acts.yml
```
Modify `config_cache_acts.yml` for your job. Why is this YAML and not JSON? Because YAML supports comments.

#### Run the Job

You'll want to have the [AWS EC2 Console](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1) open so you can watch the instance be launched, terminate it manually in case it has any problems. By default if you kill/exit Ansible prematurely we will not stop your EC2 instance, so you'll keep getting billed for it.


__Cache Activations__

```
cd scripts/ansible
ansible-playbook playbookvar.yml --tags cache_acts
```

#### See All Supported Commands ("tags")

```
cd scripts/ansible
ansible-playbook playbook.yml -i aws_ec2.yml --list-tags
```
Then run specific tags with the --tags flag

### TODO
    - use containers to simplify instance configuration
    - use 'typer' on `cache_acts.py` and `train_sae.py` 
    - split into different playbooks, roles
    - don't use 777 permissions
	- AWX server for GUI monitoring jobs
    - Automatically pull the latest AMI using Ansible