# CHANGELOG



## v3.2.3 (2024-06-05)

### Fix

* fix: allow tutorial packages for colab install to use latest version (#173)

fix: allow tutorial packages for colab install to use latest version (#173) ([`f73cb73`](https://github.com/jbloomAus/SAELens/commit/f73cb73072e006f6dda0984e05e8accda014a2d8))

### Unknown

* fix pip install in HookedSAETransformer Demo (#172) ([`5d0faed`](https://github.com/jbloomAus/SAELens/commit/5d0faedf868d6e68035527a52ba718807e8ed196))


## v3.2.2 (2024-06-02)

### Fix

* fix: removing truncation in activations store data loading (#62) ([`43c93e2`](https://github.com/jbloomAus/SAELens/commit/43c93e2c3e19e9e9d81b48d3a472d894dac83d65))


## v3.2.1 (2024-06-02)

### Fix

* fix: moving non-essential deps to dev (#121) ([`1a2cde0`](https://github.com/jbloomAus/SAELens/commit/1a2cde04d306061e7765a61e44d77ad1f3f8a1d4))


## v3.2.0 (2024-05-30)

### Feature

* feat: activation norm scaling factor folding (#170)

* feat: add convenience function for folding scaling factor

* keep playing around with benchmark ([`773e308`](https://github.com/jbloomAus/SAELens/commit/773e30898bb0957d4c9bd79343776ff3e225d13c))


## v3.1.1 (2024-05-29)

### Fix

* fix: share config defaulting between hf and local loading (#169) ([`7df479c`](https://github.com/jbloomAus/SAELens/commit/7df479c3a1e4d2ed187090ef65c5059f6fa8bf24))


## v3.1.0 (2024-05-29)

### Feature

* feat: add w_dec_norm folding (#167)

* feat: add w_dec_norm folding

* format ([`f1908a3`](https://github.com/jbloomAus/SAELens/commit/f1908a39b3d78a03288ca721aa18fc5bfbf9b83e))

### Unknown

* Fixed typo in Hooked_SAE_Transformer_Demo.ipynb preventing Open in Colab badge from working (#166)

Minor typo in file name was preventing Hooked_SAE_Transformer_Demo.ipynb &#34;Open in Colab&#34; badge from working. ([`4850b16`](https://github.com/jbloomAus/SAELens/commit/4850b16a17c08ef39a8df74a5f4df5074395474b))

* Fix hook z training reshape bug (#165)

* remove file duplicate

* fix: hook-z evals working, and reshaping mode more explicit ([`0550ae3`](https://github.com/jbloomAus/SAELens/commit/0550ae3defe778e08a050faff5e1345aee6de1a4))


## v3.0.0 (2024-05-28)

### Breaking

* feat: refactor SAE code

BREAKING CHANGE: renamed and re-implemented paths ([`3c67666`](https://github.com/jbloomAus/SAELens/commit/3c6766604f5b2079206e0c073e75a72c67a76f43))

### Unknown

* major: trigger release

BREAKING CHANGE: https://python-semantic-release.readthedocs.io/en/latest/commit-parsing.html#commit-parser-angular

BREAKING CHANGE: ([`fac8533`](https://github.com/jbloomAus/SAELens/commit/fac8533be338dcacbae0045ab0d7a7396c630aa8))

* major: trigger release

BREAKING CHANGE: trigger release (apparently we need a newline) ([`90ed2c2`](https://github.com/jbloomAus/SAELens/commit/90ed2c296fb65b6e1935435690d8cddb007ce04b))

* BREAKING CHANGE: Quality of Life Refactor of SAE Lens adding SAE Analysis with HookedSAETransformer and some other breaking changes. (#162)

* move HookedSAETransformer from TL

* add tests

* move runners one level up

* fix docs name

* trainer clean up

* create training sae, not fully seperate yet

* remove accidentally commited notebook

* commit working code in the middle of refactor, more work to do

* don&#39;t use act layers plural

* make tutorial not use the activation store

* moved this file

* move import of toy model runner

* saes need to store at least enough information to run them

* further refactor and add tests

* finish act store device rebase

* fix config type not caught by test

* partial progress, not yet handling error term for hooked sae transformer

* bring tests in line with trainer doing more work

* revert some of the simplification to preserve various features, ghost grads, noising

* hooked sae transformer is working

* homogenize configs

* re-enable sae compilation

* remove old file that doesn&#39;t belong

* include normalize activations in base sae config

* make sure tutorial works

* don&#39;t forget to update pbar

* rename sparse autoencoder to sae for brevity

* move non-training specific modules out of training

* rename to remove _point

* first steps towards better docs

* final cleanup

* have ci use same test coverage total as make check-ci

* clean up docs a bit

---------

Co-authored-by: ckkissane &lt;67170576+ckkissane@users.noreply.github.com&gt; ([`e4eaccc`](https://github.com/jbloomAus/SAELens/commit/e4eaccc87b277a42d463624656a3548ead0db359))

* Move activation store to cpu (#159)

* add act store device to config

* fix serialisation issue with device

* fix accidental hardcoding of a device

* test activations get moved correctly

* fix issue with test cacher that shared state

* add split store &amp; model test + fix failure

* clarify comment

* formatting fixes ([`eb9489a`](https://github.com/jbloomAus/SAELens/commit/eb9489a2dd11fe4841857309dcc369e98a6b9360))

* Refactor training (#158)

* turn training runner into a class

* make a trainer class

* further refactor

* update runner call

* update docs ([`72179c8`](https://github.com/jbloomAus/SAELens/commit/72179c8336fcdb5e159ddca930af99700362e377))

* Enable autocast for LM activation creation (#157)

* add LM autocasting

* add script to test autocast performance

* format fix

* update autocast demo script ([`cf94845`](https://github.com/jbloomAus/SAELens/commit/cf94845129f0e2d0bbe5135d90797a03611e983c))

* gemma 2b sae resid post 12. fix ghost grad print ([`2a676b2`](https://github.com/jbloomAus/SAELens/commit/2a676b210832e789dbb80f33b2d8f747a7209e0f))

* don&#39;t hardcode hook ([`a10283d`](https://github.com/jbloomAus/SAELens/commit/a10283de5b402cbac9c2afbd6263b9e5798f9e1c))

* add mlp out SAEs to from pretrained ([`ee9291e`](https://github.com/jbloomAus/SAELens/commit/ee9291eae91908c398377b199bb9e3b33a5a2622))

* remove resuming ability, keep resume config but complain if true (#156) ([`64e4dcd`](https://github.com/jbloomAus/SAELens/commit/64e4dcd3fe142cee751a348f9ed581edf2a6e3f0))

* Add notebook to transfer W&amp;B models to HF (#154)

hard to check this works quickly but assuming it does. ([`91239c1`](https://github.com/jbloomAus/SAELens/commit/91239c1c6e0abd06aea3aa7669fc5b56adc6e792))

* Remove sae parallel training, simplify code (#155)

* remove sae parallel training, simplify code
* remove unused import
* remove accidental inclusion of file

(not tagging this as breaking since we&#39;re do a new major release this week and I don&#39;t want to keep bumping the major version) ([`f445fdf`](https://github.com/jbloomAus/SAELens/commit/f445fdfc823cb6be8b1910a28a89c8bd20661be8))

* Update pretrained_saes.yaml ([`37fb150`](https://github.com/jbloomAus/SAELens/commit/37fb15083a5427894b65de9654272e99291ce46a))

* Ansible: update incorrect EC2 quota request link ([`432c7e1`](https://github.com/jbloomAus/SAELens/commit/432c7e1fd7e0ea64fcba5941e36d9740c2c58a07))

* Merge pull request #153 from jbloomAus/ansible_dev

Ansible: dev only mode ([`51d2175`](https://github.com/jbloomAus/SAELens/commit/51d2175d05ce99a83da7b29210e725163a578c1a))

* Ansible: dev only mode ([`027460f`](https://github.com/jbloomAus/SAELens/commit/027460f48819f7953754406f5cc0499a08ed4ebc))

* feature: add gemma-2b bootleg saes (#152) ([`b9b7e32`](https://github.com/jbloomAus/SAELens/commit/b9b7e32562a1c48003671464f0ed5084d3541e97))


## v2.1.3 (2024-05-15)

### Fix

* fix: Fix normalisation (#150)

* fix GPT2 sweep settings to use correct dataset

* add gpt2 small block sweep to check norm

* larger buffer + more evals

* fix activation rescaling so normalisation works

* formatting fixes ([`9ce0fe4`](https://github.com/jbloomAus/SAELens/commit/9ce0fe4747ad31be5570baa7cf31714374c98e10))

### Unknown

* Fix checkpointing of training state that includes a compiled SAE (#143)

* Adds state_dict to L1Scheduler

* investigating test failure

* fix: Fix issues with resumption testing (#144)

* fix always-true comparison in train context testing

* set default warmup steps to zero

* remove unused type attribute from L1Scheduler

* update training tests to use real context builder

* add docstring for build_train_ctx

* 2.1.2

Automatically generated by python-semantic-release

* Adds state_dict to L1Scheduler

* investigating test failure

---------

Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`2f8c4e1`](https://github.com/jbloomAus/SAELens/commit/2f8c4e17316658b14dd3bc9d1f7e50cea36b0db4))

* fix GPT2 sweep settings to use correct dataset (#147)

* fix GPT2 sweep settings to use correct dataset

* add gpt2 small block sweep to check norm

* larger buffer + more evals

---------

Co-authored-by: Joseph Bloom &lt;69127271+jbloomAus@users.noreply.github.com&gt; ([`448d911`](https://github.com/jbloomAus/SAELens/commit/448d911e803aaa051d70e5f532933f71dcb72be8))

* Pretokenize runner (#148)

* feat: adding a pretokenize runner

* rewriting pretokenization based on feedback ([`f864178`](https://github.com/jbloomAus/SAELens/commit/f8641783e48cc01f7184b1c91ddc39994afd4f4b))

* Fix config files for Ansible ([`ec70cea`](https://github.com/jbloomAus/SAELens/commit/ec70cea88fbad149f45472dae7bfe7be56351b60))

* Pin Ansible config example to a specific version, update docs (#142)

* Pin Ansible config example to a specific version, update docs

* Allow running cache acts or train sae separately. Update README

* Update readme ([`41785ae`](https://github.com/jbloomAus/SAELens/commit/41785ae31dc826ac99a142eca65b05d0e57b5ce1))


## v2.1.2 (2024-05-13)

### Fix

* fix: Fix issues with resumption testing (#144)

* fix always-true comparison in train context testing

* set default warmup steps to zero

* remove unused type attribute from L1Scheduler

* update training tests to use real context builder

* add docstring for build_train_ctx ([`085d04f`](https://github.com/jbloomAus/SAELens/commit/085d04f7e57e3819810b18e12b011adc8c7f2ba1))


## v2.1.1 (2024-05-13)

### Fix

* fix: hardcoded mps device in ckrk attn saes (#141) ([`eba3f4e`](https://github.com/jbloomAus/SAELens/commit/eba3f4e54ad6a02553f0ed2b575b0547df68a200))

### Unknown

* feature: run saelens on AWS with one command (#138)

* Ansible playbook for automating caching activations and training saes

* Add automation

* Fix example config

* Fix bugs with ansible mounting s3

* Reorg, more automation, Ubuntu instead of Amazon Linux

* More automation

* Train SAE automation

* Train SAEs and readme

* fix gitignore

* Fix automation config bugs, clean up paths

* Fix shutdown time, logs ([`13de52a`](https://github.com/jbloomAus/SAELens/commit/13de52a5e12fd275ca8601aa22fd5ec66a5c6e9a))

* Gpt 2 sweep (#140)

* sweep settings for gpt2-small

* get model string right

* fix some comments that don&#39;t apply now

* formatting fix ([`4cb270b`](https://github.com/jbloomAus/SAELens/commit/4cb270b7680585c5758910dfeafa727185ac88b9))

* Remove cuda cache emptying in evals.py (#139) ([`bdef2cf`](https://github.com/jbloomAus/SAELens/commit/bdef2cf0e4ad3e3070f9be1c46a1adf13094eb13))


## v2.1.0 (2024-05-11)

### Chore

* chore: remove use_deterministic_algorithms=True since it causes cuda errors (#137) ([`1a3bedb`](https://github.com/jbloomAus/SAELens/commit/1a3bedbc3192ca919fc0716ce52d06f060ad2550))

### Feature

* feat: Hooked toy model (#134)

* adds initial re-implementation of toy models

* removes instance dimension from toy models

* fixing up minor nits and adding more tests

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`03aa25c`](https://github.com/jbloomAus/SAELens/commit/03aa25c6e8589c1eb9a6b94911e6d77187d6bef7))


## v2.0.0 (2024-05-10)

### Breaking

* feat: rename batch sizes to give informative units (#133)

BREAKING CHANGE: renamed batch sizing config params

* renaming batch sizes to give units

* changes in notebooks

* missed one!

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`cc78e27`](https://github.com/jbloomAus/SAELens/commit/cc78e277ae639f57df389e61899278919c16d993))

### Chore

* chore: tools to make tests more deterministic (#132) ([`2071d09`](https://github.com/jbloomAus/SAELens/commit/2071d096c46b0b532e2d99381b300c3c64071747))

* chore: Make tutorial notebooks work in Google Colab (#120)

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`007141e`](https://github.com/jbloomAus/SAELens/commit/007141e67a447f8dfe8d6797a6fba0fc4fce61bd))


## v1.8.0 (2024-05-09)

### Chore

* chore: closing &#34; in docs (#130) ([`5154d29`](https://github.com/jbloomAus/SAELens/commit/5154d29498a480c4e7ddc9edc9effd30cecbeda7))

### Feature

* feat: Add model_from_pretrained_kwargs as config parameter (#122)

* add model_from_pretrained_kwargs config parameter to allow full control over model used to extract activations from. Update tests to cover new cases

* tweaking test style

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`094b1e8`](https://github.com/jbloomAus/SAELens/commit/094b1e8250c1ae6cd3fc8336e09075d61ce967d1))


## v1.7.0 (2024-05-08)

### Feature

* feat: Add torch compile (#129)

* Surface # of eval batches and # of eval sequences

* fix formatting

* config changes

* add compilation to lm_runner.py

* remove accidental print statement

* formatting fix ([`5c41336`](https://github.com/jbloomAus/SAELens/commit/5c41336853beac6bbe2105fefc55c746e3e2e61f))

* feat: Change eval batch size (#128)

* Surface # of eval batches and # of eval sequences

* fix formatting

* fix print statement accidentally left in ([`758a50b`](https://github.com/jbloomAus/SAELens/commit/758a50b073777028cd0dabcc50049798c2fcd68f))


## v1.6.1 (2024-05-07)

### Fix

* fix: Revert &#34;feat: Add kl eval (#124)&#34; (#127)

This reverts commit c1d9cbe8627f27f4d5384ed4c9438c3ad350d412. ([`1a0619c`](https://github.com/jbloomAus/SAELens/commit/1a0619ccb758a0a8b7130fe163cbfb06bf4bc7cc))


## v1.6.0 (2024-05-07)

### Feature

* feat: Add bf16 autocast (#126)

* add bf16 autocast and gradient scaling

* simplify autocast setup

* remove completed TODO

* add autocast dtype selection (generally keep bf16)

* formatting fix

* remove autocast dtype ([`8e28bfb`](https://github.com/jbloomAus/SAELens/commit/8e28bfb6ddded2e006f38a18ca0603627ed32ae2))


## v1.5.0 (2024-05-07)

### Feature

* feat: Add kl eval (#124)

* add kl divergence to evals.py

* fix linter ([`c1d9cbe`](https://github.com/jbloomAus/SAELens/commit/c1d9cbe8627f27f4d5384ed4c9438c3ad350d412))

### Unknown

* major: How we train saes replication (#123)

* l1 scheduler, clip grad norm

* add provisional ability to normalize activations

* notebook

* change heuristic norm init to constant, report b_e and W_dec norms (fix tests later)

* fix mse calculation

* add benchmark test

* update heuristic init to 0.1

* make tests pass device issue

* continue rebase

* use better args in benchmark

* remove stack in get activations

* broken! improve CA runner

* get cache activation runner working and add some tests

* add training steps to path

* avoid ghost grad tensor casting

* enable download of full dataset if desired

* add benchmark for cache activation runner

* add updated tutorial

* format

---------

Co-authored-by: Johnny Lin &lt;hijohnnylin@gmail.com&gt; ([`5f46329`](https://github.com/jbloomAus/SAELens/commit/5f46329d1df90e374d44729966e57542c435d6cf))


## v1.4.0 (2024-05-05)

### Feature

* feat: Store state to allow resuming a run (#106)

* first pass of saving

* added runner resume code

* added auto detect most recent checkpoint code

* make linter happy (and one small bug)

* blak code formatting

* isort

* help pyright

* black reformatting:

* activations store flake

* pyright typing

* black code formatting

* added test for saving and loading

* bigger training set

* black code

* move to pickle

* use pickle because safetensors doesn&#39;t support all the stuff needed for optimizer and scheduler state

* added resume test

* added wandb_id for resuming

* use wandb id for checkpoint

* moved loaded to device and minor fixes to resuming

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`4d12e7a`](https://github.com/jbloomAus/SAELens/commit/4d12e7a4e42079442922ccad7e22e9aca279b6aa))

### Unknown

* Fix: sparsity norm calculated at incorrect dimension. (#119)

* Fix: sparsity norm calculated at incorrect dimension.

For L1 this does not effect anything as essentially it&#39;s calculating the abs() and average everything. For L2 this is problematic as L2 involves sum and sqrt. Unexpected behaviors occur when x is of shape (batch, sen_length, hidden_dim).

* Added tests.

* Changed sparsity calculation to handle 3d inputs. ([`ce95fb2`](https://github.com/jbloomAus/SAELens/commit/ce95fb200e67fab6f9dc3cd24ac6078c9a4b1050))


## v1.3.0 (2024-05-03)

### Feature

* feat: add activation bins for neuronpedia outputs, and allow customizing quantiles (#113) ([`05d650d`](https://github.com/jbloomAus/SAELens/commit/05d650d8ece48bee64077ac075b18a1efae206d4))

* feat: Update for Neuropedia auto-interp (#112)

* cleanup Neuronpedia autointerp code

* Fix logic bug with OpenAI key

---------

Co-authored-by: Joseph Bloom &lt;69127271+jbloomAus@users.noreply.github.com&gt; ([`033283d`](https://github.com/jbloomAus/SAELens/commit/033283d99447a0949c1241adc2c99ed31342650f))

* feat: SparseAutoencoder.from_pretrained() similar to transformer lens (#111)

* add partial work so David can continue

* feat: adding a SparseAutoencoder.from_pretrained() function

---------

Co-authored-by: jbloomaus &lt;jbloomaus@gmail.com&gt; ([`617d416`](https://github.com/jbloomAus/SAELens/commit/617d416f34101047c6bceb31fdbd325fb9ff7c84))

### Fix

* fix: replace list_files_info with list_repo_tree (#117) ([`676062c`](https://github.com/jbloomAus/SAELens/commit/676062cae4cd72e198e87b14fcae124ec2c534ca))

* fix: Improved activation initialization, fix using argument to pass in API key (#116) ([`7047bcc`](https://github.com/jbloomAus/SAELens/commit/7047bcc6d5d8d15b990605a7c2f68a210db603d0))


## v1.2.0 (2024-04-29)

### Feature

* feat: breaks up SAE.forward() into encode() and decode() (#107)

* breaks up SAE.forward() into encode() and decode()

* cleans up return typing of encode by splitting into a hidden and public function ([`7b4311b`](https://github.com/jbloomAus/SAELens/commit/7b4311bab965775bbb37e5d1f7b27d1379954fa8))


## v1.1.0 (2024-04-29)

### Feature

* feat: API for generating autointerp + scoring for neuronpedia (#108)

* API for generating autointerp for neuronpedia

* Undo pytest vscode setting change

* Fix autointerp import

* Use pypi import for automated-interpretability ([`7c43c4c`](https://github.com/jbloomAus/SAELens/commit/7c43c4caa84aea421ac81ae0e326d9c62bb17bec))


## v1.0.0 (2024-04-27)

### Breaking

* chore: empty commit to bump release

BREAKING CHANGE: v1 release ([`2615a3e`](https://github.com/jbloomAus/SAELens/commit/2615a3ec472db25678971c4a11b804e316daa8a5))

### Chore

* chore: fix outdated lr_scheduler_name in docs (#109)

* chore: fix outdated lr_scheduler_name in docs

* add tutorial hparams ([`7cba332`](https://github.com/jbloomAus/SAELens/commit/7cba332800ff6aa826aaf53b1c86a56afabde6ec))

### Unknown

* BREAKING CHANGE: 1.0.0 release

BREAKING CHANGE: 1.0.0 release ([`c23098f`](https://github.com/jbloomAus/SAELens/commit/c23098f17615cd092d82ee12b2e61edc93dbb1ec))

* Neuronpedia: allow resuming upload (#102) ([`0184671`](https://github.com/jbloomAus/SAELens/commit/0184671bafa2ff53e97e6ff6e157a334df2428b9))


## v0.7.0 (2024-04-24)

### Feature

* feat: make a neuronpedia list with features via api call (#101) ([`23e680d`](https://github.com/jbloomAus/SAELens/commit/23e680d2ae03b4c375651e14feb247d3eb29e516))

### Unknown

* Merge pull request #100 from jbloomAus/np_improvements

Improvements to Neuronpedia Runner ([`5118f7f`](https://github.com/jbloomAus/SAELens/commit/5118f7f7019cb3b33a26f0acc8fca55e2074202b))

* neuronpedia: save run settings to json file to avoid errors when resuming later. automatically skip batch files that already exist ([`4b5412b`](https://github.com/jbloomAus/SAELens/commit/4b5412b4c351156f20b965a4675b5448781dc951))

* skip batch file if it already exists ([`7d0e396`](https://github.com/jbloomAus/SAELens/commit/7d0e3961903fbc559bc6f7e92b497d87b5a34244))

* neuronpedia: include log sparsity threshold in skipped_indexes.json ([`5c967e7`](https://github.com/jbloomAus/SAELens/commit/5c967e7e83e27dd10bb5fbf5ba4a1291af41a16b))


## v0.6.0 (2024-04-21)

### Chore

* chore: enabling pythong 3.12 checks for CI ([`25526ea`](https://github.com/jbloomAus/SAELens/commit/25526ea2f72aeee77daad677cfd3555fe48d9e88))

* chore: setting up precommit to be consistent with CI ([`18e706d`](https://github.com/jbloomAus/SAELens/commit/18e706dad75f3f9bddf388f5f5c32328669749fe))

### Feature

* feat: Added `tanh-relu` activation fn and input noise options  (#77)

* Still need to pip-install from GitHub hufy implementation.

* Added support for `tanh_sae`.

* Added notebook for loading the `tanh_sae`

* tweaking config options to be more declarating / composable

* testing adding noise to SAE forward pass

* updating notebook

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`551e94d`](https://github.com/jbloomAus/SAELens/commit/551e94de83797791e10d9c67e328031e207718c5))

### Unknown

* Update proposal.md ([`6d45b33`](https://github.com/jbloomAus/SAELens/commit/6d45b33423fb308343c0b5bb0a49b12322b8c620))

* Merge pull request #96 from jbloomAus/github-templates

add templates for PR&#39;s / issues ([`241a201`](https://github.com/jbloomAus/SAELens/commit/241a20195d58fbc03f33218458d43e5cbedf1745))

* add templates for PR&#39;s / issues ([`74ff597`](https://github.com/jbloomAus/SAELens/commit/74ff5979e9d64b7af9fc4e0978da37857c6c8dcf))

* Merge pull request #95 from jbloomAus/load-state-dict-not-strict

Make load_state_dict use strict=False ([`4a9e274`](https://github.com/jbloomAus/SAELens/commit/4a9e2748ce34510b92328606fd5351851472eb48))

* fix accidental bug ([`c22fbbd`](https://github.com/jbloomAus/SAELens/commit/c22fbbdebc31c42cc579a1ff6433cfc73a8aeb2b))

* fix load pretrained legacy with state dict change ([`b5e97f8`](https://github.com/jbloomAus/SAELens/commit/b5e97f8b343be5c2bed36914741df9dad96361b7))

* Make load_state_dict use strict=False ([`fdf7fe9`](https://github.com/jbloomAus/SAELens/commit/fdf7fe951bf8275b7b9bee62e7c06b83570f2798))

* Merge pull request #94 from jbloomAus/update-pre-commit

chore: setting up precommit to be consistent with CI ([`6a056b7`](https://github.com/jbloomAus/SAELens/commit/6a056b7b6914f6bf6dfca5eb006f3114d4d61d11))

* Merge pull request #87 from evanhanders/old_to_new

Adds function that converts old .pt pretrained SAEs to new folder format ([`1cb1725`](https://github.com/jbloomAus/SAELens/commit/1cb17251c4a5f9d3e19c1cd4b109b58e7c6f0e7c))

* Merge pull request #93 from jbloomAus/py-312-ci

chore: enabling python 3.12 checks for CI ([`87be422`](https://github.com/jbloomAus/SAELens/commit/87be42246e9173bacd98530368e2267e8e917912))


## v0.5.1 (2024-04-19)

### Chore

* chore: re-enabling isort in CI (#86) ([`9c44731`](https://github.com/jbloomAus/SAELens/commit/9c44731a9b7718c9f0913136ed9df42dac87c390))

### Fix

* fix: pin pyzmq==26.0.1 temporarily ([`0094021`](https://github.com/jbloomAus/SAELens/commit/00940219754ddb1be6708e54cdd0ac6ed5dc3948))

* fix: typing issue, temporary ([`25cebf1`](https://github.com/jbloomAus/SAELens/commit/25cebf1e5e0630a377a5045c1b3571a5f181853f))

### Unknown

* v0.5.1 ([`0ac218b`](https://github.com/jbloomAus/SAELens/commit/0ac218bf8068b8568310b40a1399f9eb3c8d992e))

* fixes string vs path typing errors ([`94f1fc1`](https://github.com/jbloomAus/SAELens/commit/94f1fc127f74dbd95ae586ff3ce4d15605403e14))

* removes unused import ([`06406b0`](https://github.com/jbloomAus/SAELens/commit/06406b096f5295aab2235247def704cddd4d3dd4))

* updates formatting for alignment with repo standards. ([`5e1f342`](https://github.com/jbloomAus/SAELens/commit/5e1f3428058d63ffa8e627f322909685ed1fdf59))

* consolidates with SAE class load_legacy function &amp; adds test ([`0f85ded`](https://github.com/jbloomAus/SAELens/commit/0f85dede8437271724f1ec5dc79cc97829dee49b))

* adds old-&gt;new file conversion function ([`fda2b57`](https://github.com/jbloomAus/SAELens/commit/fda2b571bc303545b5714bee3c27d3f4a2e56186))

* Merge pull request #91 from jbloomAus/decoder-fine-tuning

Decoder fine tuning ([`1fc652c`](https://github.com/jbloomAus/SAELens/commit/1fc652c19e2a34172c1fd520565e9620366f565c))

* par update ([`2bb5975`](https://github.com/jbloomAus/SAELens/commit/2bb5975226807d352f2d3cf6b6dad7aefaf1b662))

* Merge pull request #89 from jbloomAus/fix_np

Enhance + Fix Neuronpedia generation / upload ([`38d507c`](https://github.com/jbloomAus/SAELens/commit/38d507c052875cbd78f8fd9dae45a658e47c2b9d))

* minor changes ([`bc766e4`](https://github.com/jbloomAus/SAELens/commit/bc766e4f7a8d472f647408b8a5cd3c6140d856b7))

* reformat run.ipynb ([`822882c`](https://github.com/jbloomAus/SAELens/commit/822882cac9c05449b7237b7d42ce17297903da2f))

* get decoder fine tuning working ([`11a71e1`](https://github.com/jbloomAus/SAELens/commit/11a71e1b95576ef6dc3dbec7eb1c76ce7ca44dfd))

* format ([`040676d`](https://github.com/jbloomAus/SAELens/commit/040676db6814c1f64171a32344e0bed40528c8f9))

* Merge pull request #88 from jbloomAus/get_feature_from_neuronpedia

FEAT: Add API for getting Neuronpedia feature ([`1666a68`](https://github.com/jbloomAus/SAELens/commit/1666a68bb7d7ee4837e03d95b203ee371ca9ea9e))

* Fix resuming from batch ([`145a407`](https://github.com/jbloomAus/SAELens/commit/145a407f8d57755301bf56c87efd4e775c59b980))

* Use original repo for sae_vis ([`1a7d636`](https://github.com/jbloomAus/SAELens/commit/1a7d636c95ef508dde8bd100ab6d9f241b0be977))

* Use correct model name for np runner ([`138d5d4`](https://github.com/jbloomAus/SAELens/commit/138d5d445878c0830c6c96a5fbe6b10a1d9644b0))

* Merge main, remove eindex ([`6578436`](https://github.com/jbloomAus/SAELens/commit/6578436891e71a0ef60fb2ed6d6a6b6279d71cbc))

* Add API for getting Neuronpedia feature ([`e78207d`](https://github.com/jbloomAus/SAELens/commit/e78207d086cb3372dc805cbb4c87b694749cd905))


## v0.5.0 (2024-04-17)

### Feature

* feat: Mamba support vs mamba-lens (#79)

* mamba support

* added init

* added optional model kwargs

* Support transformers and mamba

* forgot one model kwargs

* failed opts

* tokens input

* hack to fix tokens, will look into fixing mambalens

* fixed checkpoint

* added sae group

* removed some comments and fixed merge error

* removed unneeded params since that issue is fixed in mambalens now

* Unneded input param

* removed debug checkpoing and eval

* added refs to hookedrootmodule

* feed linter

* added example and fixed loading

* made layer for eval change

* fix linter issues

* adding mamba-lens as optional dep, and fixing typing/linting

* adding a test for loading mamba model

* adding mamba-lens to dev for CI

* updating min mamba-lens version

* updating mamba-lens version

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`eea7db4`](https://github.com/jbloomAus/SAELens/commit/eea7db4b99098c33cd862e7e2280a32b630826bd))

### Unknown

* update readme ([`440df7b`](https://github.com/jbloomAus/SAELens/commit/440df7b6c0ef55ba3d116054f81e1ee4a58f9089))

* update readme ([`3694fd2`](https://github.com/jbloomAus/SAELens/commit/3694fd2c4cc7438121e4549636508c45835a5d38))

* Fix upload skipped/dead features ([`932f380`](https://github.com/jbloomAus/SAELens/commit/932f380971ce3d431e6592c804d12f6df2b4ec78))

* Use python typer instead of shell script for neuronpedia jobs ([`b611e72`](https://github.com/jbloomAus/SAELens/commit/b611e721dd2620ab5a030cc0f6e37029c30711ca))

* Merge branch &#39;main&#39; into fix_np ([`cc6cb6a`](https://github.com/jbloomAus/SAELens/commit/cc6cb6a96b793e41fb91f4ebbaf3bfa5e7c11b4e))

* convert sparsity to log sparsity if needed ([`8d7d404`](https://github.com/jbloomAus/SAELens/commit/8d7d4040033fb80c5b994cdc662b0f90b8fcc7aa))


## v0.4.0 (2024-04-16)

### Feature

* feat: support orthogonal decoder init and no pre-decoder bias ([`ac606a3`](https://github.com/jbloomAus/SAELens/commit/ac606a3b85ac48dd800e434878b3a8bfe1838408))

### Fix

* fix: sae dict bug ([`484163e`](https://github.com/jbloomAus/SAELens/commit/484163ed4c7395694d78f8a0ce1be9570a0ded2a))

* fix: session loader wasn&#39;t working ([`a928d7e`](https://github.com/jbloomAus/SAELens/commit/a928d7e24224cc79eedfd5c050d02d5f22de86e8))

### Unknown

* enable setting adam pars in config ([`1e53ede`](https://github.com/jbloomAus/SAELens/commit/1e53edee5cc4756283e95f824693de3c781ec532))

* fix sae dict loader and format ([`c558849`](https://github.com/jbloomAus/SAELens/commit/c558849bbf310961cd0905eab087572708272774))

* default orthogonal init false ([`a8b0113`](https://github.com/jbloomAus/SAELens/commit/a8b0113140bd2f9b97befccc8f158dace02a4810))

* Formatting ([`1e3d53e`](https://github.com/jbloomAus/SAELens/commit/1e3d53ec2b72897bfebb6065f3b530fe65d3a97c))

* Eindex required by sae_vis ([`f769e7a`](https://github.com/jbloomAus/SAELens/commit/f769e7a65ab84d4073852931a86ff3b5076eea3c))

* Upload dead feature stubs ([`9067380`](https://github.com/jbloomAus/SAELens/commit/9067380bf67b89d8b2d235944f696016286f683e))

* Make feature sparsity an argument ([`8230570`](https://github.com/jbloomAus/SAELens/commit/8230570297d68e35cb614a63abf442e4a01174d2))

* Fix buffer&#34; ([`dde2481`](https://github.com/jbloomAus/SAELens/commit/dde248162b70ff4311d4182333b7cce43aed78df))

* Merge branch &#39;main&#39; into fix_np ([`6658392`](https://github.com/jbloomAus/SAELens/commit/66583923cd625bfc1c1ef152bc5f5beaa764b2d6))

* notebook update ([`feca408`](https://github.com/jbloomAus/SAELens/commit/feca408cf003737cd4eb529ca7fea2f77984f5c6))

* Merge branch &#39;main&#39; into fix_np ([`f8fb3ef`](https://github.com/jbloomAus/SAELens/commit/f8fb3efbde7fc79e6fafe2d9b3324c9f0b2a337d))

* Final fixes ([`e87788d`](https://github.com/jbloomAus/SAELens/commit/e87788d63a9b767e34e497c85a318337ab8aabb8))

* Don&#39;t use buffer, fix anomalies ([`2c9ca64`](https://github.com/jbloomAus/SAELens/commit/2c9ca642b334b7a444544a4640c483229dc04c62))


## v0.3.0 (2024-04-15)

### Feature

* feat: add basic tutorial for training saes ([`1847280`](https://github.com/jbloomAus/SAELens/commit/18472800481dbe584e5fab8533ac47a1ee39a062))


## v0.2.2 (2024-04-15)

### Fix

* fix: dense batch dim mse norm optional ([`8018bc9`](https://github.com/jbloomAus/SAELens/commit/8018bc939811bdb7e59c999e055c26401af6d0d2))

### Unknown

* format ([`c359c27`](https://github.com/jbloomAus/SAELens/commit/c359c272ae4d5b1e25da5333c4beff99e924532c))

* make dense_batch_mse_normalization optional ([`c41774e`](https://github.com/jbloomAus/SAELens/commit/c41774e5cfaeb195e3320e9e3fc93d60d921337d))

* Runner is fixed, faster, cleaned up, and now gives whole sequences instead of buffer. ([`3837884`](https://github.com/jbloomAus/SAELens/commit/383788485917cee114fba24e8ded944aefcfb568))

* Merge branch &#39;main&#39; into fix_np ([`3ed30cf`](https://github.com/jbloomAus/SAELens/commit/3ed30cf2b84a2444c8ed030641214f0dbb65898a))

* add warning in run script ([`9a772ca`](https://github.com/jbloomAus/SAELens/commit/9a772ca6da155b5e97bc3109da74457f5addfbfd))

* update sae loading code ([`356a8ef`](https://github.com/jbloomAus/SAELens/commit/356a8efba06e4f453d2f15afe9171b71d780819a))

* add device override to session loader ([`96b1e12`](https://github.com/jbloomAus/SAELens/commit/96b1e120d78f5f05cd94aec7a763bc14849aa1d3))

* update readme ([`5cd5652`](https://github.com/jbloomAus/SAELens/commit/5cd5652a4b19ba985d20c229b6a92d17774bc6b9))


## v0.2.1 (2024-04-13)

### Fix

* fix: neuronpedia quicklist ([`6769466`](https://github.com/jbloomAus/SAELens/commit/676946654c89ea63f6244f7326cc970b9354f4e3))


## v0.2.0 (2024-04-13)

### Chore

* chore: improving CI speed ([`9e3863c`](https://github.com/jbloomAus/SAELens/commit/9e3863c0e473797bd213ada901415834b934cb8c))

* chore: updating README.md with pip install instructions and PyPI badge ([`682db80`](https://github.com/jbloomAus/SAELens/commit/682db803ae6128d6b74cba968c23925d8f5effff))

### Feature

* feat: overhaul saving and loading ([`004e8f6`](https://github.com/jbloomAus/SAELens/commit/004e8f64498a8cca52c9bb485baee8cb934dbfc7))

### Unknown

* Use legacy loader, add back histograms, logits. Fix anomaly characters. ([`ebbb622`](https://github.com/jbloomAus/SAELens/commit/ebbb622353bef21c953f844a108ea8d9fe31e9f9))

* Merge branch &#39;main&#39; into fix_np ([`586e088`](https://github.com/jbloomAus/SAELens/commit/586e0881e08a9b013e2d4101878ef054c1f3dd8b))

* Merge pull request #80 from wllgrnt/will-update-tutorial

bugfix - minimum viable updates to tutorial notebook ([`e51016b`](https://github.com/jbloomAus/SAELens/commit/e51016b01f3b0f30c83365c54430908779671d87))

* minimum viable fixes to evaluation notebook ([`b907567`](https://github.com/jbloomAus/SAELens/commit/b907567118ac891e712c5f31a5f0bc02d672008e))

* Merge pull request #76 from jbloomAus/faster-ci

perf: improving CI speed ([`8b00000`](https://github.com/jbloomAus/SAELens/commit/8b000002764b3d2d8cebea2d8fe92e2092b1756e))

* try partial cache restore ([`392f982`](https://github.com/jbloomAus/SAELens/commit/392f982f413d02358d14a9a0702ee63857fd7232))

* Merge branch &#39;main&#39; into faster-ci ([`89e1568`](https://github.com/jbloomAus/SAELens/commit/89e1568d114b49cd10778e902d4342c2c48a8359))

* Merge pull request #78 from jbloomAus/fix-artifact-saving-loading

Fix artifact saving loading ([`8784c74`](https://github.com/jbloomAus/SAELens/commit/8784c74df08eece5d349cab9207ea1d544921c5b))

* remove duplicate code ([`6ed6af5`](https://github.com/jbloomAus/SAELens/commit/6ed6af55c5642629e019feeada1319df3cfd5c7a))

* set device in load from pretrained ([`b4e12cd`](https://github.com/jbloomAus/SAELens/commit/b4e12cd4b35e834402a9b915d4f02082e4eea711))

* fix typing issue which required ignore ([`a5df8b0`](https://github.com/jbloomAus/SAELens/commit/a5df8b00f33b1a6036ecf0517ad28f7fd92cb2fa))

* remove print statement ([`295e0e4`](https://github.com/jbloomAus/SAELens/commit/295e0e46421683a97e9871b7522a7c722e957e01))

* remove load with session option ([`74926e1`](https://github.com/jbloomAus/SAELens/commit/74926e162b47c04d71dbb151defbded6e27293b3))

* fix broken test ([`16935ef`](https://github.com/jbloomAus/SAELens/commit/16935efee3fb13195aa3931f2465b8f82e474cb8))

* avoid tqdm repeating during training ([`1d70af8`](https://github.com/jbloomAus/SAELens/commit/1d70af82278f5afb2406e6a8aa41ade3997f30b4))

* avoid division by 0 ([`2c7c6d8`](https://github.com/jbloomAus/SAELens/commit/2c7c6d857e6820c0a101765182393c591bf83369))

* remove old notebook ([`e1ad1aa`](https://github.com/jbloomAus/SAELens/commit/e1ad1aaeca1558cdc18174c70499f26f33bf26cf))

* use-sae-dict-not-group ([`27f8003`](https://github.com/jbloomAus/SAELens/commit/27f8003a6d655c0f08a2dd947ee662f68660b176))

* formatting ([`827abd0`](https://github.com/jbloomAus/SAELens/commit/827abd08e45aa4d436240442d9d2f843b260b346))

* improve artifact loading storage, tutorial forthcoming ([`604f102`](https://github.com/jbloomAus/SAELens/commit/604f102d54ab1f8981fb33c27addbd9f0f1128dc))

* add safetensors to project ([`0da48b0`](https://github.com/jbloomAus/SAELens/commit/0da48b044357eed17e5afffd3ce541e064185043))

* Don&#39;t precompute background colors and tick values ([`271dbf0`](https://github.com/jbloomAus/SAELens/commit/271dbf05567b6e6ae4cfc1dab138132872038381))

* Merge pull request #71 from weissercn/main

Addressing notebook issues ([`8417505`](https://github.com/jbloomAus/SAELens/commit/84175055ba5876b335cbc0de38bf709d0b11cec1))

* Merge pull request #70 from jbloomAus/update-readme-install

chore: updating README.md with pip install instructions and PyPI badge ([`4d7d1e7`](https://github.com/jbloomAus/SAELens/commit/4d7d1e7db5e952c7e9accf19c0ccce466cdcf6cf))

* FIX: Add back correlated neurons, frac_nonzero ([`d532b82`](https://github.com/jbloomAus/SAELens/commit/d532b828bd77c18b73f495d6b42ca53b5148fd2f))

* linting ([`1db0b5a`](https://github.com/jbloomAus/SAELens/commit/1db0b5ae7e091822c72bba0488d30fc16bc9a1c6))

* fixed graph name ([`ace4813`](https://github.com/jbloomAus/SAELens/commit/ace481322103737de2e80d688683d0c937ac5558))

* changed key for df_enrichment_scores, so it can be run ([`f0a9d0b`](https://github.com/jbloomAus/SAELens/commit/f0a9d0b30d52a163e0d715b1841a42009e444b64))

* fixed space in notebook 2 ([`2278419`](https://github.com/jbloomAus/SAELens/commit/2278419afb33a016b9a8bcf0759144b66cfe10da))

* fixed space in notebook 2 ([`24a6696`](https://github.com/jbloomAus/SAELens/commit/24a6696c07d7e06ac7f9487209bd4aa6d78412ee))

* fixed space in notebook ([`d2f8c8e`](https://github.com/jbloomAus/SAELens/commit/d2f8c8e28264368358a211a2a502e1638d7d9a25))

* fixed pickle backwards compatibility in tutorial ([`3a97a04`](https://github.com/jbloomAus/SAELens/commit/3a97a04097b9a0f480268661e38904926853cac3))


## v0.1.0 (2024-04-06)

### Feature

* feat: release ([`c70b148`](https://github.com/jbloomAus/SAELens/commit/c70b148fd49f880d424adb62a0817e03f2aacb44))

### Fix

* fix: removing paths-ignore from action to avoid blocking releases ([`28ff797`](https://github.com/jbloomAus/SAELens/commit/28ff7971aeeba2d90d544c0b6a6fb7b743780ef4))

* fix: updating saevis version to use pypi ([`dbd96a2`](https://github.com/jbloomAus/SAELens/commit/dbd96a2ef5d10012bfec43347b3ee920e59cdf9c))

### Unknown

* Merge pull request #69 from chanind/remove-ci-ignore

fix: removing paths-ignore from action to avoid blocking releases ([`179cea1`](https://github.com/jbloomAus/SAELens/commit/179cea190e8ae1a54659e325b8700d8d88026180))

* Update README.md ([`1720ce8`](https://github.com/jbloomAus/SAELens/commit/1720ce827f31b7ef769bd85345b9f1b996d9c3a8))

* Merge pull request #68 from chanind/updating-sae-vis

fix: hotfix updating saevis version to use pypi ([`a13cee3`](https://github.com/jbloomAus/SAELens/commit/a13cee3b716624219523294a51b5e3655ad9630c))


## v0.0.0 (2024-04-06)

### Chore

* chore: adding more tests to ActivationsStore + light refactoring ([`cc9899c`](https://github.com/jbloomAus/SAELens/commit/cc9899c7c7ecfa734bdb9260b80878092a0b3d4f))

* chore: running isort to fix imports ([`53853b9`](https://github.com/jbloomAus/SAELens/commit/53853b94c2809fe7145c92b6e6e68035ca5f2fe9))

* chore: setting up pyright type checking and fixing typing errors ([`351995c`](https://github.com/jbloomAus/SAELens/commit/351995c1e0735073dd08a7fb0344cf0991d4a5a2))

* chore: enable full flake8 default rules list ([`19886e2`](https://github.com/jbloomAus/SAELens/commit/19886e28ed8358e787e1dad9b1dcfa5cf3dba7b5))

* chore: using poetry for dependency management ([`465e003`](https://github.com/jbloomAus/SAELens/commit/465e00333b9fae79ce73c277e7e1018e863002ca))

* chore: removing .DS_Store files ([`32f09b6`](https://github.com/jbloomAus/SAELens/commit/32f09b67e68b116ee546da16059b34f8b0db23fa))

### Unknown

* Merge pull request #66 from chanind/pypi

feat: setting up sae_lens package and auto-deploy with semantic-release ([`34633e8`](https://github.com/jbloomAus/SAELens/commit/34633e892624808fe1f98048beb63939e610e3af))

* Merge branch &#39;main&#39; into pypi ([`3ce7f99`](https://github.com/jbloomAus/SAELens/commit/3ce7f99fe7a6be2d87fb3fff484d6915419c0dcb))

* Merge pull request #60 from chanind/improve-config-typing

fixing config typing ([`b8fba4f`](https://github.com/jbloomAus/SAELens/commit/b8fba4fce043a41e3b997fae2d067eafc6264268))

* setting up sae_lens package and auto-deploy with semantic-release ([`ba41f32`](https://github.com/jbloomAus/SAELens/commit/ba41f327364901c40c6613a300e1ceaafe67e8fa))

* fixing config typing

switch to using explicit params for ActivationsStore config instead of RunnerConfig base class ([`9be3445`](https://github.com/jbloomAus/SAELens/commit/9be344512f15c24f5106b1e43200ea49b8c33981))

* Merge pull request #65 from chanind/fix-forgotten-scheduler-opts

passing accidentally overlooked scheduler opts ([`773bc02`](https://github.com/jbloomAus/SAELens/commit/773bc02e6a3b5f2a2c45e156c7f236485cdd79fe))

* passing accidentally overlooked scheduler opts ([`ad089b7`](https://github.com/jbloomAus/SAELens/commit/ad089b7c0402d0cf4d1b8a37c597e9551b93b55e))

* Merge pull request #64 from chanind/lr-decay

adding lr_decay_steps and refactoring get_scheduler ([`c960d99`](https://github.com/jbloomAus/SAELens/commit/c960d99ab9465fc86793c554d3ed810e9eb226ee))

* adding lr_decay_steps and refactoring get_scheduler ([`fd5448c`](https://github.com/jbloomAus/SAELens/commit/fd5448c5a016a8134a881732f6c61cd7a37eab67))

* Merge pull request #53 from hijohnnylin/neuronpedia_runner

Generate and upload Neuronpedia artifacts ([`0b94f84`](https://github.com/jbloomAus/SAELens/commit/0b94f845739884146faead573f484cd1bc6d8737))

* format ([`792c7cb`](https://github.com/jbloomAus/SAELens/commit/792c7cb235faace99acfff2416a29a88363ca245))

* ignore type incorrectness in imported package ([`5fe83a9`](https://github.com/jbloomAus/SAELens/commit/5fe83a9b09238201219b65df6b70ce074fed68eb))

* Merge pull request #63 from chanind/remove-eindex

removing unused eindex depencency ([`1ce44d7`](https://github.com/jbloomAus/SAELens/commit/1ce44d7e3a3c51e013a68b0d7d82349da0ff8edf))

* removing unused eindex depencency ([`7cf991b`](https://github.com/jbloomAus/SAELens/commit/7cf991ba534c6d9b5ea8a96cc1c4622183b211f9))

* Safe to_str_tokens, fix memory issues ([`901b888`](https://github.com/jbloomAus/SAELens/commit/901b888c98df7de77626adfbc31e96298585958e))

* Allow starting neuronpedia generation at a specific batch numbe ([`85d8f57`](https://github.com/jbloomAus/SAELens/commit/85d8f578839687159b0c0ba6b1879746a23f3288))

* FIX: Linting &#39;do not use except&#39; ([`ce3d40c`](https://github.com/jbloomAus/SAELens/commit/ce3d40c61ceb1a1e7fd5c598371a3386f62a39d8))

* Fix vocab: Ċ should be line break. Also set left and right buffers ([`205b1c1`](https://github.com/jbloomAus/SAELens/commit/205b1c18fba46ef0b4629846bc4dc224c34bc2f6))

* Merge ([`b159010`](https://github.com/jbloomAus/SAELens/commit/b1590102bceb0d859d00a65fa68925e1a1ce2341))

* Update Neuronpedia Runner ([`885de27`](https://github.com/jbloomAus/SAELens/commit/885de2770279ed4188042985a500c3c393c57ced))

* Merge pull request #58 from canrager/main

Make prepend BOS optional: Default True ([`48a07f9`](https://github.com/jbloomAus/SAELens/commit/48a07f928ec569bba5502af379c789747e15a9e0))

* make tests pass with use_bos flag ([`618d4bb`](https://github.com/jbloomAus/SAELens/commit/618d4bbdbaf8fed344886eb292e2f5423f98fbbc))

* Merge pull request #59 from chanind/fix-docs-deploy

attempting to fix docs deploy ([`cfafbe7`](https://github.com/jbloomAus/SAELens/commit/cfafbe75026104d707c21d82efddd988bd1a3833))

* force docs push ([`3aa179d`](https://github.com/jbloomAus/SAELens/commit/3aa179d350d3602d873ebd5fa585d0a9c7475c79))

* ignore type eror ([`e87198b`](https://github.com/jbloomAus/SAELens/commit/e87198bba4c65a77be2e7c1b072e563c9497c9d0))

* format ([`67dfb46`](https://github.com/jbloomAus/SAELens/commit/67dfb4685ba293db2266b1b4885964e2313104f9))

* attempting to fix docs deploy ([`cda8ece`](https://github.com/jbloomAus/SAELens/commit/cda8eceb0a8d356fd4f4b461a2fb473ab0071a1d))

* Merge branch &#39;main&#39; of https://github.com/jbloomAus/mats_sae_training into main ([`8aadcd3`](https://github.com/jbloomAus/SAELens/commit/8aadcd3171748ecf0ff759c67522d04ac0f3c3a1))

* add prepend bos flag ([`c0b29cc`](https://github.com/jbloomAus/SAELens/commit/c0b29cc5037bfeea88888f2bb90ebd2ecdf2ed05))

* fix attn out on run evals ([`02fa90b`](https://github.com/jbloomAus/SAELens/commit/02fa90be6c70cfc93f13262462d3ec4cc86641b8))

* Merge pull request #57 from chanind/optim-tests

Adding tests to get_scheduler ([`13c8085`](https://github.com/jbloomAus/SAELens/commit/13c8085e1becb654e50fe69435a3f5814f0d2145))

* Merge pull request #56 from chanind/sae-tests

minor refactoring to SAE and adding tests ([`2c425ca`](https://github.com/jbloomAus/SAELens/commit/2c425cabf7718c16eb62146972f389d2d5ff2a32))

* minor refactoring to SAE and adding tests ([`92a98dd`](https://github.com/jbloomAus/SAELens/commit/92a98ddf8e6eba0b4cd4ac2e2350281f9ed2e2d0))

* adding tests to get_scheduler ([`3b7e173`](https://github.com/jbloomAus/SAELens/commit/3b7e17368c7abfac2747f6fa8e3e03650914cfe5))

* Generate and upload Neuronpedia artifacts ([`b52e0e2`](https://github.com/jbloomAus/SAELens/commit/b52e0e2c427a4b74bebff5d94961725148c404f0))

* Merge pull request #54 from jbloomAus/hook_z_suppourt

notional support, needs more thorough testing ([`277f35b`](https://github.com/jbloomAus/SAELens/commit/277f35b35b2b8f6475b5b8ae9f15356b168e9f50))

* Merge pull request #55 from chanind/contributing-docs

adding a contribution guide to docs ([`8ac8f05`](https://github.com/jbloomAus/SAELens/commit/8ac8f051c2686169f5143582ace162f7f337af50))

* adding a contribution guide to docs ([`693c5b3`](https://github.com/jbloomAus/SAELens/commit/693c5b335527630dc246ebc55519008f87465913))

* notional support, needs more thorough testing ([`9585022`](https://github.com/jbloomAus/SAELens/commit/9585022366677c66d6cdbedd82d10711d1001fef))

* Generate and upload Neuronpedia artifacts ([`4540268`](https://github.com/jbloomAus/SAELens/commit/45402681bc4051bdbc160cd018c8c73ed4c7321f))

* Merge pull request #52 from hijohnnylin/fix_db_runner_assert

FIX: Don&#39;t check wandb assert if not using wandb ([`5c48811`](https://github.com/jbloomAus/SAELens/commit/5c488116a0e5096e9e37d6d714dbd7a3835cb6fe))

* FIX: Don&#39;t check wandb assert if not using wandb ([`1adefda`](https://github.com/jbloomAus/SAELens/commit/1adefda8f5c2f07c468ce63b10ba608faefc0877))

* add docs badge ([`f623ed1`](https://github.com/jbloomAus/SAELens/commit/f623ed1b57f8753af3c995b68443b2459b258b42))

* try to get correct deployment ([`777dd6c`](https://github.com/jbloomAus/SAELens/commit/777dd6cfc57b3a8cd5e1d5056298d253ccb49d9f))

* Merge pull request #51 from jbloomAus/mkdocs

Add Docs to the project. ([`d2ebbd7`](https://github.com/jbloomAus/SAELens/commit/d2ebbd7b8ad1468da97f3e407934b8dd8e9e3f2f))

* mkdocs, test ([`9f14250`](https://github.com/jbloomAus/SAELens/commit/9f142509da4e1a63ffc88a32a692dc5218aad96d))

* code cov ([`2ae6224`](https://github.com/jbloomAus/SAELens/commit/2ae62243172088b309e40b3b4fca0ce43c08f19a))

* Merge pull request #48 from chanind/fix-sae-vis-version

Pin sae_vis to previous working version ([`3f8a30b`](https://github.com/jbloomAus/SAELens/commit/3f8a30bf577efdaae717f1c6b930be35c9ad7883))

* fix suffix issue ([`209ba13`](https://github.com/jbloomAus/SAELens/commit/209ba1324a775c30a0a1fc2941d0fdb46271ac53))

* pin sae_vis to previous working version ([`ae0002a`](https://github.com/jbloomAus/SAELens/commit/ae0002a67838ce33474d2697dce20d47356f8abb))

* don&#39;t ignore changes to .github ([`35fdeec`](https://github.com/jbloomAus/SAELens/commit/35fdeec3d3c1e31c2eb884060e7550a8d5fa9b22))

* add cov report ([`971d497`](https://github.com/jbloomAus/SAELens/commit/971d4979193efe3f30cc3d580df648a413fd16b2))

* Merge pull request #40 from chanind/refactor-train-sae

Refactor train SAE and adding unit tests ([`5aa0b11`](https://github.com/jbloomAus/SAELens/commit/5aa0b1121112024e44dc5a2d02e33e532b741ed9))

* Merge branch &#39;main&#39; into refactor-train-sae ([`0acdcb3`](https://github.com/jbloomAus/SAELens/commit/0acdcb395aaa42e0344191e680a4f521af6d430c))

* Merge pull request #41 from jbloomAus/move_to_sae_vis

Move to sae vis ([`bcb9a52`](https://github.com/jbloomAus/SAELens/commit/bcb9a5207339528d55d5c89640eec0b1a9c449cb))

* flake8 can ignore imports, we&#39;re using isort anyway ([`6b7ae72`](https://github.com/jbloomAus/SAELens/commit/6b7ae72cd324929c7b53d4c1043ae422d8eec81d))

* format ([`af680e2`](https://github.com/jbloomAus/SAELens/commit/af680e28e0978583758d7d584ff663f99482fdce))

* fix mps bug ([`e7b238f`](https://github.com/jbloomAus/SAELens/commit/e7b238fd4813b76c647468949603b90fc37c1d8d))

* more tests ([`01978e6`](https://github.com/jbloomAus/SAELens/commit/01978e6dd141c269c403a4e84f5de2fb26af6057))

* wip ([`4c03b3d`](https://github.com/jbloomAus/SAELens/commit/4c03b3d3c4a89275d7b08d62d0c9ae18e0b865db))

* more tests ([`7c1cb6b`](https://github.com/jbloomAus/SAELens/commit/7c1cb6b827e1e455e715cf9aa1f6c82e687eaf2e))

* testing that sparsity counts get updated correctly ([`5b5d653`](https://github.com/jbloomAus/SAELens/commit/5b5d6538c90d11fcc040ff05085642066c799518))

* adding some unit tests to _train_step() ([`dbf3f01`](https://github.com/jbloomAus/SAELens/commit/dbf3f01ffb5ad3c2f403c6a81f7eab5c4646100c))

* Merge branch &#39;main&#39; into refactor-train-sae ([`2d5ec98`](https://github.com/jbloomAus/SAELens/commit/2d5ec98164dd4fde0c34c0e966e78b9a3848b842))

* Update README.md ([`d148b6a`](https://github.com/jbloomAus/SAELens/commit/d148b6a2ad77e546b433cf2c3c5f9993f04228ba))

* Merge pull request #20 from chanind/activations_store_tests

chore: adding more tests to ActivationsStore + light refactoring ([`69dcf8e`](https://github.com/jbloomAus/SAELens/commit/69dcf8e551415f6a3884a509ab4428818ad35936))

* Merge branch &#39;main&#39; into activations_store_tests ([`4896d0a`](https://github.com/jbloomAus/SAELens/commit/4896d0a4a850551d4e7bc91358adc7b98b7fc705))

* refactoring train_sae_on_language_model.py into smaller functions ([`e75a15d`](https://github.com/jbloomAus/SAELens/commit/e75a15d3acb4b02a435d70b25a67e8103c9196b3))

* suppourt apollo pretokenized datasets ([`e814054`](https://github.com/jbloomAus/SAELens/commit/e814054a5693361ba51c9aed422b3a8b9d6d0fe6))

* handle saes saved before groups ([`5acd89b`](https://github.com/jbloomAus/SAELens/commit/5acd89b02c02fe43fae5cea9d1dadeff41424069))

* typechecker ([`fa6cc49`](https://github.com/jbloomAus/SAELens/commit/fa6cc4902f25843e1f7ae8f6b9841ecd4f074e9b))

* fix geom median bug ([`8d4a080`](https://github.com/jbloomAus/SAELens/commit/8d4a080c404f3fa7c16e95e72d0be3b3a7812d0c))

* remove references to old code ([`861151f`](https://github.com/jbloomAus/SAELens/commit/861151fd04c5fc8963396f655c08818bcff08d1c))

* remove old geom median code ([`05e0aca`](https://github.com/jbloomAus/SAELens/commit/05e0acafc690b7f6e08f8a8c62989afb21b7d1f1))

* Merge pull request #22 from themachinefan/faster_geometric_median

Faster geometric median. ([`341c49a`](https://github.com/jbloomAus/SAELens/commit/341c49ac91b8876b020c9a22c5ea3c0b7ea74b3e))

* makefile check type and types of geometric media ([`736bf83`](https://github.com/jbloomAus/SAELens/commit/736bf83842f2e4be9baeee6ac12731b9b4834fc6))

* Merge pull request #21 from schmatz/fix-dashboard-image

Fix broken dashboard image on README ([`eb90cc9`](https://github.com/jbloomAus/SAELens/commit/eb90cc9d835da6febe4a4c5830a134237e3705eb))

* Merge pull request #24 from neelnanda-io/add-post-link

Added link to AF post ([`39f8d3d`](https://github.com/jbloomAus/SAELens/commit/39f8d3d8ff5659c9815f9a1ff3be7bf81370813c))

* Added link to AF post ([`f0da9ea`](https://github.com/jbloomAus/SAELens/commit/f0da9ea8af61239c22de5b5778e9026f834fc929))

* formatting ([`0168612`](https://github.com/jbloomAus/SAELens/commit/016861243e11bc1b0f0fe1414bcde6c0a5eec334))

* use device, don&#39;t use cuda if not there ([`20334cb`](https://github.com/jbloomAus/SAELens/commit/20334cb26bee5ddcc2ae6632b18a1f1933ab50f0))

* format ([`ce49658`](https://github.com/jbloomAus/SAELens/commit/ce496583e397a71cec0ae376f9920e981417c0e8))

*  fix tsea typing ([`449d90f`](https://github.com/jbloomAus/SAELens/commit/449d90ffc54a548432da6aca3e47651bebfbf585))

* faster geometric median. Run geometric_median,py to test. ([`92cad26`](https://github.com/jbloomAus/SAELens/commit/92cad26b4dc0fdd03d73a4a8dc9b1d5844308eb8))

* Fix dashboard image ([`6358862`](https://github.com/jbloomAus/SAELens/commit/6358862f1f52961bd444feb1b22a7dde27235021))

*  fix incorrect code used to avoid typing issue ([`ed0b0ea`](https://github.com/jbloomAus/SAELens/commit/ed0b0eafdd7424bbcc83c39d400928536bba3378))

* add nltk ([`bc7e276`](https://github.com/jbloomAus/SAELens/commit/bc7e2766109fd775292d8af1226b188bdec8d807))

* ignore various typing issues ([`6972c00`](https://github.com/jbloomAus/SAELens/commit/6972c0016f1ce5a9b20960727fc7778a7655f833))

* add babe package ([`481069e`](https://github.com/jbloomAus/SAELens/commit/481069ef129c37bfec9cd1cd94d43800c45ae969))

* make formatter happy ([`612c7c7`](https://github.com/jbloomAus/SAELens/commit/612c7c71ea5dfecc8f81bea3484976d82fb505ca))

* share scatter so can link ([`9f88dc3`](https://github.com/jbloomAus/SAELens/commit/9f88dc31dc01ce5b10a3772c013d99b9f9d4cf13))

* add_analysis_files_for_post ([`e75323c`](https://github.com/jbloomAus/SAELens/commit/e75323c13ffda589b8c71ed9de5c7c1adb3d6ce7))

* don&#39;t block on isort linting ([`3949a46`](https://github.com/jbloomAus/SAELens/commit/3949a4641c810c74f8ed093e92fabd31d8e2d02c))

* formatting ([`951a320`](https://github.com/jbloomAus/SAELens/commit/951a320fdcb4e01c95bf2bac7a4929676562e323))

* Update README.md ([`b2478c1`](https://github.com/jbloomAus/SAELens/commit/b2478c14d98cbfc2b0c2b0984d251f3c92d05a41))

* Merge pull request #18 from chanind/type-checking

chore: setting up pyright type checking and fixing typing errors ([`bd5fc43`](https://github.com/jbloomAus/SAELens/commit/bd5fc439e56fc1c9e0e6f9dd2613b77f57c763b2))

* Merge branch &#39;main&#39; into type-checking ([`57c4582`](https://github.com/jbloomAus/SAELens/commit/57c4582c38906b9e6d887ceb25901de5734d76c9))

* Merge pull request #17 from Benw8888/sae_group_pr

SAE Group for sweeps PR ([`3e78bce`](https://github.com/jbloomAus/SAELens/commit/3e78bce5c4bccbb9555d7878dc48adf7cfd823cd))

* Merge pull request #1 from chanind/sae_group_pr_isort_fix

chore: running isort to fix imports ([`dd24413`](https://github.com/jbloomAus/SAELens/commit/dd24413749191fab0001f154a225eb991820a12b))

* black format ([`0ffcf21`](https://github.com/jbloomAus/SAELens/commit/0ffcf21bbe73066cad152680255a25b73f98331c))

* fixed expansion factor sweep ([`749b8cf`](https://github.com/jbloomAus/SAELens/commit/749b8cff1a4ccb130a9ff6be698b766ae7642140))

* remove tqdm from data loader, too noisy ([`de3b1a1`](https://github.com/jbloomAus/SAELens/commit/de3b1a1fad3b176b2f547513374f0ff00b784975))

* fix tests ([`b3054b1`](https://github.com/jbloomAus/SAELens/commit/b3054b1e7cc938bee874db95a7ebfd911417a84a))

* don&#39;t calculate geom median unless you need to ([`d31bc31`](https://github.com/jbloomAus/SAELens/commit/d31bc31689c84382044db1a8663df2d352674d7a))

* add to method ([`b3f6dc6`](https://github.com/jbloomAus/SAELens/commit/b3f6dc6882ea052f6fecb2234d8ff4f6212184d7))

* flake8 and black ([`ed8345a`](https://github.com/jbloomAus/SAELens/commit/ed8345ae4ade1445e7182f5c7deeb876203b6524))

* flake8 linter changes ([`8e41e59`](https://github.com/jbloomAus/SAELens/commit/8e41e599d67a7d8eda644321eda8d0481f813f15))

* Merge branch &#39;main&#39; into sae_group_pr ([`082c813`](https://github.com/jbloomAus/SAELens/commit/082c8138045557a592f57f53256d251f76902b4f))

* Delete evaluating.ipynb ([`d3cafa3`](https://github.com/jbloomAus/SAELens/commit/d3cafa3f28029f4d12285a810df29e7de90311c2))

* Delete activation_storing.py ([`fa82992`](https://github.com/jbloomAus/SAELens/commit/fa829924b0b5caab9c7653876a399732277c5d4e))

* Delete lp_sae_training.py ([`0d1e1c9`](https://github.com/jbloomAus/SAELens/commit/0d1e1c9a5f8b5c865f620bd97619cf108bf06796))

* implemented SAE groups ([`66facfe`](https://github.com/jbloomAus/SAELens/commit/66facfef1094f311f00ee6a8abf53f62548802cf))

* Merge pull request #16 from chanind/flake-default-rules

chore: enable full flake8 default rules list ([`ad84706`](https://github.com/jbloomAus/SAELens/commit/ad84706cc9d3d202669fa8175af9f181f323cd66))

* implemented sweeping via config list ([`80f61fa`](https://github.com/jbloomAus/SAELens/commit/80f61faac4dfdab27d03cec672641d898441a530))

* Merge pull request #13 from chanind/poetry

chore: using poetry for dependency management ([`496f7b4`](https://github.com/jbloomAus/SAELens/commit/496f7b4fa2d8835cec6673bec1ffa3d1f5fba268))

* progress on implementing multi-sae support ([`2ba2131`](https://github.com/jbloomAus/SAELens/commit/2ba2131553b28af08d969304c2a2b68a8861fd98))

* Merge pull request #11 from lucyfarnik/fix-caching-shuffle-edge-case

Fixed edge case in activation cache shuffling ([`3727b5d`](https://github.com/jbloomAus/SAELens/commit/3727b5d585efd3932fd49b540d11778c76bef561))

* Merge pull request #12 from lucyfarnik/add-run-name-to-config

Added run name to config ([`c2e05c4`](https://github.com/jbloomAus/SAELens/commit/c2e05c46b0c7d7159a29aaa243cd33a621437f38))

* Added run name to config ([`ab2aabd`](https://github.com/jbloomAus/SAELens/commit/ab2aabdcd9e293c32168549f9a99c25bf097a969))

* Fixed edge case in activation cache shuffling ([`18fd4a1`](https://github.com/jbloomAus/SAELens/commit/18fd4a1d43bbc8b7a490779f06e4f3c6983e3ea7))

* Merge pull request #9 from chanind/rm-ds-store

chore: removing .DS_Store files ([`37771ce`](https://github.com/jbloomAus/SAELens/commit/37771ce9a9bbd3b4bcf2d8749452f2f8e4516855))

* improve readmen ([`f3fe937`](https://github.com/jbloomAus/SAELens/commit/f3fe937303f0eb54e83771d1b307c3f09d1145fd))

* fix_evals_bad_rebase ([`22e415d`](https://github.com/jbloomAus/SAELens/commit/22e415dfc0ab961470fd2c7ed476f4181e2afa44))

* evals changes, incomplete ([`736c40e`](https://github.com/jbloomAus/SAELens/commit/736c40e16cb89cf7d8bd721518df268fb4a35ee7))

* make tutorial independent of artefact and delete old artefact ([`6754e65`](https://github.com/jbloomAus/SAELens/commit/6754e650e032efbc85b0dd11db480c96640bd0b6))

* fix MSE in ghost grad ([`44f7988`](https://github.com/jbloomAus/SAELens/commit/44f7988e5a97bd93daedc3559cc4988caa128974))

* Merge pull request #5 from jbloomAus/clean_up_repo

Add CI/CD, black formatting, pre-commit with flake8 linting. Fix some bugs. ([`01ccb92`](https://github.com/jbloomAus/SAELens/commit/01ccb92c4dbc161a26b22d60b5c9e0fa2d560519))

* clean up run examples ([`9d46bdd`](https://github.com/jbloomAus/SAELens/commit/9d46bdda9ab3aaf8246bb01cb270dadf3e5e06c9))

* move where we save the final artifact ([`f445fac`](https://github.com/jbloomAus/SAELens/commit/f445facde3eb7c2011905501a7644f8ef6cbd6e6))

* fix activations store innefficiency ([`07d38a0`](https://github.com/jbloomAus/SAELens/commit/07d38a0eb5271d86fe2d5add8ae2ea5f7ae8aadc))

* black format and linting ([`479765b`](https://github.com/jbloomAus/SAELens/commit/479765bc9ecc4f04fe76e1f4f447d0d0281e9045))

* dummy file change ([`912a748`](https://github.com/jbloomAus/SAELens/commit/912a748f7b5bff0193a997ee7369c312f073c35f))

* try adding this branch listed specifically ([`7fd0e0c`](https://github.com/jbloomAus/SAELens/commit/7fd0e0c3d3f951ae18b6b101043ba4a36c9933c4))

* yml not yaml ([`9f3f1c8`](https://github.com/jbloomAus/SAELens/commit/9f3f1c87ceed84afefef5bc6d4b519d538bcac91))

* add ci ([`91aca91`](https://github.com/jbloomAus/SAELens/commit/91aca9142459e2a72478d4546ee0fa5a2910c161))

* get units tests working ([`ade2976`](https://github.com/jbloomAus/SAELens/commit/ade29762b4b94e02c3b44a66bedf150e721aec7c))

* make unit tests pass, add make file ([`08b2c92`](https://github.com/jbloomAus/SAELens/commit/08b2c92f7fbfed38ea44312d53945cf1c9471d90))

* add pytest-cov to requirements.txt ([`ce526df`](https://github.com/jbloomAus/SAELens/commit/ce526df64115113ecb68bd7f1a51b6fe11957553))

* seperate research from main repo ([`32b668c`](https://github.com/jbloomAus/SAELens/commit/32b668ce96c8231eaaa93152d2b7d67dacc91dc5))

* remove comma and set default store batch size lower ([`9761b9a`](https://github.com/jbloomAus/SAELens/commit/9761b9ae5f0d4dd51d445ba1229f9b2ebc628e62))

* notebook for Johny ([`39a18f2`](https://github.com/jbloomAus/SAELens/commit/39a18f2f3fc138ba4743976b139071b7830c36a2))

* best practices ghost grads fix ([`f554b16`](https://github.com/jbloomAus/SAELens/commit/f554b161e7b188d25ca9c21bf80629dc1b5b4e35))

* Update README.md

improved the hyperpars ([`2d4caf6`](https://github.com/jbloomAus/SAELens/commit/2d4caf63e55d86967b906915c90a47094631276f))

* dashboard runner ([`a511223`](https://github.com/jbloomAus/SAELens/commit/a511223ca3b5ee87930dc753b3164e00833dc17b))

* readme update ([`c303c55`](https://github.com/jbloomAus/SAELens/commit/c303c554b6599c38a9b78ef2f5e1651267d38fab))

* still hadn&#39;t fixed the issue, now fixed ([`a36ee21`](https://github.com/jbloomAus/SAELens/commit/a36ee21bb351d279952c638442b3874de7756d0c))

* fix mean of loss which broke in last commit ([`b4546db`](https://github.com/jbloomAus/SAELens/commit/b4546dbdcea26d8812131572aa2c15fdfe4b4d28))

* generate dashboards ([`35fa631`](https://github.com/jbloomAus/SAELens/commit/35fa63110f2ad790036a7eef2e4b7d46fd54fa61))

* Merge pull request #3 from jbloomAus/ghost_grads_dev

Ghost grads dev ([`4d150c2`](https://github.com/jbloomAus/SAELens/commit/4d150c255c2637f485875a5709002195b2e51ce0))

* save final log sparsity ([`98e4f1b`](https://github.com/jbloomAus/SAELens/commit/98e4f1bb416b33d49bef1acc676cc3b1e9ed6441))

* start saving log sparsity ([`4d6df6f`](https://github.com/jbloomAus/SAELens/commit/4d6df6f41482e08530806495d22c5cedb1307448))

* get ghost grads working ([`e863ed7`](https://github.com/jbloomAus/SAELens/commit/e863ed70afe1fb1d423262ee976196d2a69cf8b5))

* add notebook/changes for ghost-grad (not working yet) ([`73053c1`](https://github.com/jbloomAus/SAELens/commit/73053c1fe9eed5a1e145f64a1fe11a300201544b))

* idk, probs good ([`0407ad9`](https://github.com/jbloomAus/SAELens/commit/0407ad9b3749ab6baa1bf6dd3040d2e394369836))

* bunch of shit ([`1ec8f97`](https://github.com/jbloomAus/SAELens/commit/1ec8f97330cc605d003757b9c8d6edb8040119bc))

* Merge branch &#39;main&#39; of github.com:jbloomAus/mats_sae_training ([`a22d856`](https://github.com/jbloomAus/SAELens/commit/a22d85643d54e0924e5e4de8573671edf046bca5))

* Reverse engineering the &#34;not only... but&#34; feature ([`74d4fb8`](https://github.com/jbloomAus/SAELens/commit/74d4fb8d7299ee6ace0331a2d9b8cbab2c965bcb))

* Merge pull request #2 from slavachalnev/no_reinit

Allow sampling method to be None ([`4c5fed8`](https://github.com/jbloomAus/SAELens/commit/4c5fed8bbff8f13fdc534385caeb5455ff9d8a55))

* Allow sampling method to be None ([`166799d`](https://github.com/jbloomAus/SAELens/commit/166799d0a9d10c01d39fdbf3c7e338bc99cc4d96))

* research/week_15th_jan/gpt2_small_resid_pre_3.ipynb ([`52a1da7`](https://github.com/jbloomAus/SAELens/commit/52a1da72b25827b4ade2b13c679ac54c17def800))

* add arg for dead neuron calc ([`ffb75fb`](https://github.com/jbloomAus/SAELens/commit/ffb75fb118db527e4633768534b04b5ef5779122))

* notebooks for lucy ([`0319d89`](https://github.com/jbloomAus/SAELens/commit/0319d890d9da994f830b71361d4850b800f6862b))

* add args for b_dec_init ([`82da877`](https://github.com/jbloomAus/SAELens/commit/82da877da7bf273870eb1446d6c9584650876981))

* add geom median as submodule instead ([`4c0d001`](https://github.com/jbloomAus/SAELens/commit/4c0d0014e4a5f0b5661963ce9fd9318c1cdfca78))

* add geom median to req ([`4c8ac9d`](https://github.com/jbloomAus/SAELens/commit/4c8ac9d53585e60647e140d13903ef44e55aa8b6))

* add-geometric-mean-b_dec-init ([`d5853f8`](https://github.com/jbloomAus/SAELens/commit/d5853f8e075c731309cf208a5cd07b0ae0bdfa1e))

* reset feature sparsity calculation ([`4c7f6f2`](https://github.com/jbloomAus/SAELens/commit/4c7f6f2d0960edc365369141671359c91281c600))

* anthropic sampling ([`048d267`](https://github.com/jbloomAus/SAELens/commit/048d267c0f964086e755c3ca245399c68be4d70d))

* get anthropic resampling working ([`ca74543`](https://github.com/jbloomAus/SAELens/commit/ca74543a117f7bc25e9b40b1dd5e4109a2c0121d))

* add ability to finetune existing autoencoder ([`c1208eb`](https://github.com/jbloomAus/SAELens/commit/c1208eb61ebcc1dfbfc428b8b5fa090eb090bbd8))

* run notebook ([`879ad27`](https://github.com/jbloomAus/SAELens/commit/879ad272f541bb5acc05be7b2a042ac9d45479a6))

* switch to batch size independent loss metrics ([`0623d39`](https://github.com/jbloomAus/SAELens/commit/0623d39b323da05f4ae3b67df25841cabf83d200))

* track mean sparsity ([`75f1547`](https://github.com/jbloomAus/SAELens/commit/75f15476400f26fddd2da9562be995e821745691))

* don&#39;t stop early ([`44078a6`](https://github.com/jbloomAus/SAELens/commit/44078a6a7dd10423619ff3a348fd5276f9aad9b2))

* name runs better ([`5041748`](https://github.com/jbloomAus/SAELens/commit/5041748e9166d6e84d017053fe61e60e194cead2))

* improve-eval-metrics-for-attn ([`00d9b65`](https://github.com/jbloomAus/SAELens/commit/00d9b652d09f18c38dbd4c4e2376fcee3c457ac5))

* add hook q ([`b061ee3`](https://github.com/jbloomAus/SAELens/commit/b061ee30e9cf86226e625e5e2e05ea7b4cc4550e))

* add copy suppression notebook ([`1dc893a`](https://github.com/jbloomAus/SAELens/commit/1dc893a17db8a7866a598a4632034c91c139d862))

* fix check in neuron sampling ([`809becd`](https://github.com/jbloomAus/SAELens/commit/809becdbd67b2e23c8be572188f8f098d770871a))

* Merge pull request #1 from jbloomAus/activations_on_disk

Activations on disk ([`e5f198e`](https://github.com/jbloomAus/SAELens/commit/e5f198ea52741dc54e463c55f36409b2b559bf3a))

* merge into main ([`94ed3e6`](https://github.com/jbloomAus/SAELens/commit/94ed3e62636267d5cfe4163dcb37bbd30827e064))

* notebook ([`b5344a3`](https://github.com/jbloomAus/SAELens/commit/b5344a3048706e4b828c80eaccbc5b8b75eeb2ae))

* various research notebooks ([`be63fce`](https://github.com/jbloomAus/SAELens/commit/be63fcea9db2555c1ac9803fc86008487ed97958))

* Added activations caching to run.ipynb ([`054cf6d`](https://github.com/jbloomAus/SAELens/commit/054cf6dc7082465b05bfbfd138566f41f1631471))

* Added activations dir to gitignore ([`c4a31ae`](https://github.com/jbloomAus/SAELens/commit/c4a31ae1e0876d475bf6881c1570e0fbb00fdff9))

* Saving and loading activations from disk ([`309e2de`](https://github.com/jbloomAus/SAELens/commit/309e2de8881e0b9e9f3b6a786501cb4336143890))

* Fixed typo that threw out half of activations ([`5f73918`](https://github.com/jbloomAus/SAELens/commit/5f73918a9447d63dcee9a73c6f9e2e23bec96abe))

* minor speed improvement ([`f7ea316`](https://github.com/jbloomAus/SAELens/commit/f7ea31679222ff03af0048a0730097eccb644de0))

* add notebook with different example runs ([`c0eac0a`](https://github.com/jbloomAus/SAELens/commit/c0eac0a4f3889be932f50cc7485ef0cc236a46a3))

* add ability to train on attn heads ([`18cfaad`](https://github.com/jbloomAus/SAELens/commit/18cfaad66432c827528e28d3969fcc6dbd16391a))

* add gzip for pt artefacts ([`9614a23`](https://github.com/jbloomAus/SAELens/commit/9614a237ef96a79383e1ead8d7769a427279b310))

* add_example_feature_dashboard ([`e90e54d`](https://github.com/jbloomAus/SAELens/commit/e90e54dd2e7cd6e2844710d9b8ba64a64ef25993))

* get_shit_done ([`ce73042`](https://github.com/jbloomAus/SAELens/commit/ce7304265a01ce4e30537f58b148dd069149c474))

* commit_various_things_in_progress ([`3843c39`](https://github.com/jbloomAus/SAELens/commit/3843c3976936e293b5c87b3501d6a7f397183f68))

* add sae visualizer and tutorial ([`6f4030c`](https://github.com/jbloomAus/SAELens/commit/6f4030c5d411f7c2d23518ce247b553b216499e7))

* make it possible to load  sae trained on cuda onto mps ([`3298b75`](https://github.com/jbloomAus/SAELens/commit/3298b75a5ff1c0a660840c6dfa8931a9faf226f7))

* reduce hist freq, don&#39;t cap re-init ([`debcf0f`](https://github.com/jbloomAus/SAELens/commit/debcf0fd62d4d252710d6b2f947efc33ba934ef7))

* add loader import to readme ([`b63f14e`](https://github.com/jbloomAus/SAELens/commit/b63f14e01758a243487ac87bee710298bfef2603))

* Update README.md ([`88f086b`](https://github.com/jbloomAus/SAELens/commit/88f086b664f1c3750dd4b1b3fe0b79df1bfa180f))

* improve-resampling ([`a3072c2`](https://github.com/jbloomAus/SAELens/commit/a3072c26105b1053ece5390da26cdf8605c4a4b7))

* add readme ([`e9b8e56`](https://github.com/jbloomAus/SAELens/commit/e9b8e56aac2ef694e6cab4a6fc24cf343e188ded))

* fixl0_plus_other_stuff ([`2f162f0`](https://github.com/jbloomAus/SAELens/commit/2f162f09be39a9bf56c40d1e461c064e4f2ed834))

* add checkpoints ([`4cacbfc`](https://github.com/jbloomAus/SAELens/commit/4cacbfc2d32a509b8c34ccb940d0a21470c585ca))

* improve_model_saving_loading ([`f6697c6`](https://github.com/jbloomAus/SAELens/commit/f6697c6cfe654bee8b677401e423f2b09e682f73))

* stuff ([`19d278a`](https://github.com/jbloomAus/SAELens/commit/19d278a223bbdc2673616644f4da90492fe7b169))

* Added support for non-tokenized datasets ([`afcc239`](https://github.com/jbloomAus/SAELens/commit/afcc239de1aa91961a14e5734644d2ffaf4b764a))

* notebook_for_keith ([`d06e09b`](https://github.com/jbloomAus/SAELens/commit/d06e09b70bd7fcf2419f9196970025097b7b55ef))

* fix resampling bug ([`2b43980`](https://github.com/jbloomAus/SAELens/commit/2b43980cc2bae402e2016efaaa042f2cb545a02f))

* test pars ([`f601362`](https://github.com/jbloomAus/SAELens/commit/f6013625963ebabdb86063f8db7a5e8339e37873))

* further-lm-improvments ([`63048eb`](https://github.com/jbloomAus/SAELens/commit/63048eb9e5f12a2f631b454f82a17563efcae328))

* get_lm_working_well ([`eba5f79`](https://github.com/jbloomAus/SAELens/commit/eba5f79ff267217a3f69b4dcab4903c98ed56055))

* basic-lm-training-currently-broken ([`7396b8b`](https://github.com/jbloomAus/SAELens/commit/7396b8ba65a579d8aff0987c61ea9d7ac9177380))

* set_up_lm_runner ([`d1095af`](https://github.com/jbloomAus/SAELens/commit/d1095afefb17f87a2a63841fbe6845b46650dde7))

* fix old test, may remove ([`b407aab`](https://github.com/jbloomAus/SAELens/commit/b407aabfcd9d8ff9a8752a92a26701dbc8da04a2))

* happy with hyperpars on benchmark ([`836298a`](https://github.com/jbloomAus/SAELens/commit/836298a897e759e1f99a35c7d8195bcfb580afe3))

* improve metrics ([`f52c7bb`](https://github.com/jbloomAus/SAELens/commit/f52c7bb142e019f9dd6d6f9d73ff3874bf33b529))

* make toy model runner ([`4851dd1`](https://github.com/jbloomAus/SAELens/commit/4851dd1a15ee57d658d41c8673fd6b3fe71a2b45))

* various-changes-toy-model-test ([`a61b75f`](https://github.com/jbloomAus/SAELens/commit/a61b75f058b79262e587a302319a1bb9136c353e))

* Added activation store and activation gathering ([`a85f24d`](https://github.com/jbloomAus/SAELens/commit/a85f24d9389936dc7ad18aaf74d600669294b9cc))

* First successful run on toy models ([`4927145`](https://github.com/jbloomAus/SAELens/commit/4927145a6dfe9026d47c1bea1497667eb5a715fe))

* halfway-to-toy-models ([`feeb411`](https://github.com/jbloomAus/SAELens/commit/feeb411b4526c0627a9a558cfcab91252f90046b))

* Initial commit ([`7a94b0e`](https://github.com/jbloomAus/SAELens/commit/7a94b0ebc34edd24e98df753d100e8a4c238d4f6))
