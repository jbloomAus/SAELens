# CHANGELOG



## v4.1.1 (2024-11-06)

### Chore

* chore: Update training_a_sparse_autoencoder.ipynb (#358)

Changed &#34;She lived in a big, happy little girl.&#34; to &#34;She lived in a big, happy little town.&#34; ([`b8703fe`](https://github.com/jbloomAus/SAELens/commit/b8703fe8332b6eb6c49df778f6550c59d2276458))

### Fix

* fix: load the same config from_pretrained and get_sae_config (#361)

* fix: load the same config from_pretrained and get_sae_config

* merge neuronpedia_id into get_sae_config

* fixing test ([`8e09458`](https://github.com/jbloomAus/SAELens/commit/8e094581c4772e33ec4577349ed0d02c6c90ed27))


## v4.1.0 (2024-11-03)

### Feature

* feat: Support training JumpReLU SAEs (#352)

* adds JumpReLU logic to TrainingSAE

* adds unit tests for JumpReLU

* changes classes to match tutorial

* replaces bandwidth constant with param

* re-add logic to JumpReLU logic to TrainingSAE

* adds TrainingSAE.save_model()

* changes threshold to match paper

* add tests for TrainingSAE when archicture is jumprelu

* adds test for SAE.load_from_pretrained() for JumpReLU

* removes code causing test to fail

* renames initial_threshold to threshold

* removes setattr()

* adds test for TrainingSAE.save_model()

* renames threshold to jumprelu_init_threshold

* adds jumprelu_bandwidth

* removes default value for jumprelu_init_threshold downstream

* replaces zero tensor with None in Step.backward()

* adds jumprelu to architecture type ([`0b56d03`](https://github.com/jbloomAus/SAELens/commit/0b56d035ce0fa12722d62cc1bc559bd4fd35e9f3))


## v4.0.10 (2024-10-30)

### Fix

* fix: normalize decoder bias in fold_norm_scaling_factor (#355)

* WIP: fix fold_norm_scaling

* fixing test ([`6951e74`](https://github.com/jbloomAus/SAELens/commit/6951e7437f0bf9a33727c2929982917d9f51e7d2))


## v4.0.9 (2024-10-24)

### Fix

* fix: typo in layer 12 YAML ([`d634c8b`](https://github.com/jbloomAus/SAELens/commit/d634c8b2e8665bc3156c46fc8b1b439e26c289c9))

### Unknown

* Merge pull request #349 from jbloomAus/np_id_fix_2

fix: use the correct layer for new gemma scope SAE sparsities ([`4c32de0`](https://github.com/jbloomAus/SAELens/commit/4c32de0de3efe9f35007df00c6b5aad102552150))


## v4.0.8 (2024-10-24)

### Fix

* fix: use the correct layer for new gemma scope SAE sparsities ([`a78b93e`](https://github.com/jbloomAus/SAELens/commit/a78b93e33ecfee5ff5e5b08cdf9076cdeabec573))

### Unknown

* Merge pull request #348 from jbloomAus/np_id_fix

fix: use the correct layer for new gemma scope SAE sparsities ([`1f6823a`](https://github.com/jbloomAus/SAELens/commit/1f6823a4881a26df18b7c23e2f3a29a8cc93bcf6))


## v4.0.7 (2024-10-23)

### Fix

* fix: Test JumpReLU/Gated SAE and fix sae forward with error term (#328)

* chore: adding tests a slight refactoring for SAE forward methods

* refactoring forward methods using a helper to avoid firing hooks

* rewording intermediate var

* use process_sae_in helper in training sae encode

* testing that sae.forward() with error term works with hooks

* cleaning up more unneeded device=cpu in tests ([`ae345b6`](https://github.com/jbloomAus/SAELens/commit/ae345b642ceeeb87851af1ffa180979cc3670c9b))


## v4.0.6 (2024-10-23)

### Chore

* chore: Add tests for evals (#346)

* add unit tests for untested functions

* adds test to increase coverage

* fixes typo ([`06594f9`](https://github.com/jbloomAus/SAELens/commit/06594f97cb56f9f013ae420c147db09300ef9be4))

### Fix

* fix: pass device through to SAEConfigLoadOptions properly (#347) ([`531b1c7`](https://github.com/jbloomAus/SAELens/commit/531b1c7cac6971a3a5e9178710e9b3773d415a00))


## v4.0.5 (2024-10-22)

### Fix

* fix: last NP id fix, hopefully ([`a470460`](https://github.com/jbloomAus/SAELens/commit/a47046055664ee7a42322c41e1067d286e475d9b))

### Unknown

* Merge pull request #345 from jbloomAus/last_np_id_fix

fix: last NP id fix, hopefully ([`d5a7906`](https://github.com/jbloomAus/SAELens/commit/d5a7906a676d688c44ff71245c1f3f3ba8419b1a))


## v4.0.4 (2024-10-22)

### Fix

* fix: np ids should contain model id ([`da5c622`](https://github.com/jbloomAus/SAELens/commit/da5c6224cd91027367cc3842be885a4f6ef5af78))

### Unknown

* Merge pull request #344 from jbloomAus/fix_np_ids_again

fix: np ids should contain model id ([`88d9a0b`](https://github.com/jbloomAus/SAELens/commit/88d9a0baec99596f0fb346a26c257758c9187662))


## v4.0.3 (2024-10-22)

### Fix

* fix: fix duplicate np ids ([`bcaf802`](https://github.com/jbloomAus/SAELens/commit/bcaf80291dc4ef5247df5c1223ee53bb38085da6))

* fix: yaml was missing some gemmascope np ids, update np id formats ([`3f43590`](https://github.com/jbloomAus/SAELens/commit/3f435903ff436ce93e78b444e8fd6bb63636ebf0))

### Unknown

* Merge pull request #343 from jbloomAus/fix_duplicate_np_ids

fix: fix duplicate np ids ([`4060d07`](https://github.com/jbloomAus/SAELens/commit/4060d0705903f5eb6c991ae9601a986af7c9c48d))

* Merge pull request #342 from jbloomAus/fix_yaml_missing_gemmascope_and_np_ids

fix: yaml was missing some gemmascope np ids, update np id formats ([`db2fa5f`](https://github.com/jbloomAus/SAELens/commit/db2fa5fd0e462c48b25005a9b49010cc6a962cce))


## v4.0.2 (2024-10-22)

### Fix

* fix: previous saebench yaml fixes were incomplete for pythia-70m-deduped ([`b0adf2d`](https://github.com/jbloomAus/SAELens/commit/b0adf2ddc39cd4ffd3bf9e4fa4eda8505e2ce17b))

### Unknown

* Merge pull request #341 from jbloomAus/fix_pythia70md_again

fix: previous saebench yaml fixes were incomplete for pythia-70m-deduped ([`72e3ef4`](https://github.com/jbloomAus/SAELens/commit/72e3ef4488f517e0992ff2648cb270fd2c4454ee))


## v4.0.1 (2024-10-20)

### Chore

* chore: reduce test space usage in CI (#336)

* chore: reduce test space usage in CI

* busting caches

* try reducing sizes further

* try using smaller datasets where possible

* tokenizing a super tiny dataset for tests ([`36e1d86`](https://github.com/jbloomAus/SAELens/commit/36e1d8662329981772b5b205f1a811b35e7f1d50))

### Fix

* fix: changes dtype default value in read_sae_from_disk() (#340) ([`5820585`](https://github.com/jbloomAus/SAELens/commit/5820585c33d880391fbdeb3faa69771889663f25))

### Unknown

* Merge pull request #339 from jbloomAus/fix/saeb-bench-model-names

updated SAE Bench pythia model names (and loader device cfg) ([`2057455`](https://github.com/jbloomAus/SAELens/commit/20574550118726d9e1c54f7ae1204e5e0f9daa9e))

* Merge pull request #324 from jbloomAus/improving-evals

chore: Misc basic evals improvements (eg: consistent activation heuristic, cli args) ([`10f4773`](https://github.com/jbloomAus/SAELens/commit/10f4773bae2f18c37af72b3192403903e9425caa))

* Updated tests to be correct ([`d1b4f5d`](https://github.com/jbloomAus/SAELens/commit/d1b4f5d973e4c6715e404bf3e084102ff94fe24a))

* Organized basic eval metrics and eliminated NaNs ([`f6be1a6`](https://github.com/jbloomAus/SAELens/commit/f6be1a67725be22b64ad1b926e0c21a0556d0348))

* format with updated env ([`97622b5`](https://github.com/jbloomAus/SAELens/commit/97622b5e335fd7e866e908fc247ca1ec3c4c2bf8))

* format ([`31b2e9d`](https://github.com/jbloomAus/SAELens/commit/31b2e9df67d9c7f769cb99a592a1c02595421e52))

* fix other test from rebase ([`530a426`](https://github.com/jbloomAus/SAELens/commit/530a426424c0eddf6bbd961808753de05ba26b69))

* fix tests ([`92019ac`](https://github.com/jbloomAus/SAELens/commit/92019acc03356c3e8435cbb2ad59242b06c648cf))

* set type to int for ctx lens ([`e4b5be6`](https://github.com/jbloomAus/SAELens/commit/e4b5be6464e2959411e31611c87d72215d5a782a))

* update evaluating SAEs tutorial ([`d2bebbc`](https://github.com/jbloomAus/SAELens/commit/d2bebbc589482d782ddc11a11c9698a7bd1d29d1))

* moving SAE to correct device ([`b881b05`](https://github.com/jbloomAus/SAELens/commit/b881b05194e88ab831eafb71a4d75947c1711ddf))

* Added dataset_trust_remote_code arg ([`53dbde6`](https://github.com/jbloomAus/SAELens/commit/53dbde6ee1520494e12009bb1a39fe603d1a55de))

* Added trust_remote_code arg ([`b738924`](https://github.com/jbloomAus/SAELens/commit/b738924c2122ce6abece6ac584c0733af4e84a6e))

* Updated eval config explanations ([`d439927`](https://github.com/jbloomAus/SAELens/commit/d439927db290c1a4dc8c84b9fa9c36db559e997a))

* Added updated plots for feature metrics ([`7a4ce2d`](https://github.com/jbloomAus/SAELens/commit/7a4ce2de34d351e9beaee3a73edaad4846dcfe1c))

* Initial draft of evals tutorial ([`42309c8`](https://github.com/jbloomAus/SAELens/commit/42309c872173666518eb89c82d4313f43cc2d625))

* first pass evals notebook ([`8005ff9`](https://github.com/jbloomAus/SAELens/commit/8005ff91e6d722b412cd53505c956991a362e6d2))

* add verbose mode ([`15f1b59`](https://github.com/jbloomAus/SAELens/commit/15f1b59ec6506eeea7d0c5695681aaa7fe2b6244))

* add more cli args ([`bc17fa5`](https://github.com/jbloomAus/SAELens/commit/bc17fa502c67fd94609ecdafd46bdd048e93db3a))

* fix featurewise weight based metric type ([`d504a96`](https://github.com/jbloomAus/SAELens/commit/d504a963e37afe4d3f50fcb0daf9227027b3a97f))

* add featurewise weight based metrics ([`ed365c4`](https://github.com/jbloomAus/SAELens/commit/ed365c4889dd4e0ed39eff9e386472c4edbe8bc7))

* fix during training eval config ([`333d71c`](https://github.com/jbloomAus/SAELens/commit/333d71cb7b187a21fb1b6c5fd384c8a6dbd72aca))

* add feature density histogram to evals + consistent activation heuristic ([`9341398`](https://github.com/jbloomAus/SAELens/commit/9341398f2e16406acb166e09489909b474b57d17))

* keep track of tokens used seperately ([`c168c2b`](https://github.com/jbloomAus/SAELens/commit/c168c2b88af9102192848869a2b7fc879313cf5d))

* use evals code in CI ([`87601ba`](https://github.com/jbloomAus/SAELens/commit/87601ba7ee0b0e9a0626b349551401cda34e3daa))

* add cossim and relative reconstruction bias ([`c028072`](https://github.com/jbloomAus/SAELens/commit/c028072c2d6f0321f73950142e4cf3657c1d7d5d))

* add sae_lens version ([`a3123c8`](https://github.com/jbloomAus/SAELens/commit/a3123c84d767bf78f3d49d8d0ae16d2e47ae5989))

* remove redundant string in keys ([`a2dd2e0`](https://github.com/jbloomAus/SAELens/commit/a2dd2e029138cf22fdebb8bd20926cce9b1f3a89))

* updated SAE Bench pythia model names (and loader device cfg) ([`2078eac`](https://github.com/jbloomAus/SAELens/commit/2078eaca0811356ecef1e84e5d9446032dcd59f1))


## v4.0.0 (2024-10-15)

### Breaking

* feat: Use hf datasets for activation store (#321)

BREAKING CHANGE: use huggingface for cached activations

* refactored load activations into new function

* activation store

* cache activation runner

* formatting and get total_size

* doing tests

* cleaner load buffer

* cleaner load dataset

* cleanup cache activation runner

* add comments

* failing test

* update

* fixed! set shuffle param in get_buffer

* fixed linting

* added more tests

* refactor tests &amp; cleanup

* format config.py

* added hook name mismatch test

* set deperacted to -1

* fix tempshards test

* update test name

* add benchmark: safetensors vs dataset

* added stop iteration at end of dataset

* don&#39;t double save

* add push to hub

* fix save

* fomatting

* comments

* removed unecessary write

* cleanup pushing to hub, same as PretokenizeRunnerConfig

* use num_buffers by default (rather than 64)

* update comment

* shuffle and save to disk

* cleanup error checking

* added cfg info

* delete to iterable

* formatting

* delete deprectated params

* set format of dataset

* fix tests

* delete shuffle args

* fix test

* made dynamic dataset creation shorter

* removed print statements

* showcase hf_repo_id in docs

---------

Co-authored-by: Tom Pollak &lt;tompollak100@gmail.com&gt;
Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`ff335f0`](https://github.com/jbloomAus/SAELens/commit/ff335f0a5dad1b2348854a3f20254f9de7310d83))

### Feature

* feat: support othellogpt in SAELens (#317)

* support seqpos slicing

* add basic tests, ensure it&#39;s in the SAE config

* format

* fix tests

* fix tests 2

* fix: Changing the activations store to handle context sizes smaller than dataset lengths for tokenized datasets.

* fix: Found bug which allowed for negative context lengths. Removed the bug

* Update pytest to test new logic for context size of tokenized dataset

* Reformat code to pass CI tests

* Add warning for when context_size is smaller than the dataset context_size

* feat: adding support for start and end position offsets for token sequences

* Add start_pos_offset and end_pos_offset to the SAERunnerConfig

* Add tests for start_pos_offset and end_pos_offset in the LanguageModelSAERunnerConfig

* feat: start and end position offset support for SAELens.

* Add test for CacheActivationsRunnerConfig with start and end pos offset

* Test cache activation runner wtih valid start and end pos offset

* feat: Enabling loading of start and end pos offset from saes. Adding
tests for this

* fix: Renaming variables and a test

* adds test for position offests for saes

* reformats files with black

* Add start and end pos offset to the base sae dict

* fix test for sae training runner config with position offsets

* add a benchmark test to train an SAE on OthelloGPT

* Remove double import from typing

* change dead_feature_window to int

* remove print statements from test file

* Rebase on seqpos tuple implementation and remove start/end pos offset

* Reword docstring for seqpos to be clearer.

* Added script to train an SAE on othelloGPT

---------

Co-authored-by: callummcdougall &lt;cal.s.mcdougall@gmail.com&gt;
Co-authored-by: jbloomAus &lt;jbloomaus@gmail.com&gt;
Co-authored-by: liuman &lt;zhenninghimme@gmail.com&gt; ([`7047f87`](https://github.com/jbloomAus/SAELens/commit/7047f877979952836e6778827248918818716b96))

* feat: add get_sae_config() function (#331)

* extracts code to get_connor_rob_hook_z_config()

* extracts code into get_dictionary_learning_config_1()

* extract repeated lines to above conditions

* fixes incorrect function name

* extracts code in generate_sae_table.py to function

* removes unnecessary update()

* replaces calls to specific loaders with get_sae_config()

* replaces **kwargs with dataclass

* refactors attribute access

* renames SAEConfigParams to SAEConfigLoadOptions

* gets rid of indent

* replaces repo_id, folder_name with release, sae_id

* extracts to get_conversion_loader_name()

* extracts if-else to dict

* move blocks to sensible place

* extracts to get_repo_id_and_folder_name()

* adds tests for get_repo_id_and_folder_name()

* adds tests for get_sae_config()

* removes mocking

* fixes test

* removes unused import ([`d451b1d`](https://github.com/jbloomAus/SAELens/commit/d451b1dbad5ebd273bd69bbebfff89c6b947634e))

### Fix

* fix: force new build ([`26fead6`](https://github.com/jbloomAus/SAELens/commit/26fead6ce86e7595d2a78e0a2a9fa5c2fe6961b3))

* fix: add neuronpedia links for gemmascope 32plus ([`1087f19`](https://github.com/jbloomAus/SAELens/commit/1087f1999d04ee281be8cfda9832a6376151d0d1))

### Unknown

* Merge pull request #332 from jbloomAus/pretrained_yaml_gs_32plus

fix: add neuronpedia links for gemmascope 32plus ([`42ba557`](https://github.com/jbloomAus/SAELens/commit/42ba5575f1e786a860f073943daf817e882ba76c))

* Add Curt to citation (#329) ([`24b8560`](https://github.com/jbloomAus/SAELens/commit/24b8560c9272530c0090c6bb945653cacc68b7f5))


## v3.23.4 (2024-10-10)

### Fix

* fix: add-neuronpedia-ids-correct-gemma-2-2b-model-name (#327) ([`6ed1400`](https://github.com/jbloomAus/SAELens/commit/6ed1400709a20e5813324c7cd2c4bfb62a881fe6))


## v3.23.3 (2024-10-08)

### Fix

* fix: properly manage pyproject.toml version (#325) ([`432df87`](https://github.com/jbloomAus/SAELens/commit/432df87d97c5e5b4fd62ba835a01acd4ffd1758a))


## v3.23.2 (2024-10-07)

### Chore

* chore: adds black-jupyter to dependencies (#318)

* adds black-jupyter to dependencies

* formats notebooks ([`ed5d791`](https://github.com/jbloomAus/SAELens/commit/ed5d79148fa161c2e69e959f4ea7a8d9d3a87290))

### Fix

* fix: `hook_sae_acts_post` for Gated models should be post-masking (#322)

* first commit

* formatting ([`5e70edc`](https://github.com/jbloomAus/SAELens/commit/5e70edc0a167d895931cf85a4b068970599128b8))


## v3.23.1 (2024-10-03)

### Chore

* chore: Change print warning messages to warnings.warn messages in activations-store (#314) ([`606f464`](https://github.com/jbloomAus/SAELens/commit/606f4648db6dd5fdee377d5c9e6a5f54a4a848b4))

### Fix

* fix: Correctly load SAE Bench TopK SAEs (#308) ([`4fb5bbe`](https://github.com/jbloomAus/SAELens/commit/4fb5bbe4f066ede1f0bc13bf0146dea2ffd29ea9))


## v3.23.0 (2024-10-01)

### Chore

* chore: deletes print() in unit tests (#306) ([`7d9fe10`](https://github.com/jbloomAus/SAELens/commit/7d9fe10d4173ea90030866bf7621151fdbf7d24c))

### Feature

* feat: allow smaller context size of a tokenized dataset (#310)

* fix: Changing the activations store to handle context sizes smaller than dataset lengths for tokenized datasets.

* fix: Found bug which allowed for negative context lengths. Removed the bug

* Update pytest to test new logic for context size of tokenized dataset

* Reformat code to pass CI tests

* Add warning for when context_size is smaller than the dataset context_size

---------

Co-authored-by: liuman &lt;zhenninghimme@gmail.com&gt; ([`f04c0f9`](https://github.com/jbloomAus/SAELens/commit/f04c0f92df19922f38d41bb9e7dd623ff7818073))

### Fix

* fix: Add entity argument to wandb.init (#309) ([`305e576`](https://github.com/jbloomAus/SAELens/commit/305e576290b1e82412d1c0376e63084271cce3d3))


## v3.22.2 (2024-09-27)

### Fix

* fix: force new build for adding llama-3-8b-it ([`ed23343`](https://github.com/jbloomAus/SAELens/commit/ed2334382c87d6245bb826d92aee5693ced48eb8))

### Unknown

* Merge pull request #307 from jbloomAus/feature/llama-3-8b-it

Feature/llama 3 8b it ([`f786712`](https://github.com/jbloomAus/SAELens/commit/f7867128818ec58922840ff59e919c6a4b22f614))

* corrected entry ([`0415c93`](https://github.com/jbloomAus/SAELens/commit/0415c93584cbc8156154d4e8ace4ec85cd85c34b))

* added support through yaml ([`57e4a52`](https://github.com/jbloomAus/SAELens/commit/57e4a529e311317053458ebe53437c5a931c15f8))


## v3.22.1 (2024-09-26)

### Chore

* chore: delete dashboard_runner.py (#303) ([`df0aba7`](https://github.com/jbloomAus/SAELens/commit/df0aba7e89efa1bcfa049a1053c3e29d0b111258))

### Fix

* fix: fixing canrager SAEs in SAEs table docs (#304) ([`54cfc67`](https://github.com/jbloomAus/SAELens/commit/54cfc67796ccec25e6157a1c66a06d260a63902c))


## v3.22.0 (2024-09-25)

### Feature

* feat: Add value error if both d sae and expansion factor set (#301)

* adds ValueError if both d_sae and expansion_factor set

* renames class

* removes commented out line ([`999ffe8`](https://github.com/jbloomAus/SAELens/commit/999ffe80c263d58232e2a9c03d1223695c08b8ce))


## v3.21.1 (2024-09-23)

### Fix

* fix: log-spaced-checkpoints-instead (#300) ([`cdc64c1`](https://github.com/jbloomAus/SAELens/commit/cdc64c137fd6da5c201c33740d64a2e4514ab6b2))


## v3.21.0 (2024-09-23)

### Feature

* feat: Add experimental Gemma Scope embedding SAEs (#299)

* add experimental embedding gemmascope SAEs

* format and lint ([`bb9ebbc`](https://github.com/jbloomAus/SAELens/commit/bb9ebbc72ed70b2485a581f86fee50645549f048))

### Unknown

* Fix gated forward functions (#295)

* support seqpos slicing

* fix forward functions for gated

* remove seqpos changes

* fix formatting (remove my changes)

* format

---------

Co-authored-by: jbloomAus &lt;jbloomaus@gmail.com&gt; ([`a708220`](https://github.com/jbloomAus/SAELens/commit/a70822074c94c3bafe8502b54bc6e553483d08e3))


## v3.20.5 (2024-09-20)

### Fix

* fix: removing missing layer 11, 16k, l0=79 sae (#293)

Thanks! ([`e20e21f`](https://github.com/jbloomAus/SAELens/commit/e20e21f7942f889daf78d88a59dc23dda7d3f0e8))


## v3.20.4 (2024-09-18)

### Chore

* chore: Update README.md (#292) ([`d44c7c2`](https://github.com/jbloomAus/SAELens/commit/d44c7c21396fbfcf4a4a2c024987c810221c9cf4))

### Fix

* fix: Fix imports from huggingface_hub.utils.errors package (#296)

* Fix imports from huggingface_hub.utils.errors package

* Load huggingface error classes from huggingface_hub.utils ([`9d8ba77`](https://github.com/jbloomAus/SAELens/commit/9d8ba7716650fa86f8abd9af8587aba03f033e63))


## v3.20.3 (2024-09-13)

### Fix

* fix: Improve error message for Gemma Scope non-canonical ID not found (#288)

* Update sae.py as a nicer Gemma Scope error encouraging canonical

* Update sae.py

* Update sae.py

* format

---------

Co-authored-by: jbloomAus &lt;jbloomaus@gmail.com&gt; ([`9d34598`](https://github.com/jbloomAus/SAELens/commit/9d34598385cc8ae023dad7f21f3fdc490a0f981d))


## v3.20.2 (2024-09-13)

### Fix

* fix: Update README.md (#290) ([`1d1ac1e`](https://github.com/jbloomAus/SAELens/commit/1d1ac1e4c811f6062c1c4eb777d790a86957a082))


## v3.20.1 (2024-09-13)

### Fix

* fix: neuronpedia oai v5 sae ids ([`ffec7ed`](https://github.com/jbloomAus/SAELens/commit/ffec7ed2ee015846839c3ba128f48fe7edbe53b6))

### Unknown

* Merge pull request #291 from jbloomAus/fix_np_oai_ids

fix: neuronpedia oai v5 sae ids ([`253191e`](https://github.com/jbloomAus/SAELens/commit/253191ee98e2a16637fe3f70183e81b8bd0b7705))

* fix the ids for att&#34; ([`9e81c1c`](https://github.com/jbloomAus/SAELens/commit/9e81c1c216a599fdc80ac061895457e5eddce8fe))


## v3.20.0 (2024-09-12)

### Feature

* feat: Add SAE Bench SAEs (#285)

* add ignore tokens in evals

* remove accidental hard coding

* fix mse

* extract sae filtering code

* add sae_bench saes

* use from pretrained no processing by default

* use open web text by default

* add estimate of number of SAEs print statements

* add unit tests

* type fix ([`680c52b`](https://github.com/jbloomAus/SAELens/commit/680c52b95abfb039687f5328c52b1b41f8c3e05d))


## v3.19.4 (2024-09-05)

### Fix

* fix: add OAI mid SAEs for neuronpedia ([`a9cb852`](https://github.com/jbloomAus/SAELens/commit/a9cb852ae89256a8912e260e9c1dcaf7eaa8bef4))

* fix: Gemma Scope 9b IT ids for Neuronpedia ([`dcafbff`](https://github.com/jbloomAus/SAELens/commit/dcafbffc15bdb46b77b7277f751cb915833bedf6))

### Unknown

* Merge pull request #282 from jbloomAus/oai_mid_fix

fix: add OAI mid SAEs for neuronpedia ([`643f0c5`](https://github.com/jbloomAus/SAELens/commit/643f0c5cdf0d441a38d4c4279e42828fd6896aae))

* Merge pull request #281 from jbloomAus/fix_np_gemmascope_ids

fix: Gemma Scope 9b IT ids for Neuronpedia ([`e354918`](https://github.com/jbloomAus/SAELens/commit/e354918e24f2e4df2a3a0b8a49833774801484d2))


## v3.19.3 (2024-09-04)

### Fix

* fix: more gemma scope canonical ids + a few canonical ids were off.  (#280)

* fix model name in config for it models

* add 9b non-standard sizes

* fix att 131k canonical that were off

* fix mlp 131k canonical that were off ([`aa3e733`](https://github.com/jbloomAus/SAELens/commit/aa3e733a352052b7b27b48b9617cea99099b4c0a))


## v3.19.2 (2024-09-04)

### Fix

* fix: centre writing weights defaults (#279)

* add model_from_pretrained_kwargs to SAE config

* default to using no processing when training SAEs

* add centre writing weights true as config override for some SAEs

* add warning about from pretrained kwargs

* fix saving of config by trainer

* fix: test ([`7c0d1f7`](https://github.com/jbloomAus/SAELens/commit/7c0d1f70e5169b1209fccf2280ebeb82b5e5917b))


## v3.19.1 (2024-09-04)

### Chore

* chore: Update usage of Neuronpedia explanations export (#267)

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`f100aed`](https://github.com/jbloomAus/SAELens/commit/f100aedfda14f11206b69f5e210dd5d6e31832d4))

### Fix

* fix: reset hooks before saes in tutorial (#278) ([`2c225fd`](https://github.com/jbloomAus/SAELens/commit/2c225fd1d5c35325f075c0580d65101990e50e94))

### Unknown

* updating howpublished url in docs (#270) ([`25d9ba4`](https://github.com/jbloomAus/SAELens/commit/25d9ba4fe5d7b9b347aafac38792b95379b8151c))


## v3.19.0 (2024-09-03)

### Chore

* chore: Cleanup basic tutorial (#271)

* saelens: remove unnecessary html outputsaving, to save some space

* saelens: update top comment on basic tutorial ([`396e66e`](https://github.com/jbloomAus/SAELens/commit/396e66eee1c699decdf71c52a12d65c793b1c994))

* chore: update get neuronpedia quicklist function in logit lens tutorial (#274) ([`5b819b5`](https://github.com/jbloomAus/SAELens/commit/5b819b52b6e3784bb571d39a337d28b57fc3c14e))

* chore: Corrected outdated code to call API (#269)

Co-authored-by: dhuynh95 &lt;daniel.huynh@mithrilsecurity.io&gt; ([`7b19adc`](https://github.com/jbloomAus/SAELens/commit/7b19adcbbc4a7413bc1fdeba1965b61764e54c1b))

* chore: updating mkdocs deps (#268)

* chore: updating mkdocs deps

* adding type: ignore to wandb calls ([`75d142f`](https://github.com/jbloomAus/SAELens/commit/75d142fd457d75bece89a16afeec782073cc2128))

### Feature

* feat: only log ghost grad if you&#39;ve enabled it (#272)

* saelens: improve log output to only include ghost grad logs if you&#39;re using them

* sae: update ghostgrad log tests ([`da05f08`](https://github.com/jbloomAus/SAELens/commit/da05f08c051e3c7c9256ab6b2cebb76f2a81cfbd))


## v3.18.2 (2024-08-25)

### Fix

* fix: gemma scope saes yml. 16k for Gemma 2 9b was missing entries.  (#266)

* add missing saes, 16k was missing for 9b att and mlp

* remove file name not needed ([`86c04ac`](https://github.com/jbloomAus/SAELens/commit/86c04acb7198c45d0fdbe59ea96dacbd88d4011a))


## v3.18.1 (2024-08-23)

### Chore

* chore: adding more metatadata to pyproject.toml for PyPI (#263) ([`5c2d391`](https://github.com/jbloomAus/SAELens/commit/5c2d391686e67516827a89a559a1ff5d56584f41))

### Fix

* fix: modify duplicate neuronpedia ids in config.yml, add test. (#265)

* fix duplicate ids

* fix test that had mistake ([`0555178`](https://github.com/jbloomAus/SAELens/commit/055517806cccc41be1de87495df20445fd7d6b18))


## v3.18.0 (2024-08-22)

### Feature

* feat: updated pretrained yml gemmascope and neuronpedia ids (#264) ([`a3cb00d`](https://github.com/jbloomAus/SAELens/commit/a3cb00dc792a139c89f7311e5d1fa3f5bd9e855f))


## v3.17.1 (2024-08-18)

### Fix

* fix: fix memory crash when batching huge samples (#262) ([`f0bec81`](https://github.com/jbloomAus/SAELens/commit/f0bec8164a34f0ea14bb63ea7917ab54c226aa3d))


## v3.17.0 (2024-08-16)

### Feature

* feat: add end-2-end saes from Braun et al to yaml (#261) ([`1d4eac1`](https://github.com/jbloomAus/SAELens/commit/1d4eac1dfde0bb8056779c0a7266e8455cb7fe4e))


## v3.16.0 (2024-08-15)

### Feature

* feat: make canonical saes for attn (#259) ([`ed2437b`](https://github.com/jbloomAus/SAELens/commit/ed2437bd9c435cbfaef6597a089a93608755c355))


## v3.15.0 (2024-08-14)

### Chore

* chore: updating slack link in docs (#255) ([`5c7595a`](https://github.com/jbloomAus/SAELens/commit/5c7595a8ef145d1dad65552a8e9abfedd11b3063))

### Feature

* feat: support uploading and loading arbitrary huggingface SAEs (#258) ([`5994827`](https://github.com/jbloomAus/SAELens/commit/599482768daf463ab41b9e2df4ddc2e05e5d0a45))

### Unknown

* Remove duplicate link (#256) ([`c40f1c5`](https://github.com/jbloomAus/SAELens/commit/c40f1c583d44aee96b27504a1da1e29b788da6cd))

* Update index.md (#257)

removes comment asking for table creation and links to it ([`1e185b3`](https://github.com/jbloomAus/SAELens/commit/1e185b3a266e0b6e011b0b5430a56463026149a8))

* Merge pull request #244 from jbloomAus/add_pythia_70m_saes

Added pythia-70m SAEs to yaml ([`022f1de`](https://github.com/jbloomAus/SAELens/commit/022f1deeec252d63990375432efde738cc92f3b1))

* Merge branch &#39;main&#39; into add_pythia_70m_saes ([`32901f2`](https://github.com/jbloomAus/SAELens/commit/32901f25f5d3a990661978d2d8f3e3693268b227))


## v3.14.0 (2024-08-05)

### Feature

* feat: GemmaScope SAEs + fix gemma-scope in docs (#254) ([`3da4cee`](https://github.com/jbloomAus/SAELens/commit/3da4ceea61a0c62c2c2fc9a4182d1b5409e91688))

### Unknown

* More complete set of Gemma Scope SAEs (#252)

* commit for posterity

* ignore pt files in home

* add canonical saes

* improve gemma 2 loader

* better error msg on wrong id

* handle config better

* handle hook z weirdness better

* add joseph / curt script

* add gemma scope saes

* format

* make linter happy ([`68de42c`](https://github.com/jbloomAus/SAELens/commit/68de42cb770952afe425725a710df96ab4e92b66))

* Updated dashes ([`7c7a271`](https://github.com/jbloomAus/SAELens/commit/7c7a271ef04c8a1984f765c11ba3774daa0f289c))

* Changed gemma repo to google ([`fa483f0`](https://github.com/jbloomAus/SAELens/commit/fa483f068032af8a9c642e71ccb27e813e9e1052))

* Fixed pretrained_saes.yaml Gemma 2 paths ([`920b77e`](https://github.com/jbloomAus/SAELens/commit/920b77eace6da6796f770c69814fc7e93f23cc39))

* Gemma2 2b saes (#251)

* Added support for Gemma 2

* minor fixes

* format

* remove faulty error raise

---------

Co-authored-by: jbloomAus &lt;jbloomaus@gmail.com&gt; ([`df273c4`](https://github.com/jbloomAus/SAELens/commit/df273c4bbcb66e74bf1864eed11973380393e82a))


## v3.13.1 (2024-07-31)

### Fix

* fix: update automated-interpretability dep to use newly released version (#247)

* fix: update automated-interpretability dep to use newly released version

* fixing / ignore optim typing errors ([`93b2ebe`](https://github.com/jbloomAus/SAELens/commit/93b2ebee8d1a4962faff55c0a522636b35d3f0bb))

### Unknown

* Tutorial 2.0 (#250)

* tutorial 2.0 draft

* minor changes

* Various additions to tutorial

* Added ablation

* better intro text

* improve content further

* fix steering

* fix ablation to be true ablation

* current tutorial

---------

Co-authored-by: curt-tigges &lt;ct@curttigges.com&gt; ([`fe27b7c`](https://github.com/jbloomAus/SAELens/commit/fe27b7c21b4e5452b6ef348d5190ec3c57a19754))

* Fix typo in readme (#249) ([`fe987f1`](https://github.com/jbloomAus/SAELens/commit/fe987f1d1b06a5f0116e390efc87a69f59e04473))

* Merge pull request #242 from jbloomAus/add_openai_gpt2_small_saes

Added OpenAI TopK SAEs to pretrained yaml ([`2c1cbc4`](https://github.com/jbloomAus/SAELens/commit/2c1cbc4d0a6c446bf62ac9f84760e3f041bc021e))

* Added pythia-70m SAEs to yaml ([`25fb167`](https://github.com/jbloomAus/SAELens/commit/25fb167d6df4d4fd58a0bbb750d3ff90f985b3b2))

* Neuronpedia API key is now in header, not in body (#243) ([`caacef1`](https://github.com/jbloomAus/SAELens/commit/caacef1d5e6f48ec18ead19d2e72be033417ebc9))

* Merge pull request #237 from jbloomAus/use_error_term_param

Use error term param ([`ac86d10`](https://github.com/jbloomAus/SAELens/commit/ac86d10f1321c5a5a5345c00dc688c2634d35f7f))

* Update pyproject.toml ([`4b032ab`](https://github.com/jbloomAus/SAELens/commit/4b032ab840e8f9e04041b74fd27b856aeda95359))

* Added OpenAI TopK SAEs to pretrained yaml ([`7463e9f`](https://github.com/jbloomAus/SAELens/commit/7463e9f3f04453cfb01a6099cb46b8680515c5b6))


## v3.13.0 (2024-07-18)

### Feature

* feat: validate that pretokenized dataset tokenizer matches model tokenizer (#215)

Co-authored-by: Joseph Bloom &lt;69127271+jbloomAus@users.noreply.github.com&gt; ([`c73b811`](https://github.com/jbloomAus/SAELens/commit/c73b81147dc9b47c9578fa24e1279bca7fc724af))

### Unknown

* add more bootleg gemma saes (#240)

* add more bootleg gemma saes

* removed unused import ([`22a0841`](https://github.com/jbloomAus/SAELens/commit/22a08416e9759fdefe52eca646cdca23a64d2049))


## v3.12.5 (2024-07-18)

### Fix

* fix: fixing bug with cfg loading for fine-tuning (#241) ([`5a88d2c`](https://github.com/jbloomAus/SAELens/commit/5a88d2c2f3086393a35e2b8cc5d356e161494e9e))

### Unknown

* Update deploy_docs.yml

Removed the Debug Info step that was causing issues. ([`71fd509`](https://github.com/jbloomAus/SAELens/commit/71fd509bf719481b2e89718cca53a39f9a113c80))

* More tests for the negative case ([`a0b0f54`](https://github.com/jbloomAus/SAELens/commit/a0b0f547df51e17568f92d15390c2356f05beed5))

* Upped version ([`845d5d7`](https://github.com/jbloomAus/SAELens/commit/845d5d75f3033ce325f32e2d3c4c9421e15f847f))

* Added tests ([`e5ff793`](https://github.com/jbloomAus/SAELens/commit/e5ff793f32c1f3d6644c97127754fbc64ec631b8))


## v3.12.4 (2024-07-17)

### Fix

* fix: Trainer eval config will now respect trainer config params (#238)

* Trainer eval config will now respect trainer config params

* Corrected toml version ([`5375505`](https://github.com/jbloomAus/SAELens/commit/5375505f86785c96157719f6540f5e0dba541fe6))

### Unknown

* Neuronpedia Autointerp/Explanation Improvements (#239)

* Neuronpedia autointerp API improvements: new API, new flags for save to disk and test key, fix bug with scoring disabled

* Ignore C901 ([`ba7d218`](https://github.com/jbloomAus/SAELens/commit/ba7d21815c6429e1b92b36ebc030278acdea8c5e))

* Fixed toml file ([`8211cac`](https://github.com/jbloomAus/SAELens/commit/8211cac570a7c8baf0dbcc61e6cf72d4635c05a3))

* Ensured that even detatched SAEs are returned to former state ([`90ac661`](https://github.com/jbloomAus/SAELens/commit/90ac66179a3bfec9324161bdd86f4f6245aa3791))

* Added use_error_term functionality to run_with_x functions ([`1531c1f`](https://github.com/jbloomAus/SAELens/commit/1531c1f72d21b8068ad78ae60ec08af9da1d3751))

* Added use_error_term to hooked sae transformer ([`d172e79`](https://github.com/jbloomAus/SAELens/commit/d172e792441bb8c233e50cf5c7af74b8c37b865f))

* Trainer will now fold and log estimated norm scaling factor (#229)

* Trainer will now fold and log estimated norm scaling factor after doing fit

* Updated tutorials to use SAEDashboard

* fix: sae hook location (#235)

* 3.12.2

Automatically generated by python-semantic-release

* fix: sae to method (#236)

* 3.12.3

Automatically generated by python-semantic-release

* Trainer will now fold and log estimated norm scaling factor after doing fit

* Added functionality to load and fold in precomputed scaling factors from the YAML directory

* Fixed toml

---------

Co-authored-by: Joseph Bloom &lt;69127271+jbloomAus@users.noreply.github.com&gt;
Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`8d38d96`](https://github.com/jbloomAus/SAELens/commit/8d38d96056802a8025c60d0296624878777a9159))

* Update README.md ([`172ad6a`](https://github.com/jbloomAus/SAELens/commit/172ad6a099504acc90c4c3d5c29a3084367f560f))


## v3.12.3 (2024-07-15)

### Fix

* fix: sae to method (#236) ([`4df78ea`](https://github.com/jbloomAus/SAELens/commit/4df78ea6b44affe0ee0b3f033f4d1836a7f4a3e9))


## v3.12.2 (2024-07-15)

### Fix

* fix: sae hook location (#235) ([`94ba11c`](https://github.com/jbloomAus/SAELens/commit/94ba11c13bdf96740e6217d8c87af47237f89ca5))

### Unknown

* Updated tutorials to use SAEDashboard ([`db89dbc`](https://github.com/jbloomAus/SAELens/commit/db89dbc813e5a64a7609165474e3c66e2b537925))

* Merge branch &#39;JoshEngels-Evals&#39; ([`fe66285`](https://github.com/jbloomAus/SAELens/commit/fe66285d75bf7b0617a8d0aedef010e74b972677))

* Removed redundant lines ([`29e3aa2`](https://github.com/jbloomAus/SAELens/commit/29e3aa2b1a5b808dc32c5fc8187f5d7be808c7e3))

* Merge branch &#39;Evals&#39; of https://github.com/JoshEngels/SAELens into JoshEngels-Evals ([`7b84053`](https://github.com/jbloomAus/SAELens/commit/7b84053027f9113ac3cf61946c5acc0367298e05))


## v3.12.1 (2024-07-11)

### Fix

* fix: force release of dtype_fix ([`bfe7feb`](https://github.com/jbloomAus/SAELens/commit/bfe7feb10d987f39a9e0d00f5595869a64de2b5b))

### Unknown

* Merge pull request #225 from jbloomAus/dtype_fix

fix: load_from_pretrained should not require a dtype nor default to float32 ([`71d9da8`](https://github.com/jbloomAus/SAELens/commit/71d9da80a0e48a56f8d2394aecb0383a5f2a0cc6))

* TrainingSAE should: 1) respect device override and 2) not default to float32 dtype, and instead default to the SAE&#39;s dtype ([`a4a1c46`](https://github.com/jbloomAus/SAELens/commit/a4a1c469fff7e3b0685ced4c0632808fc3690359))

* load_from_pretrained should not require a dtype nor default to float32 ([`a485dc0`](https://github.com/jbloomAus/SAELens/commit/a485dc00d45bd3a69add64ccdf0cb5697d8b77ad))

* Fix SAE failing to upload to wandb due to artifact name. (#224)

* Fix SAE artifact name.

* format

---------

Co-authored-by: Joseph Bloom &lt;jbloomaus@gmail.com&gt; ([`6ae4849`](https://github.com/jbloomAus/SAELens/commit/6ae4849bb1947b6c83fd438b9dd00a8172f1f4b8))


## v3.12.0 (2024-07-09)

### Feature

* feat: use TransformerLens 2 (#214)

* Updated pyproject.toml to use TL ^2.0, and to use fork of sae-vis that also uses TL ^2.0

* Removed reliance on sae-vis

* Removed neuronpedia tutorial

* Added error handling for view operation

* Corrected formatting ([`526e736`](https://github.com/jbloomAus/SAELens/commit/526e736b937f95333969c33c83d2500dacab43d7))

### Unknown

* Fix/allow device override (#221)

* Forced load_from_pretrained to respect device and dtype params

* Removed test file ([`697dd5f`](https://github.com/jbloomAus/SAELens/commit/697dd5f0911d145d6a7c956e29bbcf28cf9fee38))

* Fixed hooks for single head SAEs (#219)

* included zero-ablation-hook for single-head SAEs

* fixed a typo in single_head_replacement_hook ([`3bb4f73`](https://github.com/jbloomAus/SAELens/commit/3bb4f73933984278bd4fb0446ead630f6edad600))


## v3.11.2 (2024-07-08)

### Fix

* fix: rename encode_fn to encode and encode to encode_standard (#218) ([`8c09ec1`](https://github.com/jbloomAus/SAELens/commit/8c09ec1ff29bf3212a43b6d65cf5c88c5c318994))


## v3.11.1 (2024-07-08)

### Fix

* fix: avoid bfloat16 errors in training gated saes (#217) ([`1e48f86`](https://github.com/jbloomAus/SAELens/commit/1e48f8668537d6b20067cc5862b9805ece5e2a70))

### Unknown

* Update README.md ([`9adba61`](https://github.com/jbloomAus/SAELens/commit/9adba61b03bb90583bea64902bc900091992b0b4))

* Update deploy_docs.yml

Modified this file to install dependencies (using caching for efficiency). ([`e90d5c1`](https://github.com/jbloomAus/SAELens/commit/e90d5c195b02bd450bd7335b83bfb952a58fa29d))

* Adding type hint ([`5da6a13`](https://github.com/jbloomAus/SAELens/commit/5da6a13df27678d59e0d233b51dbf8758e190e34))

* Actually doing merge ([`c362e81`](https://github.com/jbloomAus/SAELens/commit/c362e813b56c4cc9c5045a79236f8287d23ae53f))

* Merge remote-tracking branch &#39;upstream/main&#39; into Evals ([`52780c0`](https://github.com/jbloomAus/SAELens/commit/52780c08fa116667f77d22ec3a63a4ead9c47348))

* Making changes in response to comments ([`cf4ebcd`](https://github.com/jbloomAus/SAELens/commit/cf4ebcdfe93d96270da2ed108f37a5c8d9d97c75))


## v3.11.0 (2024-07-04)

### Feature

* feat: make pretrained sae directory docs page (#213)

* make pretrained sae directory docs page

* type issue weirdness

* type issue weirdness ([`b8a99ab`](https://github.com/jbloomAus/SAELens/commit/b8a99ab4dfe8f7790a3b15f41e351fbc3b82f1ab))


## v3.10.0 (2024-07-04)

### Feature

* feat: make activations_store re start the dataset when it runs out (#207)

* make activations_store re start the dataset when it runs out

* remove misleading comments

* allow StopIteration to bubble up where appropriate

* add test to ensure that stopiteration is raised

* formatting

* more formatting

* format tweak so we can re-try ci

* add deps back ([`91f4850`](https://github.com/jbloomAus/SAELens/commit/91f48502c39cd573d5f28aba2f3295c7694112e6))

* feat: allow models to be passed in as overrides (#210) ([`dd95996`](https://github.com/jbloomAus/SAELens/commit/dd95996efaa46c779b85ead9e52a8342869cfc24))

### Fix

* fix: Activation store factor unscaling fold fix (#212)

* add unscaling to evals

* fix act norm unscaling missing

* improved variance explained, still off for that prompt

* format

* why suddenly a typingerror and only in CI? ([`1db84b5`](https://github.com/jbloomAus/SAELens/commit/1db84b5ca4ab82fae9edbe98c1e9a563ed1eb3c9))


## v3.9.2 (2024-07-03)

### Fix

* fix: Gated SAE Note Loading (#211)

* fix: add tests, make pass

* not in ([`b083feb`](https://github.com/jbloomAus/SAELens/commit/b083feb5ffb5b7f45669403786c3c7593aa1d3ba))

### Unknown

* SAETrainingRunner takes optional HFDataset (#206)

* SAETrainingRunner takes optional HFDataset

* more explicit errors when the buffer is too large for the dataset

* format

* add warnings when a new dataset is added

* replace default dataset with empty string

* remove valueerror ([`2c8fb6a`](https://github.com/jbloomAus/SAELens/commit/2c8fb6aeed214ff47dccfe427eb2881aca4e6808))


## v3.9.1 (2024-07-01)

### Fix

* fix: pin typing-extensions version (#205) ([`3f0e4fe`](https://github.com/jbloomAus/SAELens/commit/3f0e4fe9e1a353e8b9563567919734af662ab69d))


## v3.9.0 (2024-07-01)

### Feature

* feat: OpenAI TopK SAEs for residual stream of GPT2 Small (#201)

* streamlit app

* feat:  basic top-k support + oai gpt2small saes

* fix merge mistake ([`06c4302`](https://github.com/jbloomAus/SAELens/commit/06c4302eedaaa4ba95686ab6b9a49fed4652ead7))

### Unknown

* prevent context size mismatch error (#200) ([`76389ac`](https://github.com/jbloomAus/SAELens/commit/76389ac007b77ed639035bdaee2e7587a949a9fc))

* point gpt2 dataset path to apollo-research/monology-pile (#199) ([`d3eb427`](https://github.com/jbloomAus/SAELens/commit/d3eb427c64765fc27950249d299791bfa17b6a73))


## v3.8.0 (2024-06-29)

### Feature

* feat: harmize activation store and pretokenize runner (#181)

* eat: harmize activation store and pretokenize runner

* reverting SAE cfg back to prepend_bos

* adding a benchmark test

* adding another test

* adding list of tokenized datasets to docs

* adding a warning message about lack of pre-tokenization, and linking to SAELens docs

* fixing tests after apollo deleted sae- dataset versions

* Update training_saes.md ([`2e6a3c3`](https://github.com/jbloomAus/SAELens/commit/2e6a3c3b72e0724b24dd8ed3803f3b80a17b77d5))

### Unknown

* Updating example commands ([`265687c`](https://github.com/jbloomAus/SAELens/commit/265687c09ba3c6ae090cf5a97e7f70251c0cf66c))

* Fixing test ([`389a159`](https://github.com/jbloomAus/SAELens/commit/389a15924345c17442937e98f45c8d2eb9c92b21))

* Adding script to evals.py ([`f9aa2dd`](https://github.com/jbloomAus/SAELens/commit/f9aa2ddd20c1f8c26b9181e685f04c7638511bc1))

* Moving file ([`4be5011`](https://github.com/jbloomAus/SAELens/commit/4be50115b8b2c43448557ee54ff8f0afe692d111))

* First round of evals ([`2476afb`](https://github.com/jbloomAus/SAELens/commit/2476afbffad41406840ebd5492c04acf90a0e62c))


## v3.7.0 (2024-06-25)

### Feature

* feat: new saes for gemma-2b-it and feature splitting on gpt2-small-layer-8 (#195) ([`5cfe382`](https://github.com/jbloomAus/SAELens/commit/5cfe382d43f028c2e4f4e7cb21a1c19abb5471d0))


## v3.6.0 (2024-06-25)

### Feature

* feat: Support Gated-SAEs (#188)

* Initial draft of encoder

* Second draft of Gated SAE implementation

* Added SFN loss implementation

* Latest modification of SFN loss training setup

* fix missing config use

* dont have special sfn loss

* add hooks and reshape

* sae error term not working, WIP

* make tests  pass

* add benchmark for gated

---------

Co-authored-by: Joseph Bloom &lt;jbloomaus@gmail.com&gt; ([`232c39c`](https://github.com/jbloomAus/SAELens/commit/232c39cea709ae9c4b68b204cf027fbaab385f64))

### Unknown

* fix hook z loader (#194) ([`cb30996`](https://github.com/jbloomAus/SAELens/commit/cb30996cf36c80bbf6f0fd529bd27262bbce13ce))


## v3.5.0 (2024-06-20)

### Feature

* feat: trigger release ([`1a4663b`](https://github.com/jbloomAus/SAELens/commit/1a4663b7eadb42682586697e72192346c66cf430))

### Unknown

*  Performance improvements + using multiple GPUs.  (#189)

* fix: no grads when filling cache

* trainer should put activations on sae device

* hack to allow sae device to be specific gpu when model is on multiple devices

* add some tests (not in CI, which check multiple GPU performance

* make formatter typer happy

* make sure SAE calls move data between devices as needed ([`400474e`](https://github.com/jbloomAus/SAELens/commit/400474eaf758d57ea4090fd08e84cbdd91d55cc4))


## v3.4.1 (2024-06-17)

### Fix

* fix: allow settings trust_remote_code for new huggingface version (#187)

* fix: allow settings trust_remote_code for new huggingface version

* default to True, not none

---------

Co-authored-by: jbloomAus &lt;jbloomaus@gmail.com&gt; ([`33a612d`](https://github.com/jbloomAus/SAELens/commit/33a612d7f694390a5a4596f7e15e0c51657634ba))


## v3.4.0 (2024-06-14)

### Feature

* feat: Adding Mistral SAEs (#178)

Note: normalize_activations is now a string and should be either &#39;none&#39;, &#39;expected_average_only_in&#39; (Anthropic April Update, not yet folded), &#39;constant_norm_rescale&#39; (Anthropic Feb update). 

* Adding code to load mistral saes

* Black formatting

* Removing library changes that allowed forward pass normalization

* feat: support feb update style norm scaling for mistral saes

* Adding code to load mistral saes

* Black formatting

* Removing library changes that allowed forward pass normalization

* Adding code to load mistral saes

* Black formatting

* Removing library changes that allowed forward pass normalization

* feat: support feb update style norm scaling for mistral saes

* remove accidental inclusion

---------
Co-authored-by: jbloomAus &lt;jbloomaus@gmail.com&gt; ([`227d208`](https://github.com/jbloomAus/SAELens/commit/227d2089f6fdadb54b5554056eb7721574608b58))

### Unknown

* Update README.md Slack Link Expired (this one shouldn&#39;t expire) ([`209696a`](https://github.com/jbloomAus/SAELens/commit/209696a4f74007559a650ad5357c4fd923205923))

* add expected perf for pretrained (#179)

Co-authored-by: jbloom-md &lt;joseph@massdynamics.com&gt; ([`10bd9c5`](https://github.com/jbloomAus/SAELens/commit/10bd9c58fd4d731bd453e49943d40f2ac01ff0fc))

* fix progress bar updates (#171) ([`4d92975`](https://github.com/jbloomAus/SAELens/commit/4d92975cc1aa8cd7485e73597fd52e13a0f8e44e))


## v3.3.0 (2024-06-10)

### Feature

* feat: updating docs and standardizing PretokenizeRunner export (#176) ([`03f071b`](https://github.com/jbloomAus/SAELens/commit/03f071b41e77ef07c8f8b892e52969337f5e94aa))

### Unknown

* add tutorial (#175) ([`8c67c23`](https://github.com/jbloomAus/SAELens/commit/8c67c2355211910bc5054ba9bc140e98424fa026))


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
