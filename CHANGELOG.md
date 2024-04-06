# CHANGELOG



## v0.0.0 (2024-04-06)

### Chore

* chore: adding more tests to ActivationsStore + light refactoring ([`cc9899c`](https://github.com/jbloomAus/mats_sae_training/commit/cc9899c7c7ecfa734bdb9260b80878092a0b3d4f))

* chore: running isort to fix imports ([`53853b9`](https://github.com/jbloomAus/mats_sae_training/commit/53853b94c2809fe7145c92b6e6e68035ca5f2fe9))

* chore: setting up pyright type checking and fixing typing errors ([`351995c`](https://github.com/jbloomAus/mats_sae_training/commit/351995c1e0735073dd08a7fb0344cf0991d4a5a2))

* chore: enable full flake8 default rules list ([`19886e2`](https://github.com/jbloomAus/mats_sae_training/commit/19886e28ed8358e787e1dad9b1dcfa5cf3dba7b5))

* chore: using poetry for dependency management ([`465e003`](https://github.com/jbloomAus/mats_sae_training/commit/465e00333b9fae79ce73c277e7e1018e863002ca))

* chore: removing .DS_Store files ([`32f09b6`](https://github.com/jbloomAus/mats_sae_training/commit/32f09b67e68b116ee546da16059b34f8b0db23fa))

### Unknown

* Merge pull request #66 from chanind/pypi

feat: setting up sae_lens package and auto-deploy with semantic-release ([`34633e8`](https://github.com/jbloomAus/mats_sae_training/commit/34633e892624808fe1f98048beb63939e610e3af))

* Merge branch &#39;main&#39; into pypi ([`3ce7f99`](https://github.com/jbloomAus/mats_sae_training/commit/3ce7f99fe7a6be2d87fb3fff484d6915419c0dcb))

* Merge pull request #60 from chanind/improve-config-typing

fixing config typing ([`b8fba4f`](https://github.com/jbloomAus/mats_sae_training/commit/b8fba4fce043a41e3b997fae2d067eafc6264268))

* setting up sae_lens package and auto-deploy with semantic-release ([`ba41f32`](https://github.com/jbloomAus/mats_sae_training/commit/ba41f327364901c40c6613a300e1ceaafe67e8fa))

* fixing config typing

switch to using explicit params for ActivationsStore config instead of RunnerConfig base class ([`9be3445`](https://github.com/jbloomAus/mats_sae_training/commit/9be344512f15c24f5106b1e43200ea49b8c33981))

* Merge pull request #65 from chanind/fix-forgotten-scheduler-opts

passing accidentally overlooked scheduler opts ([`773bc02`](https://github.com/jbloomAus/mats_sae_training/commit/773bc02e6a3b5f2a2c45e156c7f236485cdd79fe))

* passing accidentally overlooked scheduler opts ([`ad089b7`](https://github.com/jbloomAus/mats_sae_training/commit/ad089b7c0402d0cf4d1b8a37c597e9551b93b55e))

* Merge pull request #64 from chanind/lr-decay

adding lr_decay_steps and refactoring get_scheduler ([`c960d99`](https://github.com/jbloomAus/mats_sae_training/commit/c960d99ab9465fc86793c554d3ed810e9eb226ee))

* adding lr_decay_steps and refactoring get_scheduler ([`fd5448c`](https://github.com/jbloomAus/mats_sae_training/commit/fd5448c5a016a8134a881732f6c61cd7a37eab67))

* Merge pull request #53 from hijohnnylin/neuronpedia_runner

Generate and upload Neuronpedia artifacts ([`0b94f84`](https://github.com/jbloomAus/mats_sae_training/commit/0b94f845739884146faead573f484cd1bc6d8737))

* format ([`792c7cb`](https://github.com/jbloomAus/mats_sae_training/commit/792c7cb235faace99acfff2416a29a88363ca245))

* ignore type incorrectness in imported package ([`5fe83a9`](https://github.com/jbloomAus/mats_sae_training/commit/5fe83a9b09238201219b65df6b70ce074fed68eb))

* Merge pull request #63 from chanind/remove-eindex

removing unused eindex depencency ([`1ce44d7`](https://github.com/jbloomAus/mats_sae_training/commit/1ce44d7e3a3c51e013a68b0d7d82349da0ff8edf))

* removing unused eindex depencency ([`7cf991b`](https://github.com/jbloomAus/mats_sae_training/commit/7cf991ba534c6d9b5ea8a96cc1c4622183b211f9))

* Safe to_str_tokens, fix memory issues ([`901b888`](https://github.com/jbloomAus/mats_sae_training/commit/901b888c98df7de77626adfbc31e96298585958e))

* Allow starting neuronpedia generation at a specific batch numbe ([`85d8f57`](https://github.com/jbloomAus/mats_sae_training/commit/85d8f578839687159b0c0ba6b1879746a23f3288))

* FIX: Linting &#39;do not use except&#39; ([`ce3d40c`](https://github.com/jbloomAus/mats_sae_training/commit/ce3d40c61ceb1a1e7fd5c598371a3386f62a39d8))

* Fix vocab: Ċ should be line break. Also set left and right buffers ([`205b1c1`](https://github.com/jbloomAus/mats_sae_training/commit/205b1c18fba46ef0b4629846bc4dc224c34bc2f6))

* Merge ([`b159010`](https://github.com/jbloomAus/mats_sae_training/commit/b1590102bceb0d859d00a65fa68925e1a1ce2341))

* Update Neuronpedia Runner ([`885de27`](https://github.com/jbloomAus/mats_sae_training/commit/885de2770279ed4188042985a500c3c393c57ced))

* Merge pull request #58 from canrager/main

Make prepend BOS optional: Default True ([`48a07f9`](https://github.com/jbloomAus/mats_sae_training/commit/48a07f928ec569bba5502af379c789747e15a9e0))

* make tests pass with use_bos flag ([`618d4bb`](https://github.com/jbloomAus/mats_sae_training/commit/618d4bbdbaf8fed344886eb292e2f5423f98fbbc))

* Merge pull request #59 from chanind/fix-docs-deploy

attempting to fix docs deploy ([`cfafbe7`](https://github.com/jbloomAus/mats_sae_training/commit/cfafbe75026104d707c21d82efddd988bd1a3833))

* force docs push ([`3aa179d`](https://github.com/jbloomAus/mats_sae_training/commit/3aa179d350d3602d873ebd5fa585d0a9c7475c79))

* ignore type eror ([`e87198b`](https://github.com/jbloomAus/mats_sae_training/commit/e87198bba4c65a77be2e7c1b072e563c9497c9d0))

* format ([`67dfb46`](https://github.com/jbloomAus/mats_sae_training/commit/67dfb4685ba293db2266b1b4885964e2313104f9))

* attempting to fix docs deploy ([`cda8ece`](https://github.com/jbloomAus/mats_sae_training/commit/cda8eceb0a8d356fd4f4b461a2fb473ab0071a1d))

* Merge branch &#39;main&#39; of https://github.com/jbloomAus/mats_sae_training into main ([`8aadcd3`](https://github.com/jbloomAus/mats_sae_training/commit/8aadcd3171748ecf0ff759c67522d04ac0f3c3a1))

* add prepend bos flag ([`c0b29cc`](https://github.com/jbloomAus/mats_sae_training/commit/c0b29cc5037bfeea88888f2bb90ebd2ecdf2ed05))

* fix attn out on run evals ([`02fa90b`](https://github.com/jbloomAus/mats_sae_training/commit/02fa90be6c70cfc93f13262462d3ec4cc86641b8))

* Merge pull request #57 from chanind/optim-tests

Adding tests to get_scheduler ([`13c8085`](https://github.com/jbloomAus/mats_sae_training/commit/13c8085e1becb654e50fe69435a3f5814f0d2145))

* Merge pull request #56 from chanind/sae-tests

minor refactoring to SAE and adding tests ([`2c425ca`](https://github.com/jbloomAus/mats_sae_training/commit/2c425cabf7718c16eb62146972f389d2d5ff2a32))

* minor refactoring to SAE and adding tests ([`92a98dd`](https://github.com/jbloomAus/mats_sae_training/commit/92a98ddf8e6eba0b4cd4ac2e2350281f9ed2e2d0))

* adding tests to get_scheduler ([`3b7e173`](https://github.com/jbloomAus/mats_sae_training/commit/3b7e17368c7abfac2747f6fa8e3e03650914cfe5))

* Generate and upload Neuronpedia artifacts ([`b52e0e2`](https://github.com/jbloomAus/mats_sae_training/commit/b52e0e2c427a4b74bebff5d94961725148c404f0))

* Merge pull request #54 from jbloomAus/hook_z_suppourt

notional support, needs more thorough testing ([`277f35b`](https://github.com/jbloomAus/mats_sae_training/commit/277f35b35b2b8f6475b5b8ae9f15356b168e9f50))

* Merge pull request #55 from chanind/contributing-docs

adding a contribution guide to docs ([`8ac8f05`](https://github.com/jbloomAus/mats_sae_training/commit/8ac8f051c2686169f5143582ace162f7f337af50))

* adding a contribution guide to docs ([`693c5b3`](https://github.com/jbloomAus/mats_sae_training/commit/693c5b335527630dc246ebc55519008f87465913))

* notional support, needs more thorough testing ([`9585022`](https://github.com/jbloomAus/mats_sae_training/commit/9585022366677c66d6cdbedd82d10711d1001fef))

* Generate and upload Neuronpedia artifacts ([`4540268`](https://github.com/jbloomAus/mats_sae_training/commit/45402681bc4051bdbc160cd018c8c73ed4c7321f))

* Merge pull request #52 from hijohnnylin/fix_db_runner_assert

FIX: Don&#39;t check wandb assert if not using wandb ([`5c48811`](https://github.com/jbloomAus/mats_sae_training/commit/5c488116a0e5096e9e37d6d714dbd7a3835cb6fe))

* FIX: Don&#39;t check wandb assert if not using wandb ([`1adefda`](https://github.com/jbloomAus/mats_sae_training/commit/1adefda8f5c2f07c468ce63b10ba608faefc0877))

* add docs badge ([`f623ed1`](https://github.com/jbloomAus/mats_sae_training/commit/f623ed1b57f8753af3c995b68443b2459b258b42))

* try to get correct deployment ([`777dd6c`](https://github.com/jbloomAus/mats_sae_training/commit/777dd6cfc57b3a8cd5e1d5056298d253ccb49d9f))

* Merge pull request #51 from jbloomAus/mkdocs

Add Docs to the project. ([`d2ebbd7`](https://github.com/jbloomAus/mats_sae_training/commit/d2ebbd7b8ad1468da97f3e407934b8dd8e9e3f2f))

* mkdocs, test ([`9f14250`](https://github.com/jbloomAus/mats_sae_training/commit/9f142509da4e1a63ffc88a32a692dc5218aad96d))

* code cov ([`2ae6224`](https://github.com/jbloomAus/mats_sae_training/commit/2ae62243172088b309e40b3b4fca0ce43c08f19a))

* Merge pull request #48 from chanind/fix-sae-vis-version

Pin sae_vis to previous working version ([`3f8a30b`](https://github.com/jbloomAus/mats_sae_training/commit/3f8a30bf577efdaae717f1c6b930be35c9ad7883))

* fix suffix issue ([`209ba13`](https://github.com/jbloomAus/mats_sae_training/commit/209ba1324a775c30a0a1fc2941d0fdb46271ac53))

* pin sae_vis to previous working version ([`ae0002a`](https://github.com/jbloomAus/mats_sae_training/commit/ae0002a67838ce33474d2697dce20d47356f8abb))

* don&#39;t ignore changes to .github ([`35fdeec`](https://github.com/jbloomAus/mats_sae_training/commit/35fdeec3d3c1e31c2eb884060e7550a8d5fa9b22))

* add cov report ([`971d497`](https://github.com/jbloomAus/mats_sae_training/commit/971d4979193efe3f30cc3d580df648a413fd16b2))

* Merge pull request #40 from chanind/refactor-train-sae

Refactor train SAE and adding unit tests ([`5aa0b11`](https://github.com/jbloomAus/mats_sae_training/commit/5aa0b1121112024e44dc5a2d02e33e532b741ed9))

* Merge branch &#39;main&#39; into refactor-train-sae ([`0acdcb3`](https://github.com/jbloomAus/mats_sae_training/commit/0acdcb395aaa42e0344191e680a4f521af6d430c))

* Merge pull request #41 from jbloomAus/move_to_sae_vis

Move to sae vis ([`bcb9a52`](https://github.com/jbloomAus/mats_sae_training/commit/bcb9a5207339528d55d5c89640eec0b1a9c449cb))

* flake8 can ignore imports, we&#39;re using isort anyway ([`6b7ae72`](https://github.com/jbloomAus/mats_sae_training/commit/6b7ae72cd324929c7b53d4c1043ae422d8eec81d))

* format ([`af680e2`](https://github.com/jbloomAus/mats_sae_training/commit/af680e28e0978583758d7d584ff663f99482fdce))

* fix mps bug ([`e7b238f`](https://github.com/jbloomAus/mats_sae_training/commit/e7b238fd4813b76c647468949603b90fc37c1d8d))

* more tests ([`01978e6`](https://github.com/jbloomAus/mats_sae_training/commit/01978e6dd141c269c403a4e84f5de2fb26af6057))

* wip ([`4c03b3d`](https://github.com/jbloomAus/mats_sae_training/commit/4c03b3d3c4a89275d7b08d62d0c9ae18e0b865db))

* more tests ([`7c1cb6b`](https://github.com/jbloomAus/mats_sae_training/commit/7c1cb6b827e1e455e715cf9aa1f6c82e687eaf2e))

* testing that sparsity counts get updated correctly ([`5b5d653`](https://github.com/jbloomAus/mats_sae_training/commit/5b5d6538c90d11fcc040ff05085642066c799518))

* adding some unit tests to _train_step() ([`dbf3f01`](https://github.com/jbloomAus/mats_sae_training/commit/dbf3f01ffb5ad3c2f403c6a81f7eab5c4646100c))

* Merge branch &#39;main&#39; into refactor-train-sae ([`2d5ec98`](https://github.com/jbloomAus/mats_sae_training/commit/2d5ec98164dd4fde0c34c0e966e78b9a3848b842))

* Update README.md ([`d148b6a`](https://github.com/jbloomAus/mats_sae_training/commit/d148b6a2ad77e546b433cf2c3c5f9993f04228ba))

* Merge pull request #20 from chanind/activations_store_tests

chore: adding more tests to ActivationsStore + light refactoring ([`69dcf8e`](https://github.com/jbloomAus/mats_sae_training/commit/69dcf8e551415f6a3884a509ab4428818ad35936))

* Merge branch &#39;main&#39; into activations_store_tests ([`4896d0a`](https://github.com/jbloomAus/mats_sae_training/commit/4896d0a4a850551d4e7bc91358adc7b98b7fc705))

* refactoring train_sae_on_language_model.py into smaller functions ([`e75a15d`](https://github.com/jbloomAus/mats_sae_training/commit/e75a15d3acb4b02a435d70b25a67e8103c9196b3))

* suppourt apollo pretokenized datasets ([`e814054`](https://github.com/jbloomAus/mats_sae_training/commit/e814054a5693361ba51c9aed422b3a8b9d6d0fe6))

* handle saes saved before groups ([`5acd89b`](https://github.com/jbloomAus/mats_sae_training/commit/5acd89b02c02fe43fae5cea9d1dadeff41424069))

* typechecker ([`fa6cc49`](https://github.com/jbloomAus/mats_sae_training/commit/fa6cc4902f25843e1f7ae8f6b9841ecd4f074e9b))

* fix geom median bug ([`8d4a080`](https://github.com/jbloomAus/mats_sae_training/commit/8d4a080c404f3fa7c16e95e72d0be3b3a7812d0c))

* remove references to old code ([`861151f`](https://github.com/jbloomAus/mats_sae_training/commit/861151fd04c5fc8963396f655c08818bcff08d1c))

* remove old geom median code ([`05e0aca`](https://github.com/jbloomAus/mats_sae_training/commit/05e0acafc690b7f6e08f8a8c62989afb21b7d1f1))

* Merge pull request #22 from themachinefan/faster_geometric_median

Faster geometric median. ([`341c49a`](https://github.com/jbloomAus/mats_sae_training/commit/341c49ac91b8876b020c9a22c5ea3c0b7ea74b3e))

* makefile check type and types of geometric media ([`736bf83`](https://github.com/jbloomAus/mats_sae_training/commit/736bf83842f2e4be9baeee6ac12731b9b4834fc6))

* Merge pull request #21 from schmatz/fix-dashboard-image

Fix broken dashboard image on README ([`eb90cc9`](https://github.com/jbloomAus/mats_sae_training/commit/eb90cc9d835da6febe4a4c5830a134237e3705eb))

* Merge pull request #24 from neelnanda-io/add-post-link

Added link to AF post ([`39f8d3d`](https://github.com/jbloomAus/mats_sae_training/commit/39f8d3d8ff5659c9815f9a1ff3be7bf81370813c))

* Added link to AF post ([`f0da9ea`](https://github.com/jbloomAus/mats_sae_training/commit/f0da9ea8af61239c22de5b5778e9026f834fc929))

* formatting ([`0168612`](https://github.com/jbloomAus/mats_sae_training/commit/016861243e11bc1b0f0fe1414bcde6c0a5eec334))

* use device, don&#39;t use cuda if not there ([`20334cb`](https://github.com/jbloomAus/mats_sae_training/commit/20334cb26bee5ddcc2ae6632b18a1f1933ab50f0))

* format ([`ce49658`](https://github.com/jbloomAus/mats_sae_training/commit/ce496583e397a71cec0ae376f9920e981417c0e8))

*  fix tsea typing ([`449d90f`](https://github.com/jbloomAus/mats_sae_training/commit/449d90ffc54a548432da6aca3e47651bebfbf585))

* faster geometric median. Run geometric_median,py to test. ([`92cad26`](https://github.com/jbloomAus/mats_sae_training/commit/92cad26b4dc0fdd03d73a4a8dc9b1d5844308eb8))

* Fix dashboard image ([`6358862`](https://github.com/jbloomAus/mats_sae_training/commit/6358862f1f52961bd444feb1b22a7dde27235021))

*  fix incorrect code used to avoid typing issue ([`ed0b0ea`](https://github.com/jbloomAus/mats_sae_training/commit/ed0b0eafdd7424bbcc83c39d400928536bba3378))

* add nltk ([`bc7e276`](https://github.com/jbloomAus/mats_sae_training/commit/bc7e2766109fd775292d8af1226b188bdec8d807))

* ignore various typing issues ([`6972c00`](https://github.com/jbloomAus/mats_sae_training/commit/6972c0016f1ce5a9b20960727fc7778a7655f833))

* add babe package ([`481069e`](https://github.com/jbloomAus/mats_sae_training/commit/481069ef129c37bfec9cd1cd94d43800c45ae969))

* make formatter happy ([`612c7c7`](https://github.com/jbloomAus/mats_sae_training/commit/612c7c71ea5dfecc8f81bea3484976d82fb505ca))

* share scatter so can link ([`9f88dc3`](https://github.com/jbloomAus/mats_sae_training/commit/9f88dc31dc01ce5b10a3772c013d99b9f9d4cf13))

* add_analysis_files_for_post ([`e75323c`](https://github.com/jbloomAus/mats_sae_training/commit/e75323c13ffda589b8c71ed9de5c7c1adb3d6ce7))

* don&#39;t block on isort linting ([`3949a46`](https://github.com/jbloomAus/mats_sae_training/commit/3949a4641c810c74f8ed093e92fabd31d8e2d02c))

* formatting ([`951a320`](https://github.com/jbloomAus/mats_sae_training/commit/951a320fdcb4e01c95bf2bac7a4929676562e323))

* Update README.md ([`b2478c1`](https://github.com/jbloomAus/mats_sae_training/commit/b2478c14d98cbfc2b0c2b0984d251f3c92d05a41))

* Merge pull request #18 from chanind/type-checking

chore: setting up pyright type checking and fixing typing errors ([`bd5fc43`](https://github.com/jbloomAus/mats_sae_training/commit/bd5fc439e56fc1c9e0e6f9dd2613b77f57c763b2))

* Merge branch &#39;main&#39; into type-checking ([`57c4582`](https://github.com/jbloomAus/mats_sae_training/commit/57c4582c38906b9e6d887ceb25901de5734d76c9))

* Merge pull request #17 from Benw8888/sae_group_pr

SAE Group for sweeps PR ([`3e78bce`](https://github.com/jbloomAus/mats_sae_training/commit/3e78bce5c4bccbb9555d7878dc48adf7cfd823cd))

* Merge pull request #1 from chanind/sae_group_pr_isort_fix

chore: running isort to fix imports ([`dd24413`](https://github.com/jbloomAus/mats_sae_training/commit/dd24413749191fab0001f154a225eb991820a12b))

* black format ([`0ffcf21`](https://github.com/jbloomAus/mats_sae_training/commit/0ffcf21bbe73066cad152680255a25b73f98331c))

* fixed expansion factor sweep ([`749b8cf`](https://github.com/jbloomAus/mats_sae_training/commit/749b8cff1a4ccb130a9ff6be698b766ae7642140))

* remove tqdm from data loader, too noisy ([`de3b1a1`](https://github.com/jbloomAus/mats_sae_training/commit/de3b1a1fad3b176b2f547513374f0ff00b784975))

* fix tests ([`b3054b1`](https://github.com/jbloomAus/mats_sae_training/commit/b3054b1e7cc938bee874db95a7ebfd911417a84a))

* don&#39;t calculate geom median unless you need to ([`d31bc31`](https://github.com/jbloomAus/mats_sae_training/commit/d31bc31689c84382044db1a8663df2d352674d7a))

* add to method ([`b3f6dc6`](https://github.com/jbloomAus/mats_sae_training/commit/b3f6dc6882ea052f6fecb2234d8ff4f6212184d7))

* flake8 and black ([`ed8345a`](https://github.com/jbloomAus/mats_sae_training/commit/ed8345ae4ade1445e7182f5c7deeb876203b6524))

* flake8 linter changes ([`8e41e59`](https://github.com/jbloomAus/mats_sae_training/commit/8e41e599d67a7d8eda644321eda8d0481f813f15))

* Merge branch &#39;main&#39; into sae_group_pr ([`082c813`](https://github.com/jbloomAus/mats_sae_training/commit/082c8138045557a592f57f53256d251f76902b4f))

* Delete evaluating.ipynb ([`d3cafa3`](https://github.com/jbloomAus/mats_sae_training/commit/d3cafa3f28029f4d12285a810df29e7de90311c2))

* Delete activation_storing.py ([`fa82992`](https://github.com/jbloomAus/mats_sae_training/commit/fa829924b0b5caab9c7653876a399732277c5d4e))

* Delete lp_sae_training.py ([`0d1e1c9`](https://github.com/jbloomAus/mats_sae_training/commit/0d1e1c9a5f8b5c865f620bd97619cf108bf06796))

* implemented SAE groups ([`66facfe`](https://github.com/jbloomAus/mats_sae_training/commit/66facfef1094f311f00ee6a8abf53f62548802cf))

* Merge pull request #16 from chanind/flake-default-rules

chore: enable full flake8 default rules list ([`ad84706`](https://github.com/jbloomAus/mats_sae_training/commit/ad84706cc9d3d202669fa8175af9f181f323cd66))

* implemented sweeping via config list ([`80f61fa`](https://github.com/jbloomAus/mats_sae_training/commit/80f61faac4dfdab27d03cec672641d898441a530))

* Merge pull request #13 from chanind/poetry

chore: using poetry for dependency management ([`496f7b4`](https://github.com/jbloomAus/mats_sae_training/commit/496f7b4fa2d8835cec6673bec1ffa3d1f5fba268))

* progress on implementing multi-sae support ([`2ba2131`](https://github.com/jbloomAus/mats_sae_training/commit/2ba2131553b28af08d969304c2a2b68a8861fd98))

* Merge pull request #11 from lucyfarnik/fix-caching-shuffle-edge-case

Fixed edge case in activation cache shuffling ([`3727b5d`](https://github.com/jbloomAus/mats_sae_training/commit/3727b5d585efd3932fd49b540d11778c76bef561))

* Merge pull request #12 from lucyfarnik/add-run-name-to-config

Added run name to config ([`c2e05c4`](https://github.com/jbloomAus/mats_sae_training/commit/c2e05c46b0c7d7159a29aaa243cd33a621437f38))

* Added run name to config ([`ab2aabd`](https://github.com/jbloomAus/mats_sae_training/commit/ab2aabdcd9e293c32168549f9a99c25bf097a969))

* Fixed edge case in activation cache shuffling ([`18fd4a1`](https://github.com/jbloomAus/mats_sae_training/commit/18fd4a1d43bbc8b7a490779f06e4f3c6983e3ea7))

* Merge pull request #9 from chanind/rm-ds-store

chore: removing .DS_Store files ([`37771ce`](https://github.com/jbloomAus/mats_sae_training/commit/37771ce9a9bbd3b4bcf2d8749452f2f8e4516855))

* improve readmen ([`f3fe937`](https://github.com/jbloomAus/mats_sae_training/commit/f3fe937303f0eb54e83771d1b307c3f09d1145fd))

* fix_evals_bad_rebase ([`22e415d`](https://github.com/jbloomAus/mats_sae_training/commit/22e415dfc0ab961470fd2c7ed476f4181e2afa44))

* evals changes, incomplete ([`736c40e`](https://github.com/jbloomAus/mats_sae_training/commit/736c40e16cb89cf7d8bd721518df268fb4a35ee7))

* make tutorial independent of artefact and delete old artefact ([`6754e65`](https://github.com/jbloomAus/mats_sae_training/commit/6754e650e032efbc85b0dd11db480c96640bd0b6))

* fix MSE in ghost grad ([`44f7988`](https://github.com/jbloomAus/mats_sae_training/commit/44f7988e5a97bd93daedc3559cc4988caa128974))

* Merge pull request #5 from jbloomAus/clean_up_repo

Add CI/CD, black formatting, pre-commit with flake8 linting. Fix some bugs. ([`01ccb92`](https://github.com/jbloomAus/mats_sae_training/commit/01ccb92c4dbc161a26b22d60b5c9e0fa2d560519))

* clean up run examples ([`9d46bdd`](https://github.com/jbloomAus/mats_sae_training/commit/9d46bdda9ab3aaf8246bb01cb270dadf3e5e06c9))

* move where we save the final artifact ([`f445fac`](https://github.com/jbloomAus/mats_sae_training/commit/f445facde3eb7c2011905501a7644f8ef6cbd6e6))

* fix activations store innefficiency ([`07d38a0`](https://github.com/jbloomAus/mats_sae_training/commit/07d38a0eb5271d86fe2d5add8ae2ea5f7ae8aadc))

* black format and linting ([`479765b`](https://github.com/jbloomAus/mats_sae_training/commit/479765bc9ecc4f04fe76e1f4f447d0d0281e9045))

* dummy file change ([`912a748`](https://github.com/jbloomAus/mats_sae_training/commit/912a748f7b5bff0193a997ee7369c312f073c35f))

* try adding this branch listed specifically ([`7fd0e0c`](https://github.com/jbloomAus/mats_sae_training/commit/7fd0e0c3d3f951ae18b6b101043ba4a36c9933c4))

* yml not yaml ([`9f3f1c8`](https://github.com/jbloomAus/mats_sae_training/commit/9f3f1c87ceed84afefef5bc6d4b519d538bcac91))

* add ci ([`91aca91`](https://github.com/jbloomAus/mats_sae_training/commit/91aca9142459e2a72478d4546ee0fa5a2910c161))

* get units tests working ([`ade2976`](https://github.com/jbloomAus/mats_sae_training/commit/ade29762b4b94e02c3b44a66bedf150e721aec7c))

* make unit tests pass, add make file ([`08b2c92`](https://github.com/jbloomAus/mats_sae_training/commit/08b2c92f7fbfed38ea44312d53945cf1c9471d90))

* add pytest-cov to requirements.txt ([`ce526df`](https://github.com/jbloomAus/mats_sae_training/commit/ce526df64115113ecb68bd7f1a51b6fe11957553))

* seperate research from main repo ([`32b668c`](https://github.com/jbloomAus/mats_sae_training/commit/32b668ce96c8231eaaa93152d2b7d67dacc91dc5))

* remove comma and set default store batch size lower ([`9761b9a`](https://github.com/jbloomAus/mats_sae_training/commit/9761b9ae5f0d4dd51d445ba1229f9b2ebc628e62))

* notebook for Johny ([`39a18f2`](https://github.com/jbloomAus/mats_sae_training/commit/39a18f2f3fc138ba4743976b139071b7830c36a2))

* best practices ghost grads fix ([`f554b16`](https://github.com/jbloomAus/mats_sae_training/commit/f554b161e7b188d25ca9c21bf80629dc1b5b4e35))

* Update README.md

improved the hyperpars ([`2d4caf6`](https://github.com/jbloomAus/mats_sae_training/commit/2d4caf63e55d86967b906915c90a47094631276f))

* dashboard runner ([`a511223`](https://github.com/jbloomAus/mats_sae_training/commit/a511223ca3b5ee87930dc753b3164e00833dc17b))

* readme update ([`c303c55`](https://github.com/jbloomAus/mats_sae_training/commit/c303c554b6599c38a9b78ef2f5e1651267d38fab))

* still hadn&#39;t fixed the issue, now fixed ([`a36ee21`](https://github.com/jbloomAus/mats_sae_training/commit/a36ee21bb351d279952c638442b3874de7756d0c))

* fix mean of loss which broke in last commit ([`b4546db`](https://github.com/jbloomAus/mats_sae_training/commit/b4546dbdcea26d8812131572aa2c15fdfe4b4d28))

* generate dashboards ([`35fa631`](https://github.com/jbloomAus/mats_sae_training/commit/35fa63110f2ad790036a7eef2e4b7d46fd54fa61))

* Merge pull request #3 from jbloomAus/ghost_grads_dev

Ghost grads dev ([`4d150c2`](https://github.com/jbloomAus/mats_sae_training/commit/4d150c255c2637f485875a5709002195b2e51ce0))

* save final log sparsity ([`98e4f1b`](https://github.com/jbloomAus/mats_sae_training/commit/98e4f1bb416b33d49bef1acc676cc3b1e9ed6441))

* start saving log sparsity ([`4d6df6f`](https://github.com/jbloomAus/mats_sae_training/commit/4d6df6f41482e08530806495d22c5cedb1307448))

* get ghost grads working ([`e863ed7`](https://github.com/jbloomAus/mats_sae_training/commit/e863ed70afe1fb1d423262ee976196d2a69cf8b5))

* add notebook/changes for ghost-grad (not working yet) ([`73053c1`](https://github.com/jbloomAus/mats_sae_training/commit/73053c1fe9eed5a1e145f64a1fe11a300201544b))

* idk, probs good ([`0407ad9`](https://github.com/jbloomAus/mats_sae_training/commit/0407ad9b3749ab6baa1bf6dd3040d2e394369836))

* bunch of shit ([`1ec8f97`](https://github.com/jbloomAus/mats_sae_training/commit/1ec8f97330cc605d003757b9c8d6edb8040119bc))

* Merge branch &#39;main&#39; of github.com:jbloomAus/mats_sae_training ([`a22d856`](https://github.com/jbloomAus/mats_sae_training/commit/a22d85643d54e0924e5e4de8573671edf046bca5))

* Reverse engineering the &#34;not only... but&#34; feature ([`74d4fb8`](https://github.com/jbloomAus/mats_sae_training/commit/74d4fb8d7299ee6ace0331a2d9b8cbab2c965bcb))

* Merge pull request #2 from slavachalnev/no_reinit

Allow sampling method to be None ([`4c5fed8`](https://github.com/jbloomAus/mats_sae_training/commit/4c5fed8bbff8f13fdc534385caeb5455ff9d8a55))

* Allow sampling method to be None ([`166799d`](https://github.com/jbloomAus/mats_sae_training/commit/166799d0a9d10c01d39fdbf3c7e338bc99cc4d96))

* research/week_15th_jan/gpt2_small_resid_pre_3.ipynb ([`52a1da7`](https://github.com/jbloomAus/mats_sae_training/commit/52a1da72b25827b4ade2b13c679ac54c17def800))

* add arg for dead neuron calc ([`ffb75fb`](https://github.com/jbloomAus/mats_sae_training/commit/ffb75fb118db527e4633768534b04b5ef5779122))

* notebooks for lucy ([`0319d89`](https://github.com/jbloomAus/mats_sae_training/commit/0319d890d9da994f830b71361d4850b800f6862b))

* add args for b_dec_init ([`82da877`](https://github.com/jbloomAus/mats_sae_training/commit/82da877da7bf273870eb1446d6c9584650876981))

* add geom median as submodule instead ([`4c0d001`](https://github.com/jbloomAus/mats_sae_training/commit/4c0d0014e4a5f0b5661963ce9fd9318c1cdfca78))

* add geom median to req ([`4c8ac9d`](https://github.com/jbloomAus/mats_sae_training/commit/4c8ac9d53585e60647e140d13903ef44e55aa8b6))

* add-geometric-mean-b_dec-init ([`d5853f8`](https://github.com/jbloomAus/mats_sae_training/commit/d5853f8e075c731309cf208a5cd07b0ae0bdfa1e))

* reset feature sparsity calculation ([`4c7f6f2`](https://github.com/jbloomAus/mats_sae_training/commit/4c7f6f2d0960edc365369141671359c91281c600))

* anthropic sampling ([`048d267`](https://github.com/jbloomAus/mats_sae_training/commit/048d267c0f964086e755c3ca245399c68be4d70d))

* get anthropic resampling working ([`ca74543`](https://github.com/jbloomAus/mats_sae_training/commit/ca74543a117f7bc25e9b40b1dd5e4109a2c0121d))

* add ability to finetune existing autoencoder ([`c1208eb`](https://github.com/jbloomAus/mats_sae_training/commit/c1208eb61ebcc1dfbfc428b8b5fa090eb090bbd8))

* run notebook ([`879ad27`](https://github.com/jbloomAus/mats_sae_training/commit/879ad272f541bb5acc05be7b2a042ac9d45479a6))

* switch to batch size independent loss metrics ([`0623d39`](https://github.com/jbloomAus/mats_sae_training/commit/0623d39b323da05f4ae3b67df25841cabf83d200))

* track mean sparsity ([`75f1547`](https://github.com/jbloomAus/mats_sae_training/commit/75f15476400f26fddd2da9562be995e821745691))

* don&#39;t stop early ([`44078a6`](https://github.com/jbloomAus/mats_sae_training/commit/44078a6a7dd10423619ff3a348fd5276f9aad9b2))

* name runs better ([`5041748`](https://github.com/jbloomAus/mats_sae_training/commit/5041748e9166d6e84d017053fe61e60e194cead2))

* improve-eval-metrics-for-attn ([`00d9b65`](https://github.com/jbloomAus/mats_sae_training/commit/00d9b652d09f18c38dbd4c4e2376fcee3c457ac5))

* add hook q ([`b061ee3`](https://github.com/jbloomAus/mats_sae_training/commit/b061ee30e9cf86226e625e5e2e05ea7b4cc4550e))

* add copy suppression notebook ([`1dc893a`](https://github.com/jbloomAus/mats_sae_training/commit/1dc893a17db8a7866a598a4632034c91c139d862))

* fix check in neuron sampling ([`809becd`](https://github.com/jbloomAus/mats_sae_training/commit/809becdbd67b2e23c8be572188f8f098d770871a))

* Merge pull request #1 from jbloomAus/activations_on_disk

Activations on disk ([`e5f198e`](https://github.com/jbloomAus/mats_sae_training/commit/e5f198ea52741dc54e463c55f36409b2b559bf3a))

* merge into main ([`94ed3e6`](https://github.com/jbloomAus/mats_sae_training/commit/94ed3e62636267d5cfe4163dcb37bbd30827e064))

* notebook ([`b5344a3`](https://github.com/jbloomAus/mats_sae_training/commit/b5344a3048706e4b828c80eaccbc5b8b75eeb2ae))

* various research notebooks ([`be63fce`](https://github.com/jbloomAus/mats_sae_training/commit/be63fcea9db2555c1ac9803fc86008487ed97958))

* Added activations caching to run.ipynb ([`054cf6d`](https://github.com/jbloomAus/mats_sae_training/commit/054cf6dc7082465b05bfbfd138566f41f1631471))

* Added activations dir to gitignore ([`c4a31ae`](https://github.com/jbloomAus/mats_sae_training/commit/c4a31ae1e0876d475bf6881c1570e0fbb00fdff9))

* Saving and loading activations from disk ([`309e2de`](https://github.com/jbloomAus/mats_sae_training/commit/309e2de8881e0b9e9f3b6a786501cb4336143890))

* Fixed typo that threw out half of activations ([`5f73918`](https://github.com/jbloomAus/mats_sae_training/commit/5f73918a9447d63dcee9a73c6f9e2e23bec96abe))

* minor speed improvement ([`f7ea316`](https://github.com/jbloomAus/mats_sae_training/commit/f7ea31679222ff03af0048a0730097eccb644de0))

* add notebook with different example runs ([`c0eac0a`](https://github.com/jbloomAus/mats_sae_training/commit/c0eac0a4f3889be932f50cc7485ef0cc236a46a3))

* add ability to train on attn heads ([`18cfaad`](https://github.com/jbloomAus/mats_sae_training/commit/18cfaad66432c827528e28d3969fcc6dbd16391a))

* add gzip for pt artefacts ([`9614a23`](https://github.com/jbloomAus/mats_sae_training/commit/9614a237ef96a79383e1ead8d7769a427279b310))

* add_example_feature_dashboard ([`e90e54d`](https://github.com/jbloomAus/mats_sae_training/commit/e90e54dd2e7cd6e2844710d9b8ba64a64ef25993))

* get_shit_done ([`ce73042`](https://github.com/jbloomAus/mats_sae_training/commit/ce7304265a01ce4e30537f58b148dd069149c474))

* commit_various_things_in_progress ([`3843c39`](https://github.com/jbloomAus/mats_sae_training/commit/3843c3976936e293b5c87b3501d6a7f397183f68))

* add sae visualizer and tutorial ([`6f4030c`](https://github.com/jbloomAus/mats_sae_training/commit/6f4030c5d411f7c2d23518ce247b553b216499e7))

* make it possible to load  sae trained on cuda onto mps ([`3298b75`](https://github.com/jbloomAus/mats_sae_training/commit/3298b75a5ff1c0a660840c6dfa8931a9faf226f7))

* reduce hist freq, don&#39;t cap re-init ([`debcf0f`](https://github.com/jbloomAus/mats_sae_training/commit/debcf0fd62d4d252710d6b2f947efc33ba934ef7))

* add loader import to readme ([`b63f14e`](https://github.com/jbloomAus/mats_sae_training/commit/b63f14e01758a243487ac87bee710298bfef2603))

* Update README.md ([`88f086b`](https://github.com/jbloomAus/mats_sae_training/commit/88f086b664f1c3750dd4b1b3fe0b79df1bfa180f))

* improve-resampling ([`a3072c2`](https://github.com/jbloomAus/mats_sae_training/commit/a3072c26105b1053ece5390da26cdf8605c4a4b7))

* add readme ([`e9b8e56`](https://github.com/jbloomAus/mats_sae_training/commit/e9b8e56aac2ef694e6cab4a6fc24cf343e188ded))

* fixl0_plus_other_stuff ([`2f162f0`](https://github.com/jbloomAus/mats_sae_training/commit/2f162f09be39a9bf56c40d1e461c064e4f2ed834))

* add checkpoints ([`4cacbfc`](https://github.com/jbloomAus/mats_sae_training/commit/4cacbfc2d32a509b8c34ccb940d0a21470c585ca))

* improve_model_saving_loading ([`f6697c6`](https://github.com/jbloomAus/mats_sae_training/commit/f6697c6cfe654bee8b677401e423f2b09e682f73))

* stuff ([`19d278a`](https://github.com/jbloomAus/mats_sae_training/commit/19d278a223bbdc2673616644f4da90492fe7b169))

* Added support for non-tokenized datasets ([`afcc239`](https://github.com/jbloomAus/mats_sae_training/commit/afcc239de1aa91961a14e5734644d2ffaf4b764a))

* notebook_for_keith ([`d06e09b`](https://github.com/jbloomAus/mats_sae_training/commit/d06e09b70bd7fcf2419f9196970025097b7b55ef))

* fix resampling bug ([`2b43980`](https://github.com/jbloomAus/mats_sae_training/commit/2b43980cc2bae402e2016efaaa042f2cb545a02f))

* test pars ([`f601362`](https://github.com/jbloomAus/mats_sae_training/commit/f6013625963ebabdb86063f8db7a5e8339e37873))

* further-lm-improvments ([`63048eb`](https://github.com/jbloomAus/mats_sae_training/commit/63048eb9e5f12a2f631b454f82a17563efcae328))

* get_lm_working_well ([`eba5f79`](https://github.com/jbloomAus/mats_sae_training/commit/eba5f79ff267217a3f69b4dcab4903c98ed56055))

* basic-lm-training-currently-broken ([`7396b8b`](https://github.com/jbloomAus/mats_sae_training/commit/7396b8ba65a579d8aff0987c61ea9d7ac9177380))

* set_up_lm_runner ([`d1095af`](https://github.com/jbloomAus/mats_sae_training/commit/d1095afefb17f87a2a63841fbe6845b46650dde7))

* fix old test, may remove ([`b407aab`](https://github.com/jbloomAus/mats_sae_training/commit/b407aabfcd9d8ff9a8752a92a26701dbc8da04a2))

* happy with hyperpars on benchmark ([`836298a`](https://github.com/jbloomAus/mats_sae_training/commit/836298a897e759e1f99a35c7d8195bcfb580afe3))

* improve metrics ([`f52c7bb`](https://github.com/jbloomAus/mats_sae_training/commit/f52c7bb142e019f9dd6d6f9d73ff3874bf33b529))

* make toy model runner ([`4851dd1`](https://github.com/jbloomAus/mats_sae_training/commit/4851dd1a15ee57d658d41c8673fd6b3fe71a2b45))

* various-changes-toy-model-test ([`a61b75f`](https://github.com/jbloomAus/mats_sae_training/commit/a61b75f058b79262e587a302319a1bb9136c353e))

* Added activation store and activation gathering ([`a85f24d`](https://github.com/jbloomAus/mats_sae_training/commit/a85f24d9389936dc7ad18aaf74d600669294b9cc))

* First successful run on toy models ([`4927145`](https://github.com/jbloomAus/mats_sae_training/commit/4927145a6dfe9026d47c1bea1497667eb5a715fe))

* halfway-to-toy-models ([`feeb411`](https://github.com/jbloomAus/mats_sae_training/commit/feeb411b4526c0627a9a558cfcab91252f90046b))

* Initial commit ([`7a94b0e`](https://github.com/jbloomAus/mats_sae_training/commit/7a94b0ebc34edd24e98df753d100e8a4c238d4f6))
