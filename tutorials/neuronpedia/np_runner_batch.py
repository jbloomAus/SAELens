import sys

LAYER = int(sys.argv[1])  # 0
TYPE = sys.argv[2]  # "resid"
SOURCE_AUTHOR_SUFFIX = sys.argv[3]  # "sm"
FEATURES_AT_A_TIME = int(
    sys.argv[4]
)  # this must stay the same or your batching will be off
START_BATCH_INCLUSIVE = int(sys.argv[5])
END_BATCH_INCLUSIVE = int(sys.argv[6]) if len(sys.argv) > 6 else None
USE_LEGACY = True

# Change these depending on how your files are named
SAE_PATH = f"../../data/{SOURCE_AUTHOR_SUFFIX}/sae_{LAYER}_{TYPE}.pt"
FEATURE_SPARSITY_PATH = (
    f"../../data/{SOURCE_AUTHOR_SUFFIX}/feature_sparsity_{LAYER}_{TYPE}.pt"
)

from sae_lens.analysis.neuronpedia_runner import NeuronpediaRunner

NP_OUTPUT_FOLDER = "../../neuronpedia_outputs"

runner = NeuronpediaRunner(
    sae_path=SAE_PATH,
    use_legacy=USE_LEGACY,
    feature_sparsity_path=FEATURE_SPARSITY_PATH,
    neuronpedia_parent_folder=NP_OUTPUT_FOLDER,
    init_session=True,
    n_batches_to_sample_from=2**12,
    n_prompts_to_select=4096 * 6,
    n_features_at_a_time=FEATURES_AT_A_TIME,
    buffer_tokens_left=64,
    buffer_tokens_right=62,
    start_batch_inclusive=START_BATCH_INCLUSIVE,
    end_batch_inclusive=END_BATCH_INCLUSIVE,
)
runner.run()
