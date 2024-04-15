import sys
from sae_lens.analysis.neuronpedia_runner import NeuronpediaRunner

SAE_PATH = sys.argv[1]
MODEL_ID = sys.argv[2]
SAE_ID = sys.argv[3]
N_BATCHES_SAMPLE = int(sys.argv[4])
N_PROMPTS_SELECT = int(sys.argv[5])
FEATURES_AT_A_TIME = int(sys.argv[6])
START_BATCH_INCLUSIVE = int(sys.argv[7])
END_BATCH_INCLUSIVE = int(sys.argv[8])

NP_OUTPUT_FOLDER = "../../neuronpedia_outputs"

runner = NeuronpediaRunner(
    sae_path=SAE_PATH,
    model_id=MODEL_ID,
    sae_id=SAE_ID,
    neuronpedia_outputs_folder=NP_OUTPUT_FOLDER,
    init_session=True,
    n_batches_to_sample_from=N_BATCHES_SAMPLE,
    n_prompts_to_select=N_PROMPTS_SELECT,
    n_features_at_a_time=FEATURES_AT_A_TIME,
    start_batch_inclusive=START_BATCH_INCLUSIVE,
    end_batch_inclusive=END_BATCH_INCLUSIVE,
)
runner.run()
