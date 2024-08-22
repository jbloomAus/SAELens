import asyncio
import json
import os
import urllib.parse
import webbrowser
from datetime import datetime
from typing import Any, Optional, TypeVar

import requests
from dotenv import load_dotenv
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.calibrated_simulator import (
    UncalibratedNeuronSimulator,
)
from neuron_explainer.explanations.explainer import (
    HARMONY_V4_MODELS,
    ContextSize,
    TokenActivationPairExplainer,
)
from neuron_explainer.explanations.explanations import ScoredSimulation
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import (
    _simulate_and_score_sequence,
    aggregate_scored_sequence_simulations,
)
from neuron_explainer.explanations.simulator import (
    LogprobFreeExplanationTokenSimulator,
    NeuronSimulator,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from sae_lens import SAE

NEURONPEDIA_DOMAIN = "https://neuronpedia.org"

# Constants for replacing NaNs and Infs in outputs
POSITIVE_INF_REPLACEMENT = 9999
NEGATIVE_INF_REPLACEMENT = -9999
NAN_REPLACEMENT = 0
OTHER_INVALID_REPLACEMENT = -99999

# Pick up OPENAI_API_KEY from environment variable
load_dotenv()


def NanAndInfReplacer(value: str):
    """Replace NaNs and Infs in outputs."""
    replacements = {
        "-Infinity": NEGATIVE_INF_REPLACEMENT,
        "Infinity": POSITIVE_INF_REPLACEMENT,
        "NaN": NAN_REPLACEMENT,
    }
    if value in replacements:
        replaced_value = replacements[value]
        return float(replaced_value)
    else:
        return NAN_REPLACEMENT


def open_neuronpedia_feature_dashboard(sae: SAE, index: int):
    sae_id = sae.cfg.neuronpedia_id
    if sae_id is None:
        print(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )
    else:
        url = f"{NEURONPEDIA_DOMAIN}/{sae_id}/{index}"
        webbrowser.open(url)


def get_neuronpedia_quick_list(
    sae: SAE,
    features: list[int],
    name: str = "temporary_list",
):

    sae_id = sae.cfg.neuronpedia_id
    if sae_id is None:
        print(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )
    assert sae_id is not None

    url = NEURONPEDIA_DOMAIN + "/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {
            "modelId": sae.cfg.model_name,
            "layer": sae_id.split("/")[1],
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    webbrowser.open(url)

    return url


def get_neuronpedia_feature(
    feature: int, layer: int, model: str = "gpt2-small", dataset: str = "res-jb"
) -> dict[str, Any]:
    """Fetch a feature from Neuronpedia API."""
    url = f"{NEURONPEDIA_DOMAIN}/api/feature/{model}/{layer}-{dataset}/{feature}"
    result = requests.get(url).json()
    result["index"] = int(result["index"])
    return result


class NeuronpediaActivation:
    """Represents an activation from Neuronpedia."""

    def __init__(self, id: str, tokens: list[str], act_values: list[float]):
        self.id = id
        self.tokens = tokens
        self.act_values = act_values


class NeuronpediaFeature:
    """Represents a feature from Neuronpedia."""

    def __init__(
        self,
        modelId: str,
        layer: int,
        dataset: str,
        feature: int,
        description: str = "",
        activations: list[NeuronpediaActivation] | None = None,
        autointerp_explanation: str = "",
        autointerp_explanation_score: float = 0.0,
    ):
        self.modelId = modelId
        self.layer = layer
        self.dataset = dataset
        self.feature = feature
        self.description = description
        self.activations = activations
        self.autointerp_explanation = autointerp_explanation
        self.autointerp_explanation_score = autointerp_explanation_score

    def has_activating_text(self) -> bool:
        """Check if the feature has activating text."""
        if self.activations is None:
            return False
        else:
            return any(
                max(activation.act_values) > 0 for activation in self.activations
            )


T = TypeVar("T")


@retry(wait=wait_random_exponential(min=1, max=500), stop=stop_after_attempt(10))
def sleep_identity(x: T) -> T:
    """Dummy function for retrying."""
    return x


@retry(wait=wait_random_exponential(min=1, max=500), stop=stop_after_attempt(10))
async def simulate_and_score(
    simulator: NeuronSimulator, activation_records: list[ActivationRecord]
) -> ScoredSimulation:
    """Score an explanation of a neuron by how well it predicts activations on the given text sequences."""
    scored_sequence_simulations = await asyncio.gather(
        *[
            sleep_identity(
                _simulate_and_score_sequence(
                    simulator,
                    activation_record,
                )
            )
            for activation_record in activation_records
        ]
    )
    return aggregate_scored_sequence_simulations(scored_sequence_simulations)


def make_neuronpedia_list_with_features(
    api_key: str,
    list_name: str,
    features: list[NeuronpediaFeature],
    list_description: Optional[str] = None,
    open_browser: bool = True,
):
    url = NEURONPEDIA_DOMAIN + "/api/list/new-with-features"

    # make POST json request with body
    body = {
        "name": list_name,
        "description": list_description,
        "features": [
            {
                "modelId": feature.modelId,
                "layer": f"{feature.layer}-{feature.dataset}",
                "index": feature.feature,
                "description": feature.description,
            }
            for feature in features
        ],
    }
    response = requests.post(url, json=body, headers={"x-api-key": api_key})
    result = response.json()

    if "url" in result and open_browser:
        webbrowser.open(result["url"])
        return result["url"]
    else:
        raise Exception("Error in creating list: " + result["message"])


def test_key(api_key: str):
    """Test the validity of the Neuronpedia API key."""
    url = f"{NEURONPEDIA_DOMAIN}/api/test"
    body = {"apiKey": api_key}
    response = requests.post(url, json=body)
    if response.status_code != 200:
        raise Exception("Neuronpedia API key is not valid.")


async def autointerp_neuronpedia_features(  # noqa: C901
    features: list[NeuronpediaFeature],
    openai_api_key: str | None = None,
    autointerp_retry_attempts: int = 3,
    autointerp_score_max_concurrent: int = 20,
    neuronpedia_api_key: str | None = None,
    skip_neuronpedia_api_key_test: bool = False,
    do_score: bool = True,
    output_dir: str = "neuronpedia_outputs/autointerp",
    num_activations_to_use: int = 20,
    max_explanation_activation_records: int = 20,
    upload_to_neuronpedia: bool = True,
    autointerp_explainer_model_name: str = "gpt-4-1106-preview",
    autointerp_scorer_model_name: str | None = "gpt-3.5-turbo",
    save_to_disk: bool = True,
):
    """
    Autointerp Neuronpedia features.

    Args:
        features: List of NeuronpediaFeature objects.
        openai_api_key: OpenAI API key.
        autointerp_retry_attempts: Number of retry attempts for autointerp.
        autointerp_score_max_concurrent: Maximum number of concurrent requests for autointerp scoring.
        neuronpedia_api_key: Neuronpedia API key.
        do_score: Whether to score the features.
        output_dir: Output directory for saving the results.
        num_activations_to_use: Number of activations to use.
        max_explanation_activation_records: Maximum number of activation records for explanation.
        upload_to_neuronpedia: Whether to upload the results to Neuronpedia.
        autointerp_explainer_model_name: Model name for autointerp explainer.
        autointerp_scorer_model_name: Model name for autointerp scorer.

    Returns:
        None
    """
    print("\n\n")

    if os.getenv("OPENAI_API_KEY") is None:
        if openai_api_key is None:
            raise Exception(
                "You need to provide an OpenAI API key either in environment variable OPENAI_API_KEY or as an argument."
            )
        else:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    if autointerp_explainer_model_name not in HARMONY_V4_MODELS:
        raise Exception(
            f"Invalid explainer model name: {autointerp_explainer_model_name}. Must be one of: {HARMONY_V4_MODELS}"
        )

    if do_score and autointerp_scorer_model_name not in HARMONY_V4_MODELS:
        raise Exception(
            f"Invalid scorer model name: {autointerp_scorer_model_name}. Must be one of: {HARMONY_V4_MODELS}"
        )

    if upload_to_neuronpedia:
        if neuronpedia_api_key is None:
            raise Exception(
                "You need to provide a Neuronpedia API key to upload the results to Neuronpedia."
            )
        if not skip_neuronpedia_api_key_test:
            test_key(neuronpedia_api_key)

    print("\n\n=== Step 1) Fetching features from Neuronpedia")
    for feature in features:
        feature_data = get_neuronpedia_feature(
            feature=feature.feature,
            layer=feature.layer,
            model=feature.modelId,
            dataset=feature.dataset,
        )

        if "modelId" not in feature_data:
            raise Exception(
                f"Feature {feature.feature} in layer {feature.layer} of model {feature.modelId} and dataset {feature.dataset} does not exist."
            )

        if "activations" not in feature_data or len(feature_data["activations"]) == 0:
            raise Exception(
                f"Feature {feature.feature} in layer {feature.layer} of model {feature.modelId} and dataset {feature.dataset} does not have activations."
            )

        activations = feature_data["activations"]
        activations_to_add = []
        for activation in activations:
            if len(activations_to_add) < num_activations_to_use:
                activations_to_add.append(
                    NeuronpediaActivation(
                        id=activation["id"],
                        tokens=activation["tokens"],
                        act_values=activation["values"],
                    )
                )
        feature.activations = activations_to_add

        if not feature.has_activating_text():
            raise Exception(
                f"Feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature} appears dead - it does not have activating text."
            )

    for iteration_num, feature in enumerate(features):
        start_time = datetime.now()

        print(
            f"\n========== Feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature} ({iteration_num + 1} of {len(features)} Features) =========="
        )
        print(
            f"\n=== Step 2) Explaining feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
        )

        if feature.activations is None:
            feature.activations = []
        activation_records = [
            ActivationRecord(
                tokens=activation.tokens, activations=activation.act_values
            )
            for activation in feature.activations
        ]

        activation_records_explaining = activation_records[
            :max_explanation_activation_records
        ]

        explainer = TokenActivationPairExplainer(
            model_name=autointerp_explainer_model_name,
            prompt_format=PromptFormat.HARMONY_V4,
            context_size=ContextSize.SIXTEEN_K,
            max_concurrent=1,
        )

        explanations = []
        for _ in range(autointerp_retry_attempts):
            try:
                explanations = await explainer.generate_explanations(
                    all_activation_records=activation_records_explaining,
                    max_activation=calculate_max_activation(
                        activation_records_explaining
                    ),
                    num_samples=1,
                )
            except Exception as e:
                print(f"ERROR, RETRYING: {e}")
            else:
                break
        else:
            print(
                f"ERROR: Failed to explain feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
            )

        assert len(explanations) == 1
        explanation = explanations[0].rstrip(".")
        print(f"===== {autointerp_explainer_model_name}'s explanation: {explanation}")
        feature.autointerp_explanation = explanation

        scored_simulation = None
        if do_score and autointerp_scorer_model_name:
            print(
                f"\n=== Step 3) Scoring feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
            )
            print("=== This can take up to 30 seconds.")

            temp_activation_records = [
                ActivationRecord(
                    tokens=[
                        token.replace("<|endoftext|>", "<|not_endoftext|>")
                        .replace(" 55", "_55")
                        .encode("ascii", errors="backslashreplace")
                        .decode("ascii")
                        for token in activation_record.tokens
                    ],
                    activations=activation_record.activations,
                )
                for activation_record in activation_records
            ]

            score = None
            scored_simulation = None
            for _ in range(autointerp_retry_attempts):
                try:
                    simulator = UncalibratedNeuronSimulator(
                        LogprobFreeExplanationTokenSimulator(
                            autointerp_scorer_model_name,
                            explanation,
                            json_mode=True,
                            max_concurrent=autointerp_score_max_concurrent,
                            few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
                            prompt_format=PromptFormat.HARMONY_V4,
                        )
                    )
                    scored_simulation = await simulate_and_score(
                        simulator, temp_activation_records
                    )
                    score = scored_simulation.get_preferred_score()
                except Exception as e:
                    print(f"ERROR, RETRYING: {e}")
                else:
                    break

            if (
                score is None
                or scored_simulation is None
                or len(scored_simulation.scored_sequence_simulations)
                != num_activations_to_use
            ):
                print(
                    f"ERROR: Failed to score feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}. Skipping it."
                )
                continue
            feature.autointerp_explanation_score = score
            print(f"===== {autointerp_scorer_model_name}'s score: {(score * 100):.0f}")

        else:
            print("=== Step 3) Skipping scoring as instructed.")

        feature_data = {
            "modelId": feature.modelId,
            "layer": f"{feature.layer}-{feature.dataset}",
            "index": feature.feature,
            "explanation": feature.autointerp_explanation,
            "explanationScore": feature.autointerp_explanation_score,
            "explanationModel": autointerp_explainer_model_name,
        }
        if do_score and autointerp_scorer_model_name and scored_simulation:
            feature_data["activations"] = feature.activations
            feature_data["simulationModel"] = autointerp_scorer_model_name
            feature_data["simulationActivations"] = scored_simulation.scored_sequence_simulations  # type: ignore
            feature_data["simulationScore"] = feature.autointerp_explanation_score
        feature_data_str = json.dumps(feature_data, default=vars)

        if save_to_disk:
            output_file = f"{output_dir}/{feature.modelId}-{feature.layer}-{feature.dataset}_feature-{feature.feature}_time-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n=== Step 4) Saving feature to {output_file}")
            with open(output_file, "a") as f:
                f.write(feature_data_str)
                f.write("\n")
        else:
            print("\n=== Step 4) Skipping saving to disk.")

        if upload_to_neuronpedia:
            print("\n=== Step 5) Uploading feature to Neuronpedia")
            upload_data = json.dumps(
                {
                    "feature": feature_data,
                },
                default=vars,
            )
            upload_data_json = json.loads(upload_data, parse_constant=NanAndInfReplacer)
            url = f"{NEURONPEDIA_DOMAIN}/api/explanation/new"
            response = requests.post(
                url, json=upload_data_json, headers={"x-api-key": neuronpedia_api_key}
            )
            if response.status_code != 200:
                print(
                    f"ERROR: Couldn't upload explanation to Neuronpedia: {response.text}"
                )
            else:
                print(
                    f"===== Uploaded to Neuronpedia: {NEURONPEDIA_DOMAIN}/{feature.modelId}/{feature.layer}-{feature.dataset}/{feature.feature}"
                )

        end_time = datetime.now()
        print(f"\n========== Time Spent for Feature: {end_time - start_time}\n")

    print("\n\n========== Generation and Upload Complete ==========\n\n")
