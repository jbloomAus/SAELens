import json
import os
import urllib.parse
import webbrowser
from datetime import datetime
from typing import Literal, Optional

import requests

NEURONPEDIA_DOMAIN = "https://neuronpedia.org"

# Constants for replacing NaNs and Infs in outputs
POSITIVE_INF_REPLACEMENT = 9999
NEGATIVE_INF_REPLACEMENT = -9999
NAN_REPLACEMENT = 0
OTHER_INVALID_REPLACEMENT = -99999


def NanAndInfReplacer(value: str):
    replacements = {
        "-Infinity": NEGATIVE_INF_REPLACEMENT,
        "Infinity": POSITIVE_INF_REPLACEMENT,
        "NaN": NAN_REPLACEMENT,
    }
    if value in replacements:
        replacedValue = replacements[value]
        # print(f"Warning: Replacing value {value} with {replacedValue}")
        return float(replacedValue)
    else:
        # print(f"Warning: Replacing value {value} with {NAN_REPLACEMENT}")
        return NAN_REPLACEMENT


def get_neuronpedia_feature(
    feature: int,
    layer: int,
    model: str = "gpt2-small",
    dataset: str = "res-jb",
):
    url = NEURONPEDIA_DOMAIN + "/api/feature/"
    url = url + f"{model}/{layer}-{dataset}/{feature}"

    result = requests.get(url).json()
    result["index"] = int(result["index"])

    return result


def get_neuronpedia_quick_list(
    features: list[int],
    layer: int,
    model: str = "gpt2-small",
    dataset: str = "res-jb",
    name: str = "temporary_list",
):
    url = NEURONPEDIA_DOMAIN + "/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {
            "modelId": model,
            "layer": f"{layer}-{dataset}",
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    webbrowser.open(url)

    return url


class NeuronpediaActivation(object):
    id: str = ""
    tokens = []
    act_values = []

    def __init__(self, id: str, tokens: list[str], act_values: list[float]):
        self.id = id
        self.tokens = tokens
        self.act_values = act_values


class NeuronpediaFeature(object):
    modelId = ""
    layer = 0
    dataset = ""
    index = 0
    description = ""
    activations = []
    autointerp_explanation = ""
    autointerp_explanation_score = 0

    def __init__(
        self,
        modelId: str,
        layer: int,
        dataset: str,
        feature: int,
        description: str = "",
        activations: list[NeuronpediaActivation] = [],
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

    def has_activating_text(self):
        has_activating_text = False
        for activation in self.activations:
            if max(activation.act_values) > 0:
                has_activating_text = True
                break
        return has_activating_text


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
        "apiKey": api_key,
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
    response = requests.post(url, json=body)
    result = response.json()

    if "url" in result and open_browser:
        webbrowser.open(result["url"])
        return result["url"]
    else:
        raise Exception("Error in creating list: " + result["message"])


def test_key(api_key: str):
    url = NEURONPEDIA_DOMAIN + "/api/test"
    body = {
        "apiKey": api_key,
    }
    response = requests.post(url, json=body)
    if response.status_code != 200:
        raise Exception("Neuronpedia API key is not valid.")


async def autointerp_neuronpedia_features(
    features: list[NeuronpediaFeature],
    openai_api_key: str,
    autointerp_retry_attempts: int = 3,
    autointerp_score_max_concurrent: int = 20,  # good for this to match num_activations_to_use_for_score
    neuronpedia_api_key: str = "",
    # TODO check max budget estimate based on: num features, num act texts, act text lengths. fail if too high.
    # max_budget_approx_usd: float = 5.00,
    do_score: bool = True,
    output_dir: str = "neuronpedia_outputs/autointerp",
    num_activations_to_use: int = 20,
    upload_to_neuronpedia: bool = True,
    autointerp_model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo"] = "gpt-3.5-turbo",
):
    print("\n\n")

    # make output_file named autointerp-<timestamp>
    output_file = output_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl"
    if not os.path.exists(output_dir):
        print("Creating output directory " + output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print("===== Your results will be saved to: " + output_file + "=====")

    # we import this here instead of top of file because the library requires the API key to be set first
    os.environ["OPENAI_API_KEY"] = openai_api_key
    from neuron_explainer.activations.activation_records import calculate_max_activation
    from neuron_explainer.activations.activations import ActivationRecord
    from neuron_explainer.explanations.calibrated_simulator import (
        UncalibratedNeuronSimulator,
    )
    from neuron_explainer.explanations.explainer import (
        ContextSize,
        TokenActivationPairExplainer,
    )
    from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
    from neuron_explainer.explanations.prompt_builder import PromptFormat
    from neuron_explainer.explanations.scoring import simulate_and_score
    from neuron_explainer.explanations.simulator import (
        LogprobFreeExplanationTokenSimulator,
    )

    """
    This does the following:
    1. Fetches the features from Neuronpedia, including their activation texts
    2. Explains the features using the autointerp_explainer_model_name
    3. Scores the features using the autointerp_scorer_model_name
    4. Saves the results in output_dir
    5. Uploads the results to Neuronpedia

    The openai_api_key is not sent to Neuronpedia, only to OpenAI.
    """

    if upload_to_neuronpedia and neuronpedia_api_key == "":
        raise Exception(
            "You need to provide a Neuronpedia API key to upload the results to Neuronpedia."
        )

    test_key(neuronpedia_api_key)

    # 1. Fetches the features from Neuronpedia, including their activation texts. Perform check for dead features.
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

        if feature.has_activating_text() is False:
            raise Exception(
                f"Feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature} appears dead - it does not have activating text."
            )

    # TODO check max budget estimate based on number of features and act texts and act text length, fail if too high

    # 2. Explain the features using the selected autointerp_explainer_model_name
    for iteration_num, feature in enumerate(features):
        # print start time
        start_time = datetime.now()

        print(
            f"\n========== Feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature} ({iteration_num + 1} of {len(features)} Features) =========="
        )
        print(
            f"\n=== Step 2) Explaining feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
        )
        activationRecords = []
        for activation in feature.activations:
            activationRecord = ActivationRecord(
                tokens=activation.tokens, activations=activation.act_values
            )
            activationRecords.append(activationRecord)

        explainer = TokenActivationPairExplainer(
            model_name=autointerp_model_name,
            prompt_format=PromptFormat.HARMONY_V4,
            context_size=ContextSize.SIXTEEN_K,
            max_concurrent=1,
        )

        explanations = []
        for _ in range(autointerp_retry_attempts):
            try:
                explanations = await explainer.generate_explanations(
                    all_activation_records=activationRecords,
                    max_activation=calculate_max_activation(activationRecords),
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
        explanation = explanations[0]
        # GPT ends its explanations with a period. Remove it.
        if explanation.endswith("."):
            explanation = explanation[:-1]
        print(f"===== {autointerp_model_name}'s explanation: {explanation}")
        feature.autointerp_explanation = explanation

        # 3. Scores the features using the autointerp_scorer_model_name
        if do_score:
            print(
                f"\n=== Step 3) Scoring feature {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
            )
            print("=== This can take up to 30 seconds.")

            # GPT struggles with non-ascii so we turn them into string representations
            # make a temporary activation records copy for this, so we can return the original later
            tempActivationRecords: list[ActivationRecord] = []
            for activationRecord in activationRecords:
                replacedActTokens: list[str] = []
                for _, token in enumerate(activationRecord.tokens):
                    replacedActTokens.append(
                        token.replace("<|endoftext|>", "<|not_endoftext|>")
                        .replace(" 55", "_55")
                        .encode("ascii", errors="backslashreplace")
                        .decode("ascii")
                    )
                tempActivationRecords.append(
                    ActivationRecord(
                        tokens=replacedActTokens,
                        activations=activationRecord.activations,
                    )
                )

            # Simulate and score the explanation.
            score = None
            scored_simulation = None
            for _ in range(autointerp_retry_attempts):
                try:
                    simulator = UncalibratedNeuronSimulator(
                        LogprobFreeExplanationTokenSimulator(
                            autointerp_model_name,
                            explanation,
                            json_mode=True,
                            max_concurrent=autointerp_score_max_concurrent,
                            few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
                            prompt_format=PromptFormat.HARMONY_V4,
                        )
                    )
                    scored_simulation = await simulate_and_score(
                        simulator, tempActivationRecords
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
            print(f"===== {autointerp_model_name}'s score: {(score * 100):.0f}")

            # replace NaNs and Infs in the output so we get valid JSON
            output_data = json.dumps(
                {
                    "apiKey": neuronpedia_api_key,
                    "feature": {
                        "modelId": feature.modelId,
                        "layer": f"{feature.layer}-{feature.dataset}",
                        "index": feature.feature,
                        "activations": feature.activations,
                        "explanation": feature.autointerp_explanation,
                        "explanationScore": feature.autointerp_explanation_score,
                        "autointerpModel": autointerp_model_name,
                        "simulatedActivations": scored_simulation.scored_sequence_simulations,
                    },
                },
                default=vars,
            )
            output_data_json = json.loads(
                output_data,
                parse_constant=NanAndInfReplacer,
            )
            output_data_str = json.dumps(output_data)

            # 4. Save the results in output_file
            # open output_file and append the feature
            print(f"\n=== Step 4) Saving feature to {output_file}")
            with open(output_file, "a") as f:
                f.write(output_data_str)
                f.write("\n")

            # 5. Uploads the results to Neuronpedia
            if upload_to_neuronpedia:
                print(
                    f"\n=== Step 5) Uploading feature to Neuronpedia: {feature.modelId}@{feature.layer}-{feature.dataset}:{feature.feature}"
                )
                url = NEURONPEDIA_DOMAIN + "/api/upload-explanation"
                body = output_data_json
                response = requests.post(url, json=body)
                if response.status_code != 200:
                    print(
                        f"ERROR: Couldn't upload explanation to Neuronpedia: {response.text}"
                    )

        # print end_time minus start_time)
        end_time = datetime.now()
        print("\n========== Time Spent for Feature: {}\n".format(end_time - start_time))

    print("\n\n========== Generation and Upload Complete ==========\n\n")
