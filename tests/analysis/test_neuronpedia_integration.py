import json
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from sae_lens.analysis.neuronpedia_integration import (
    NEURONPEDIA_DOMAIN,
    NeuronpediaFeature,
    autointerp_neuronpedia_features,
    get_neuronpedia_feature,
    get_neuronpedia_quick_list,
    make_neuronpedia_list_with_features,
    open_neuronpedia_feature_dashboard,
)
from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig


def test_get_neuronpedia_feature():
    result = get_neuronpedia_feature(
        feature=0, layer=0, model="gpt2-small", dataset="res-jb"
    )

    assert result["modelId"] == "gpt2-small"
    assert result["layer"] == "0-res-jb"
    assert result["index"] == 0


@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_get_neuronpedia_quick_list(
    mock_open: MagicMock, gpt2_res_jb_l4_sae: SAE[StandardSAEConfig]
):
    features = [0, 1, 2, 3]
    name = "test_list"

    url = get_neuronpedia_quick_list(gpt2_res_jb_l4_sae, features, name=name)

    # Verify webbrowser.open was called with the returned URL
    mock_open.assert_called_once_with(url)

    # Verify URL structure
    assert url.startswith(f"{NEURONPEDIA_DOMAIN}/quick-list/")
    assert f"name={urllib.parse.quote(name)}" in url

    # Verify the features are properly encoded in the URL
    neuronpedia_id = gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id
    assert neuronpedia_id is not None  # This should be true for the test fixture

    expected_features = [
        {
            "modelId": gpt2_res_jb_l4_sae.cfg.metadata.model_name,
            "layer": neuronpedia_id.split("/")[1],
            "index": str(feature),
        }
        for feature in features
    ]
    expected_features_encoded = urllib.parse.quote(json.dumps(expected_features))
    assert f"features={expected_features_encoded}" in url


@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
def test_open_neuronpedia_feature_dashboard(
    mock_open: MagicMock, gpt2_res_jb_l4_sae: SAE[StandardSAEConfig]
):
    index = 42

    open_neuronpedia_feature_dashboard(gpt2_res_jb_l4_sae, index)

    # Verify webbrowser.open was called with the correct URL
    neuronpedia_id = gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id
    assert neuronpedia_id is not None  # This should be true for the test fixture

    expected_url = f"{NEURONPEDIA_DOMAIN}/{neuronpedia_id}/{index}"
    mock_open.assert_called_once_with(expected_url)


@patch("sae_lens.analysis.neuronpedia_integration.webbrowser.open")
@patch("sae_lens.analysis.neuronpedia_integration.logger.warning")
def test_open_neuronpedia_feature_dashboard_with_none_id(
    mock_warning: MagicMock,
    mock_open: MagicMock,
    gpt2_res_jb_l4_sae: SAE[StandardSAEConfig],
):
    # Temporarily set neuronpedia_id to None to test the warning path
    original_neuronpedia_id = gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id
    gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id = None

    try:
        index = 42
        open_neuronpedia_feature_dashboard(gpt2_res_jb_l4_sae, index)

        # Verify warning was logged
        mock_warning.assert_called_once_with(
            "SAE does not have a Neuronpedia ID. Either dashboards for this SAE do not exist (yet) on Neuronpedia, or the SAE was not loaded via the from_pretrained method"
        )

        # Verify webbrowser.open was not called
        mock_open.assert_not_called()
    finally:
        # Restore original neuronpedia_id
        gpt2_res_jb_l4_sae.cfg.metadata.neuronpedia_id = original_neuronpedia_id


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.requests.get")
@patch("sae_lens.analysis.neuronpedia_integration.requests.post")
@patch("sae_lens.analysis.neuronpedia_integration.TokenActivationPairExplainer")
@patch("sae_lens.analysis.neuronpedia_integration.simulate_and_score")
@patch("sae_lens.analysis.neuronpedia_integration.os.makedirs")
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
@patch("builtins.open", new_callable=mock_open)
async def test_autointerp_neuronpedia_features_success(
    mock_file_open: MagicMock,
    mock_getenv: MagicMock,
    mock_makedirs: MagicMock,
    mock_simulate_and_score: MagicMock,
    mock_explainer_class: MagicMock,
    mock_requests_post: MagicMock,
    mock_requests_get: MagicMock,
):
    # Mock environment variable for OpenAI API key
    mock_getenv.return_value = "test_openai_key"

    # Mock the feature data from Neuronpedia API
    mock_feature_response = MagicMock()
    mock_feature_response.json.return_value = {
        "modelId": "gpt2-small",
        "layer": "0-res-jb",
        "index": 0,
        "activations": [
            {
                "id": "test_activation_1",
                "tokens": ["hello", "world"],
                "values": [0.5, 0.8],
            },
            {
                "id": "test_activation_2",
                "tokens": ["test", "tokens"],
                "values": [0.3, 0.6],
            },
        ],
    }
    mock_requests_get.return_value = mock_feature_response

    # Mock the API key test
    mock_api_test_response = MagicMock()
    mock_api_test_response.status_code = 200
    mock_requests_post.return_value = mock_api_test_response

    # Mock the explainer
    mock_explainer = MagicMock()
    mock_explanation = MagicMock()
    mock_explanation.rstrip.return_value = "This feature detects greetings"

    # Use AsyncMock for the async method
    mock_explainer.generate_explanations = AsyncMock(return_value=[mock_explanation])
    mock_explainer_class.return_value = mock_explainer

    # Mock the scorer
    mock_scored_simulation = MagicMock()
    mock_scored_simulation.get_preferred_score.return_value = 0.85
    mock_scored_simulation.scored_sequence_simulations = ["sim1", "sim2"]
    mock_simulate_and_score.return_value = mock_scored_simulation

    # Create test features
    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    # Call the function
    await autointerp_neuronpedia_features(
        features=features,
        openai_api_key="test_openai_key",
        neuronpedia_api_key="test_neuronpedia_key",
        autointerp_explainer_model_name="gpt-4-1106-preview",
        autointerp_scorer_model_name="gpt-3.5-turbo",
        num_activations_to_use=2,
        do_score=True,
        save_to_disk=True,
        upload_to_neuronpedia=True,
    )

    # Verify API calls were made
    mock_requests_get.assert_called_once()
    assert mock_requests_post.call_count == 2  # API key test + upload

    # Verify explainer was called
    mock_explainer.generate_explanations.assert_called_once()

    # Verify scorer was called
    mock_simulate_and_score.assert_called_once()

    # Verify file operations
    mock_makedirs.assert_called_once()
    mock_file_open.assert_called_once()

    # Verify the feature was populated
    assert features[0].autointerp_explanation == "This feature detects greetings"
    assert features[0].autointerp_explanation_score == 0.85
    assert features[0].activations is not None
    assert len(features[0].activations) == 2


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_missing_openai_key(
    mock_getenv: MagicMock,
):
    mock_getenv.return_value = None

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key=None,
            neuronpedia_api_key="test_neuronpedia_key",
        )

    assert "You need to provide an OpenAI API key" in str(exc_info.value)


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_invalid_explainer_model(
    mock_getenv: MagicMock,
):
    mock_getenv.return_value = "test_openai_key"

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key="test_openai_key",
            neuronpedia_api_key="test_neuronpedia_key",
            autointerp_explainer_model_name="invalid-model",
        )

    assert "Invalid explainer model name" in str(exc_info.value)


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_invalid_scorer_model(
    mock_getenv: MagicMock,
):
    mock_getenv.return_value = "test_openai_key"

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key="test_openai_key",
            neuronpedia_api_key="test_neuronpedia_key",
            autointerp_explainer_model_name="gpt-4-1106-preview",
            autointerp_scorer_model_name="invalid-model",
            do_score=True,
        )

    assert "Invalid scorer model name" in str(exc_info.value)


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.requests.get")
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_missing_feature(
    mock_getenv: MagicMock, mock_requests_get: MagicMock
):
    mock_getenv.return_value = "test_openai_key"

    # Mock feature response that would cause the function to fail
    # The function checks for "modelId" key existence
    mock_feature_response = MagicMock()
    mock_feature_response.json.return_value = {
        "index": 999,  # Include index to avoid KeyError
        # Missing "modelId" key to simulate feature not found
    }
    mock_requests_get.return_value = mock_feature_response

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=999,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key="test_openai_key",
            neuronpedia_api_key="test_neuronpedia_key",
            upload_to_neuronpedia=False,
        )

    assert "does not exist" in str(exc_info.value)


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.requests.get")
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_no_activations(
    mock_getenv: MagicMock, mock_requests_get: MagicMock
):
    mock_getenv.return_value = "test_openai_key"

    # Mock feature with no activations
    mock_feature_response = MagicMock()
    mock_feature_response.json.return_value = {
        "modelId": "gpt2-small",
        "layer": "0-res-jb",
        "index": 0,
        "activations": [],
    }
    mock_requests_get.return_value = mock_feature_response

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key="test_openai_key",
            neuronpedia_api_key="test_neuronpedia_key",
            upload_to_neuronpedia=False,
        )

    assert "does not have activations" in str(exc_info.value)


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.requests.get")
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_dead_feature(
    mock_getenv: MagicMock, mock_requests_get: MagicMock
):
    mock_getenv.return_value = "test_openai_key"

    # Mock feature with zero activations (dead feature)
    mock_feature_response = MagicMock()
    mock_feature_response.json.return_value = {
        "modelId": "gpt2-small",
        "layer": "0-res-jb",
        "index": 0,
        "activations": [
            {
                "id": "test_activation_1",
                "tokens": ["hello", "world"],
                "values": [0.0, 0.0],  # All zeros - dead feature
            }
        ],
    }
    mock_requests_get.return_value = mock_feature_response

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key="test_openai_key",
            neuronpedia_api_key="test_neuronpedia_key",
            upload_to_neuronpedia=False,
        )

    assert "appears dead" in str(exc_info.value)


@pytest.mark.anyio
@patch("sae_lens.analysis.neuronpedia_integration.requests.get")
@patch("sae_lens.analysis.neuronpedia_integration.TokenActivationPairExplainer")
@patch("sae_lens.analysis.neuronpedia_integration.os.getenv")
async def test_autointerp_neuronpedia_features_without_scoring(
    mock_getenv: MagicMock,
    mock_explainer_class: MagicMock,
    mock_requests_get: MagicMock,
):
    mock_getenv.return_value = "test_openai_key"

    # Mock the feature data
    mock_feature_response = MagicMock()
    mock_feature_response.json.return_value = {
        "modelId": "gpt2-small",
        "layer": "0-res-jb",
        "index": 0,
        "activations": [
            {
                "id": "test_activation_1",
                "tokens": ["hello", "world"],
                "values": [0.5, 0.8],
            }
        ],
    }
    mock_requests_get.return_value = mock_feature_response

    # Mock the explainer
    mock_explainer = MagicMock()
    mock_explanation = MagicMock()
    mock_explanation.rstrip.return_value = "This feature detects greetings"

    # Use AsyncMock for the async method
    mock_explainer.generate_explanations = AsyncMock(return_value=[mock_explanation])
    mock_explainer_class.return_value = mock_explainer

    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    await autointerp_neuronpedia_features(
        features=features,
        openai_api_key="test_openai_key",
        neuronpedia_api_key="test_neuronpedia_key",
        autointerp_explainer_model_name="gpt-4-1106-preview",
        do_score=False,
        save_to_disk=False,
        upload_to_neuronpedia=False,
    )

    # Verify the feature was populated with explanation but no score
    assert features[0].autointerp_explanation == "This feature detects greetings"
    assert features[0].autointerp_explanation_score == 0.0  # Default value


@pytest.mark.anyio
async def test_autointerp_neuronpedia_features_missing_neuronpedia_key_for_upload():
    features = [
        NeuronpediaFeature(
            modelId="gpt2-small",
            layer=0,
            dataset="res-jb",
            feature=0,
        )
    ]

    with pytest.raises(Exception) as exc_info:
        await autointerp_neuronpedia_features(
            features=features,
            openai_api_key="test_openai_key",
            neuronpedia_api_key=None,
            upload_to_neuronpedia=True,
        )

    assert "You need to provide a Neuronpedia API key" in str(exc_info.value)


@pytest.mark.skip(
    reason="Need a way to test with an API key - maybe test to dev environment?"
)
def test_make_neuronpedia_list_with_features():
    make_neuronpedia_list_with_features(
        api_key="test_api_key",
        list_name="test_api",
        list_description="List descriptions are optional",
        features=[
            NeuronpediaFeature(
                modelId="gpt2-small",
                layer=0,
                dataset="att-kk",
                feature=11,
                description="List feature descriptions are optional as well.",
            ),
            NeuronpediaFeature(
                modelId="gpt2-small",
                layer=6,
                dataset="res_scefr-ajt",
                feature=7,
                description="You can add features from any model or SAE in one list.",
            ),
        ],
    )


@pytest.mark.skip(
    reason="Need a way to test with an API key - maybe test to dev environment?"
)
@pytest.mark.anyio
async def test_neuronpedia_autointerp():
    features = [
        NeuronpediaFeature(
            modelId="example-model",
            layer=0,
            dataset="test-np",
            feature=0,
        )
    ]

    await autointerp_neuronpedia_features(
        features=features,
        openai_api_key="your-oai-key",
        neuronpedia_api_key="your-np-key",
        autointerp_explainer_model_name="gpt-4-turbo-2024-04-09",
        autointerp_scorer_model_name="gpt-3.5-turbo",
        num_activations_to_use=5,
        do_score=False,
        save_to_disk=False,
        upload_to_neuronpedia=True,
    )
