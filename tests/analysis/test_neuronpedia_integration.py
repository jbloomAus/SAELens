import json
import urllib.parse
from unittest.mock import MagicMock, patch

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
