import webbrowser
from unittest import mock

import pytest

from sae_lens.analysis.neuronpedia_integration import (
    FeatureInfo,
    NeuronpediaFeature,
    SaeInfo,
    autointerp_neuronpedia_features,
    get_neuronpedia_feature,
    get_neuronpedia_quick_list,
    make_neuronpedia_list_with_features,
)


def test_get_neuronpedia_feature():
    result = get_neuronpedia_feature(
        feature=0, layer=0, model="gpt2-small", dataset="res-jb"
    )

    assert result["modelId"] == "gpt2-small"
    assert result["layer"] == "0-res-jb"
    assert result["index"] == 0


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


def test_get_neuronpedia_quick_list(monkeypatch: pytest.MonkeyPatch):
    # Mock the webbrowser.open function
    mock_open = mock.Mock()
    monkeypatch.setattr(webbrowser, "open", mock_open)

    # Test with SaeInfo
    sae_info = SaeInfo(model_name="gpt2-small", neuronpedia_id="gpt2-small/0-res-jb")
    features = [0, 1, 2]
    url = get_neuronpedia_quick_list(sae_info, features, name="Test List")

    expected_url = "https://neuronpedia.org/quick-list/?name=Test%20List&features=%5B%7B%22modelId%22%3A%20%22gpt2-small%22%2C%20%22layer%22%3A%20%220-res-jb%22%2C%20%22index%22%3A%20%220%22%7D%2C%20%7B%22modelId%22%3A%20%22gpt2-small%22%2C%20%22layer%22%3A%20%220-res-jb%22%2C%20%22index%22%3A%20%221%22%7D%2C%20%7B%22modelId%22%3A%20%22gpt2-small%22%2C%20%22layer%22%3A%20%220-res-jb%22%2C%20%22index%22%3A%20%222%22%7D%5D"
    assert url == expected_url
    mock_open.assert_called_once_with(expected_url)

    # Reset mock
    mock_open.reset_mock()

    # Test with FeatureInfo
    features_info = [
        FeatureInfo(
            feature_index=0,
            description="Feature 0",
            model_name="gpt2-medium",
            neuronpedia_id="gpt2-medium/1-res-jb",
        ),
        FeatureInfo(
            feature_index=1,
            description="Feature 1",
            model_name="gpt2-large",
            neuronpedia_id="gpt2-large/2-att-kk",
        ),
    ]
    url = get_neuronpedia_quick_list(
        sae_info,
        features_info,
        name="Test List 2",
        description="Test Description",
        default_test_text="Hello, world!",
    )

    expected_url = "https://neuronpedia.org/quick-list/?name=Test%20List%202&description=Test%20Description&default_test_text=Hello%2C%20world%21&features=%5B%7B%22modelId%22%3A%20%22gpt2-medium%22%2C%20%22layer%22%3A%20%221-res-jb%22%2C%20%22index%22%3A%20%220%22%2C%20%22description%22%3A%20%22Feature%200%22%7D%2C%20%7B%22modelId%22%3A%20%22gpt2-large%22%2C%20%22layer%22%3A%20%222-att-kk%22%2C%20%22index%22%3A%20%221%22%2C%20%22description%22%3A%20%22Feature%201%22%7D%5D"
    assert url == expected_url
    mock_open.assert_called_once_with(expected_url)
