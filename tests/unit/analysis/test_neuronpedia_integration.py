import pytest

from sae_lens.analysis.neuronpedia_integration import (
    NeuronpediaListFeature,
    get_neuronpedia_feature,
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
            NeuronpediaListFeature(
                modelId="gpt2-small",
                layer=0,
                dataset="att-kk",
                feature=11,
                description="List feature descriptions are optional as well.",
            ),
            NeuronpediaListFeature(
                modelId="gpt2-small",
                layer=6,
                dataset="res_scefr-ajt",
                feature=7,
                description="You can add features from any model or SAE in one list.",
            ),
        ],
    )
