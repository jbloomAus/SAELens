import pytest

from sae_lens.analysis.neuronpedia_integration import (
    NeuronpediaFeature,
    autointerp_neuronpedia_features,
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
