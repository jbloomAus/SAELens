from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_feature


def test_get_neuronpedia_feature():
    result = get_neuronpedia_feature(
        feature=0, layer=0, model="gpt2-small", dataset="res-jb"
    )

    assert result["modelId"] == "gpt2-small"
    assert result["layer"] == "0-res-jb"
    assert result["index"] == 0
