import json
import urllib.parse
import webbrowser
from typing import Optional

import requests


def get_neuronpedia_quick_list(
    features: list[int],
    layer: int,
    model: str = "gpt2-small",
    dataset: str = "res-jb",
    name: str = "temporary_list",
):
    url = "https://neuronpedia.org/quick-list/"
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


def get_neuronpedia_feature(
    feature: int,
    layer: int,
    model: str = "gpt2-small",
    dataset: str = "res-jb",
):
    url = "https://neuronpedia.org/api/feature/"
    url = url + f"{model}/{layer}-{dataset}/{feature}"

    result = requests.get(url).json()
    result["index"] = int(result["index"])

    return result


class NeuronpediaListFeature(object):
    modelId = ""
    layer = 0
    dataset = ""
    index = 0
    description = ""

    def __init__(
        self,
        modelId: str,
        layer: int,
        dataset: str,
        feature: int,
        description: str = "",
    ):
        self.modelId = modelId
        self.layer = layer
        self.dataset = dataset
        self.feature = feature
        self.description = description


def make_neuronpedia_list_with_features(
    api_key: str,
    list_name: str,
    features: list[NeuronpediaListFeature],
    list_description: Optional[str] = None,
    open_browser: bool = True,
):
    url = "https://neuronpedia.org/api/list/new-with-features"

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
