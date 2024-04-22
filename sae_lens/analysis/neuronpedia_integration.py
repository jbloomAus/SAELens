import json
import urllib.parse
import webbrowser

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
