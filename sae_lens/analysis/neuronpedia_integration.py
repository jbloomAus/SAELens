import webbrowser


def open_neuronpedia(
    feature_id: int, layer: int = 0, model: str = "gpt2-small", dataset: str = "res-jb"
):

    path_to_html = f"https://www.neuronpedia.org/{model}/{layer}-{dataset}/{feature_id}"

    print(f"Feature {feature_id}")
    webbrowser.open_new_tab(path_to_html)
