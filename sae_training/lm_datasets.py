from datasets import load_dataset

# To do: preprocess_tokenized_dataset, preprocess_text_dataset, preprocess other dataset
def preprocess_tokenized_dataset(source_batch: dict, context_size: int) -> dict:
    tokenized_prompts = source_batch["tokens"]

    # Chunk each tokenized prompt into blocks of context_size,
    # discarding the last block if too small.
    context_size_prompts = []
    for encoding in tokenized_prompts:
        chunks = [
            encoding[i : i + context_size]
            for i in range(0, len(encoding), context_size)
            if len(encoding[i : i + context_size]) == context_size
        ]
        context_size_prompts.extend(chunks)

    return {"input_ids": context_size_prompts}


def get_mapped_dataset(cfg):
    # Load the dataset
    context_size = cfg["context_size"]
    dataset_path = cfg["dataset_path"]
    dataset_split = "train"
    buffer_size: int = 1000,
    preprocess_batch_size: int = 1000,

    dataset = load_dataset(dataset_path, streaming=True, split=dataset_split)  # type: ignore

    # Setup preprocessing
    existing_columns = list(next(iter(dataset)).keys())
    mapped_dataset = dataset.map(
        preprocess_tokenized_dataset, # preprocess is what differentiates different datasets
        batched=True,
        batch_size=preprocess_batch_size,
        fn_kwargs={"context_size": context_size},
        remove_columns=existing_columns,
    )

    # Setup approximate shuffling. As the dataset is streamed, this just pre-downloads at least
    # `buffer_size` items and then shuffles just that buffer.
    # https://huggingface.co/docs/datasets/v2.14.5/stream#shuffle
    dataset = mapped_dataset.shuffle(buffer_size=buffer_size)
    return dataset

