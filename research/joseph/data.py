from typing import List

import numpy as np
import torch
from datasets import load_dataset

LENGTH_RANDOM_TOKS = 4
TOKEN_OF_INTEREST = " John"
N_REPEAT_TOKENS = 3

def generate_random_token_prompt(model, n_random_tokens = 10, n_repeat_tokens = 3, token_of_interest: str = " John"):
    
    random_tokens = torch.randint(0, model.tokenizer.vocab_size, (n_random_tokens,)).to(model.cfg.device)
    # append the token id for " John"
    if token_of_interest is not None:
        john_token = torch.tensor(model.to_single_token(token_of_interest)).unsqueeze(0).to(model.cfg.device)
        random_tokens = torch.cat([john_token, random_tokens], dim=0)
    
    # repeat the tokens 
    random_tokens = random_tokens.repeat(n_repeat_tokens)
    
    # generate an index for each group of tokens
    random_token_groups = torch.arange(0, n_repeat_tokens).unsqueeze(-1).repeat(1, LENGTH_RANDOM_TOKS+1).flatten()
    
    return random_tokens, random_token_groups


def get_webtext(seed: int = 420, dataset="stas/openwebtext-10k") -> List[str]:
    """Get 10,000 sentences from the OpenWebText dataset"""

    # Let's see some WEBTEXT
    raw_dataset = load_dataset(dataset)
    train_dataset = raw_dataset["train"]
    dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

    # Shuffle the dataset (I don't want the Hitler thing being first so use a seeded shuffle)
    np.random.seed(seed)
    np.random.shuffle(dataset)

    return dataset

