
#%%
"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""
from typing import Literal

import einops
import torch
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint


#%%
# TODO make sure that W_dec stays unit norm during training
class SAE(HookedRootModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.dtype = cfg.dtype
        self.device = cfg.device

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, return_mode: Literal["sae_out", "hidden_post", "both"]="both"):
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        hidden_post = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                hidden_post,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        if return_mode == "sae_out":
            return sae_out
        elif return_mode == "hidden_post":
            return hidden_post
        elif return_mode == "both":
            return sae_out, hidden_post
        else:
            raise ValueError(f"Unexpected {return_mode=}")

    def reinit_neurons(self, indices):
        new_W_enc = torch.nn.init.kaiming_uniform_(
            torch.empty(
                self.d_in, indices.shape[0], dtype=self.dtype, device=self.device
            )
        ) * self.cfg["resample_factor"]
        new_b_enc = torch.zeros(
            indices.shape[0], dtype=self.dtype, device=self.device
        )
        new_W_dec = torch.nn.init.kaiming_uniform_(
            torch.empty(
                indices.shape[0], self.d_in, dtype=self.dtype, device=self.get_test_lossevice
            )
        )
        self.W_enc.data[:, indices] = new_W_enc
        self.b_enc.data[indices] = new_b_enc
        self.W_dec.data[indices, :] = new_W_dec
        self.W_dec /= torch.norm(self.W_dec, dim=1, keepdim=True)

