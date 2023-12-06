
"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import os

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from transformer_lens.hook_points import HookedRootModule, HookPoint


class SparseAutoencoder(HookedRootModule):
    """
    
    """
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
        self.l1_coefficient = cfg.l1_coefficient
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

    def forward(self, x):
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
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )
        
        mse_loss = (sae_out.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coefficient * torch.abs(feature_acts).sum()
        loss = mse_loss + l1_loss

        return sae_out, feature_acts, loss, mse_loss, l1_loss


    @torch.no_grad()
    def resample_neurons(
        self,
        x: Float[Tensor, "batch_size n_hidden"],
        feature_sparsity: Float[Tensor, "n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        sae_out, _, _, _, _ = self.forward(x)
        per_token_l2_loss = (sae_out - x).pow(2).sum(dim=-1).squeeze()

        # Find the dead neurons in this instance. If all neurons are alive, continue
        is_dead = (feature_sparsity < 1e-8)
        dead_neurons = torch.nonzero(is_dead).squeeze(-1)
        alive_neurons = torch.nonzero(~is_dead).squeeze(-1)
        n_dead = dead_neurons.numel()
        
        if n_dead == 0:
            return 0 # If there are no dead neurons, we don't need to resample neurons
        
        # Compute L2 loss for each element in the batch
        # TODO: Check whether we need to go through more batches as features get sparse to find high l2 loss examples. 
        if per_token_l2_loss.max() < 1e-6:
            return 0 # If we have zero reconstruction loss, we don't need to resample neurons
        
        # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
        distn = Categorical(probs = per_token_l2_loss.pow(2) / (per_token_l2_loss.pow(2).sum()))
        n_samples = n_dead#min(n_dead, feature_sparsity.shape[-1] // self.cfg.expansion_factor) # don't reinit more than 10% of neurons at a time
        replacement_indices = distn.sample((n_samples,)) # shape [n_dead]

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (x - self.b_dec)[replacement_indices] # shape [n_dead n_input_ae]

        # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
        W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.W_enc[:, alive_neurons].norm(dim=0).mean().item()
        
        # Use this to renormalize the replacement values
        replacement_values = (replacement_values / (replacement_values.norm(dim=1, keepdim=True) + 1e-8)) * W_enc_norm_alive_mean * neuron_resample_scale

        # Lastly, set the new weights & biases
        d_neurons_to_be_replaced = dead_neurons[:n_samples] # not restarting all!
        self.W_enc.data[:, d_neurons_to_be_replaced] = replacement_values.T
        self.b_enc.data[d_neurons_to_be_replaced] = 0.0
        
        return n_samples

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        '''
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        '''
        
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
    
    def save_model(self, path: str):
        '''
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        '''
        
        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        
        state_dict = {
            "cfg": self.cfg,
            "state_dict": self.state_dict()
        }
        
        torch.save(state_dict, path)
        
        print(f"Saved model to {path}")
        
    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name
    
    @classmethod
    def load_from_pretrained(cls, path: str):
        '''
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        '''

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        try:
            state_dict = torch.load(path)
        except Exception as e:
            raise IOError(f"Error loading the state dictionary: {e}")

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if 'cfg' not in state_dict or 'state_dict' not in state_dict:
            raise ValueError("The loaded state dictionary must contain 'cfg' and 'state_dict' keys")

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance
