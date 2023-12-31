
"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import gzip
import os
import pickle
from functools import partial

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
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
        # move x to correct dtype
        x = x.to(self.dtype)
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
        
        # add config for whether l2 is normalized:
        mse_loss = (torch.pow((sae_out-x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()).mean()
        # mse_loss = (sae_out.float() - x.float()).pow(2).sum(-1).mean(0)
        sparsity = torch.abs(feature_acts).sum(dim=1).mean(dim=(0,)) 
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss

        return sae_out, feature_acts, loss, mse_loss, l1_loss

    @torch.no_grad()
    def resample_neurons_l2(
        self,
        x: Float[Tensor, "batch_size n_hidden"],
        feature_sparsity: Float[Tensor, "n_hidden_ae"],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        
        I'll probably break this now and fix it later!
        '''
        
        feature_reinit_scale = self.cfg.feature_reinit_scale
        
        sae_out, _, _, _, _ = self.forward(x)
        per_token_l2_loss = (sae_out - x).pow(2).sum(dim=-1).squeeze()

        # Find the dead neurons in this instance. If all neurons are alive, continue
        is_dead = (feature_sparsity < self.cfg.dead_feature_threshold)
        dead_neurons = torch.nonzero(is_dead).squeeze(-1)
        alive_neurons = torch.nonzero(~is_dead).squeeze(-1)
        n_dead = dead_neurons.numel()
        
        if n_dead == 0:
            return 0 # If there are no dead neurons, we don't need to resample neurons
        
        # Compute L2 loss for each element in the batch
        # TODO: Check whether we need to go through more batches as features get sparse to find high l2 loss examples. 
        if per_token_l2_loss.max() < 1e-6:
            return 0 # If we have zero reconstruction loss, we don't need to resample neurons
        
        # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss squared
        per_token_l2_loss = per_token_l2_loss.to(torch.float32) # wont' work with bfloat16
        distn = Categorical(probs = per_token_l2_loss.pow(2) / (per_token_l2_loss.pow(2).sum()))
        replacement_indices = distn.sample((n_dead,)) # shape [n_dead]

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (x - self.b_dec)[replacement_indices] # shape [n_dead n_input_ae]

        # unit norm
        replacement_values = (replacement_values / (replacement_values.norm(dim=1, keepdim=True) + 1e-8))

        # St new decoder weights
        self.W_dec.data[is_dead, :] = replacement_values

        # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
        W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.W_enc[:, alive_neurons].norm(dim=0).mean().item()
        
        # Lastly, set the new weights & biases
        self.W_enc.data[:, is_dead] = (replacement_values * W_enc_norm_alive_mean * feature_reinit_scale).T
        self.b_enc.data[is_dead] = 0.0
        
        
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, is_dead] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][is_dead] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_in)
                    v[v_key][is_dead, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,)
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
                
        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, is_dead].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )
        
        return n_dead

    @torch.no_grad()
    def resample_neurons_anthropic(
        self, 
        dead_neuron_indices, 
        model,
        optimizer, 
        activation_store):
        """
        Arthur's version of Anthropic's feature resampling
        procedure.
        """
        # collect global loss increases, and input activations
        global_loss_increases, global_input_activations = self.collect_anthropic_resampling_losses(
            model, activation_store
        )

        # sample according to losses
        probs = global_loss_increases / global_loss_increases.sum()
        sample_indices = torch.multinomial(
            probs,
            min(len(dead_neuron_indices), probs.shape[0]),
            replacement=False,
        )
        # if we don't have enough samples for for all the dead neurons, take the first n
        if sample_indices.shape[0] < len(dead_neuron_indices):
            dead_neuron_indices = dead_neuron_indices[:sample_indices.shape[0]]

        # Replace W_dec with normalized differences in activations
        self.W_dec.data[dead_neuron_indices, :] = (
            (
                global_input_activations[sample_indices]
                / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
            )
            .to(self.dtype)
            .to(self.device)
        )
        
        # Lastly, set the new weights & biases
        self.W_enc.data[:, dead_neuron_indices] = self.W_dec.data[dead_neuron_indices, :].T
        self.b_enc.data[dead_neuron_indices] = 0.0
        
        # Reset the Encoder Weights
        if dead_neuron_indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(dead_neuron_indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(dead_neuron_indices))
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale * average_norm

            # Set biases to resampled value
            relevant_biases = self.b_enc.data[dead_neuron_indices].mean()
            self.b_enc.data[dead_neuron_indices] = relevant_biases * 0 # bias resample factor (put in config?)

        else:
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale
            self.b_enc.data[dead_neuron_indices] = -5.0
        
        # TODO: Refactor this resetting to be outside of resampling.
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, dead_neuron_indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][dead_neuron_indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_in)
                    v[v_key][dead_neuron_indices, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,)
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
                
        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, dead_neuron_indices].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )
        
        return 


    @torch.no_grad()
    def collect_anthropic_resampling_losses(self, model, activation_store):
        """
        Collects the losses for resampling neurons (anthropic)
        """
        
        batch_size = self.cfg.store_batch_size
        
        # we're going to collect this many forward passes
        number_final_activations = self.cfg.resample_batches * batch_size
        # but have seq len number of tokens in each
        number_activations_total = number_final_activations * self.cfg.context_size
        anthropic_iterator = range(0, number_final_activations, batch_size)
        anthropic_iterator = tqdm(anthropic_iterator, desc="Collecting losses for resampling...")
        
        global_loss_increases = torch.zeros((number_final_activations,), dtype=self.dtype, device=self.device)
        global_input_activations = torch.zeros((number_final_activations, self.d_in), dtype=self.dtype, device=self.device)

        for refill_idx in anthropic_iterator:
            
            # get a batch, calculate loss with/without using SAE reconstruction.
            batch_tokens = activation_store.get_batch_tokens()
            ce_loss_with_recons = self.get_test_loss(batch_tokens, model)
            ce_loss_without_recons, normal_activations_cache = model.run_with_cache(
                batch_tokens,
                names_filter=self.cfg.hook_point,
                return_type = "loss",
                loss_per_token = True,
            )
            # ce_loss_without_recons = model.loss_fn(normal_logits, batch_tokens, True)
            # del normal_logits
            
            normal_activations = normal_activations_cache[self.cfg.hook_point]
            if self.cfg.hook_point_head_index is not None:
                normal_activations = normal_activations[:,:,self.cfg.hook_point_head_index]

            # calculate the difference in loss
            changes_in_loss = ce_loss_with_recons - ce_loss_without_recons
            changes_in_loss = changes_in_loss.cpu()
            
            # sample from the loss differences
            probs = F.relu(changes_in_loss) / F.relu(changes_in_loss).sum(dim=1, keepdim=True)
            changes_in_loss_dist = Categorical(probs)
            samples = changes_in_loss_dist.sample()
            
            assert samples.shape == (batch_size,), f"{samples.shape=}; {self.cfg.store_batch_size=}"
            
            end_idx = refill_idx + batch_size
            global_loss_increases[refill_idx:end_idx] = changes_in_loss[torch.arange(batch_size), samples]
            global_input_activations[refill_idx:end_idx] = normal_activations[torch.arange(batch_size), samples]
        
        return global_loss_increases, global_input_activations
    
    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        head_index = self.cfg.hook_point_head_index
        
        def standard_replacement_hook(activations, hook):
            activations = self.forward(activations)[0].to(activations.dtype)
            return activations
        
        def head_replacement_hook(activations, hook):
            new_actions = self.forward(activations[:,:,head_index])[0].to(activations.dtype)
            activations[:,:,head_index] = new_actions
            return activations

        replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
        
        ce_loss_with_recons = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
        )
        
        return ce_loss_with_recons
        

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
        
        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz")
        
        
        print(f"Saved model to {path}")
    
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
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    state_dict = torch.load(path, map_location="mps")
                    state_dict["cfg"].device = "mps"
                else:
                    state_dict = torch.load(path)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")
            
        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, 'rb') as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl.gz file: {e}")
        elif path.endswith(".pkl"):
            try:
                with open(path, 'rb') as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz")

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if 'cfg' not in state_dict or 'state_dict' not in state_dict:
            raise ValueError("The loaded state dictionary must contain 'cfg' and 'state_dict' keys")

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name