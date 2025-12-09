# %%

import gc
import itertools
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import circuitsvis as cv
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import torch as t
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (SAE, ActivationsStore, HookedSAETransformer,
                      LanguageModelSAERunnerConfig)
from sae_lens.toolkit.pretrained_saes_directory import \
    get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from tabulate import tabulate
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part32_interp_with_saes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# There's a single utils & tests file for both parts 3.1 & 3.2
import part31_superposition_and_saes.tests as tests
import part31_superposition_and_saes.utils as utils
from plotly_utils import imshow, line

MAIN = __name__ == "__main__"
# %%

if MAIN:
    t.manual_seed(2)

    W = t.randn(2, 5)
    W_normed = W / W.norm(dim=0, keepdim=True)

    imshow(
        W_normed.T @ W_normed,
        title="Cosine similarities of each pair of 2D feature embeddings",
        width=600,
    )

    utils.plot_features_in_2d(
        W_normed.unsqueeze(0),  # shape [instances=1 d_hidden=2 features=5]
    )

# %%

def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class ToyModelConfig:
    # We optimize n_inst models in a single training loop to let us sweep over sparsity or importance
    # curves efficiently. You should treat the number of instances `n_inst` like a batch dimension,
    # but one which is built into our training setup. Ignore the latter 3 arguments for now, they'll
    # return in later exercises.
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif", "normal"] = "unif"


class ToyModel(nn.Module):
    W: Float[Tensor, "inst d_hidden feats"]
    b_final: Float[Tensor, "inst feats"]

    # Our linear map (for a single instance) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: ToyModelConfig,
        feature_probability: float | Tensor = 0.01,
        importance: float | Tensor = 1.0,
        device=device,
    ):
        super(ToyModel, self).__init__()
        self.cfg = cfg

        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_inst, cfg.n_features)
        )
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_inst, cfg.n_features))

        self.W = nn.Parameter(
            nn.init.xavier_normal_(t.empty((cfg.n_inst, cfg.d_hidden, cfg.n_features)))
        )
        self.b_final = nn.Parameter(t.zeros((cfg.n_inst, cfg.n_features)))
        self.to(device)

    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        """
        Performs a single forward pass. For a single instance, this is given by:
            x -> ReLU(W.T @ W @ x + b_final)
        """
        return t.relu(
            einops.einsum(
                self.W.transpose(-1, -2),
                self.W,
                features,
                "i f1 h, i h f2, b i f2 -> b i f1"
            ) + self.b_final
        )

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data of shape (batch_size, n_instances, n_features).
        """
        # You'll fill this in later
        mask = (t.rand(batch_size, self.cfg.n_inst, self.cfg.n_features).to(device) > self.feature_probability).to(device)
        vals = t.rand(batch_size, self.cfg.n_inst, self.cfg.n_features).to(device)
        return t.masked_fill(vals, mask, 0).to(device)

    def calculate_loss(
        self,
        out: Float[Tensor, "batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch (as a scalar tensor), using this loss described in the
        Toy Models of Superposition paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """
        c = 1 / (out.size(0) * out.size(-1))
        loss = c * ((batch - out) ** 2) * self.importance
        result = loss.sum()
        return result

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 5_000,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(self.parameters(), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item() / self.cfg.n_inst, lr=step_lr)


if MAIN:
    tests.test_model(ToyModel)
    tests.test_generate_batch(ToyModel)
    tests.test_calculate_loss(ToyModel)

    cfg = ToyModelConfig(n_inst=8, n_features=5, d_hidden=2)

# %%

if MAIN:
    cfg = ToyModelConfig(n_inst=8, n_features=5, d_hidden=2)

    # importance varies within features for each instance
    importance = 0.9 ** t.arange(cfg.n_features)

    # sparsity is the same for all features in a given instance, but varies over instances
    feature_probability = 50 ** -t.linspace(0, 1, cfg.n_inst)

    line(
        importance,
        width=600,
        height=400,
        title="Importance of each feature (same over all instances)",
        labels={"y": "Feature importance", "x": "Feature"},
    )
    line(
        feature_probability,
        width=600,
        height=400,
        title="Feature probability (varied over instances)",
        labels={"y": "Probability", "x": "Instance"},
    )

    model = ToyModel(
        cfg=cfg,
        device=device,
        importance=importance[None, :],
        feature_probability=feature_probability[:, None],
    )
    model.optimize()


    utils.plot_features_in_2d(
        model.W,
        colors=model.importance,
        title=f"Superposition: {cfg.n_features} features represented in 2D space",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
    )

    # %%
    with t.inference_mode():
        batch = model.generate_batch(200)
        hidden = einops.einsum(
            batch,
            model.W,
            "batch instances features, instances hidden features -> instances hidden batch",
        )

    utils.plot_features_in_2d(hidden, title="Hidden state representation of a random batch of data")

    # %%

    cfg = ToyModelConfig(n_inst=10, n_features=100, d_hidden=20)

    importance = 100 ** -t.linspace(0, 1, cfg.n_features)
    feature_probability = 20 ** -t.linspace(0, 1, cfg.n_inst)

    line(
        importance,
        width=600,
        height=400,
        title="Importance of each feature (same over all instances)",
        labels={"y": "Feature importance", "x": "Feature"},
    )
    line(
        feature_probability,
        width=600,
        height=400,
        title="Feature probability (varied over instances)",
        labels={"y": "Probability", "x": "Instance"},
    )

    model = ToyModel(
        cfg=cfg,
        device=device,
        importance=importance[None, :],
        feature_probability=feature_probability[:, None],
    )
    model.optimize()

    # %%

    utils.plot_features_in_Nd(
        model.W,
        height=800,
        width=1600,
        title="ReLU output model: n_features = 100, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
        subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability],
    )

# %%

@dataclass
class ToySAEConfig:
    n_inst: int
    d_in: int
    d_sae: int
    sparsity_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    ste_epsilon: float = 0.01


class ToySAE(nn.Module):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(self, cfg: ToySAEConfig, model: ToyModel) -> None:
        super(ToySAE, self).__init__()

        assert cfg.d_in == model.cfg.d_hidden, "Model's hidden dim doesn't match SAE input dim"
        self.cfg = cfg
        self.model = model.requires_grad_(False)
        self.model.W.data[1:] = self.model.W.data[0]
        self.model.b_final.data[1:] = self.model.b_final.data[0]

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty(self.cfg.n_inst, self.cfg.d_in, self.cfg.d_sae))
        )
        if self.cfg.tied_weights:
            self._W_dec = None
        else:
            self._W_dec = nn.Parameter(
                nn.init.kaiming_uniform_(t.empty(self.cfg.n_inst, self.cfg.d_sae, self.cfg.d_in))
            )
        self.b_enc = nn.Parameter(t.zeros(self.cfg.n_inst, self.cfg.d_sae))
        self.b_dec = nn.Parameter(t.zeros(self.cfg.n_inst, self.cfg.d_in))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """
        Returns decoder weights, normalized over the autoencoder input dimension.
        """
        norms = t.norm(self.W_dec, dim=-1).unsqueeze(-1) + self.cfg.weight_normalize_eps
        return self.W_dec / norms

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst d_in"]:
        """
        Generates a batch of hidden activations from our model.
        """
        toy_model_batch = self.model.generate_batch(batch_size)
        return einops.einsum(
            self.model.W,
            toy_model_batch,
            "ni dh f, b ni f -> b ni dh"
        )

    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, "batch inst"],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict:       dict of different loss terms, each having shape (batch_size, n_inst)
            loss:            total loss (i.e. sum over terms of loss dict), same shape as loss values
            acts_post:       autoencoder latent activations, after applying ReLU
            h_reconstructed: reconstructed autoencoder input
        """
        # You'll fill this in later
        z = t.relu(
            einops.einsum(
                self.W_enc,
                h - self.b_dec,
                "ni din dsae, b ni din -> b ni dsae"
            ) + self.b_enc
        )
        o = einops.einsum(self.W_dec_normalized, z, "ni dsae din, b ni dsae -> b ni din") + self.b_dec
        recon_loss = t.sum((o - h) ** 2, dim=-1) / self.cfg.d_in
        l1_penalty = t.sum(z, dim=-1)
        return (
            {
                "L_reconstruction": recon_loss,
                "L_sparsity": l1_penalty
            },
            recon_loss + self.cfg.sparsity_coeff * l1_penalty,
            z,
            o,
        )

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        resample_method: Literal["simple", "advanced", None] = None,
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
        hidden_sample_size: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Optimizes the autoencoder using the given hyperparameters.

        Args:
            model:              we reconstruct features from model's hidden activations
            batch_size:         size of batches we pass through model & train autoencoder on
            steps:              number of optimization steps
            log_freq:           number of optimization steps between logging
            lr:                 learning rate
            lr_scale:           learning rate scaling function
            resample_method:    method for resampling dead latents
            resample_freq:      number of optimization steps between resampling dead latents
            resample_window:    number of steps needed for us to classify a neuron as dead
            resample_scale:     scale factor for resampled neurons
            hidden_sample_size: size of hidden value sample we add to the logs (for visualization)

        Returns:
            data_log:           dictionary containing data we'll use for visualization
        """
        assert resample_window <= resample_freq

        optimizer = t.optim.Adam(self.parameters(), lr=lr)  # betas=(0.0, 0.999)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists of dicts to store data we'll eventually be plotting
        data_log = []

        for step in progress_bar:
            # Resample dead latents
            if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                frac_active_in_window = t.stack(frac_active_list[-resample_window:], dim=0)
                if resample_method == "simple":
                    self.resample_simple(frac_active_in_window, resample_scale)
                elif resample_method == "advanced":
                    self.resample_advanced(frac_active_in_window, resample_scale, batch_size)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                h = self.generate_batch(batch_size)

            # Optimize
            loss_dict, loss, acts, _ = self.forward(h)
            loss.mean(0).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them directly (if not using tied weights)
            if not self.cfg.tied_weights:
                self.W_dec.data = self.W_dec_normalized.data

            # Calculate the mean sparsities over batch dim for each feature
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and log a bunch of values for creating plots / animations
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    lr=step_lr,
                    loss=loss.mean(0).sum().item(),
                    frac_active=frac_active.mean().item(),
                    **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                )
                with t.inference_mode():
                    loss_dict, loss, acts, h_r = self.forward(
                        h := self.generate_batch(hidden_sample_size)
                    )
                data_log.append(
                    {
                        "steps": step,
                        "frac_active": (acts.abs() > 1e-8).float().mean(0).detach().cpu(),
                        "loss": loss.detach().cpu(),
                        "h": h.detach().cpu(),
                        "h_r": h_r.detach().cpu(),
                        **{name: param.detach().cpu() for name, param in self.named_parameters()},
                        **{name: loss_term.detach().cpu() for name, loss_term in loss_dict.items()},
                    }
                )

        return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vecs
            - Set new values of W_dec and W_enc to be these normalized vecs, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        mask = frac_active_in_window.sum(dim=0) == 0  # inst d_sae
        n_dead = mask.sum()
        resample = t.randn(n_dead, self.cfg.d_in).to(self.W_enc.device)
        resample = resample / resample.norm(dim=-1).unsqueeze(-1)
        self.W_dec[mask] = resample
        einops.rearrange(self.W_enc, 'ninst din dsae -> ninst dsae din')[mask] = resample * resample_scale
        self.b_enc[mask] = 0

    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        """
        Resamples latents that have been dead for `dead_feature_window` steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vecs `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec & W_enc to be these centered & normalized vecs, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        raise NotImplementedError()

# %%

if MAIN:
    tests.test_sae_init(ToySAE)
    tests.test_sae_W_dec_normalized(ToySAE)
    tests.test_sae_generate_batch(ToySAE)
    tests.test_sae_forward(ToySAE)

# %%

if MAIN:
    d_hidden = d_in = 2
    n_features = d_sae = 5
    n_inst = 16

    # Create a toy model, and train it to convergence
    cfg = ToyModelConfig(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)
    model = ToyModel(cfg=cfg, device=device, feature_probability=0.025)
    model.optimize()

    sae = ToySAE(cfg=ToySAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)

    h = sae.generate_batch(512)


    utils.plot_features_in_2d(model.W[:8], title="Base model")
    utils.plot_features_in_2d(
        einops.rearrange(h[:, :8], "batch inst d_in -> inst d_in batch"),
        title="Hidden state representation of a random batch of data",
    )

    # %%

    data_log = sae.optimize(steps=20_000)

    utils.animate_features_in_2d(
        data_log,
        instances=list(range(8)),  # only plot the first 8 instances
        rows=["W_enc", "_W_dec"],
        filename=str(section_dir / "animation-training.html"),
        title="SAE on toy model",
    )

    # If this display code doesn't work, try opening the animation in your browser from where it gets saved
    with open(section_dir / "animation-training.html") as f:
        display(HTML(f.read()))

    # %%

    utils.frac_active_line_plot(
        frac_active=t.stack([data["frac_active"] for data in data_log]),
        title="Probability of sae features being active during training",
        avg_window=20,
    )

    # %%

    tests.test_resample_simple(ToySAE)

    # %%

    resampling_sae = ToySAE(cfg=ToySAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)

    resampling_data_log = resampling_sae.optimize(steps=20_000, resample_method="simple")

    utils.animate_features_in_2d(
        resampling_data_log,
        rows=["W_enc", "_W_dec"],
        instances=list(range(8)),  # only plot the first 8 instances
        filename=str(section_dir / "animation-training-resampling.html"),
        color_resampled_latents=True,
        title="SAE on toy model (with resampling)",
    )

    utils.frac_active_line_plot(
        frac_active=t.stack([data["frac_active"] for data in resampling_data_log]),
        title="Probability of sae features being active during training",
        avg_window=20,
    )

    # %%

    utils.animate_features_in_2d(
        resampling_data_log,
        rows=["W_enc", "h", "h_r"],
        instances=list(range(4)),  # plotting fewer instances for a smaller animation file size
        color_resampled_latents=True,
        filename=str(section_dir / "animation-training-reconstructions.html"),
        title="SAE on toy model (showing hidden states & reconstructions)",
    )

# %%
