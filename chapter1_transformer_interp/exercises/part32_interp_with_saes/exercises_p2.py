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
    gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    gpt2_saes = {
        layer: SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=str(device),
        )[0]
        for layer in tqdm(range(gpt2.cfg.n_layers))
    }

# %%

class SparseTensor:
    """
    Handles 2D tensor data (assumed to be non-negative) in 2 different formats:
        dense:  The full tensor, which contains zeros. Shape is (n1, ..., nk).
        sparse: A tuple of nonzero values with shape (n_nonzero,), nonzero indices with shape
                (n_nonzero, k), and the shape of the dense tensor.
    """

    sparse: tuple[Tensor, Tensor, tuple[int, ...]]
    dense: Tensor

    def __init__(self, sparse: tuple[Tensor, Tensor, tuple[int, ...]], dense: Tensor):
        self.sparse = sparse
        self.dense = dense

    @classmethod
    def from_dense(cls, dense: Tensor) -> "SparseTensor":
        sparse_idx = dense.nonzero()
        return cls(
            (
                dense[tuple(sparse_idx.transpose(0, 1))],
                sparse_idx,
                dense.shape,
            ),
            dense,
        )

    @classmethod
    def from_sparse(cls, sparse: tuple[Tensor, Tensor, tuple[int, ...]]) -> "SparseTensor":
        vals, idx, shape = sparse
        dense = t.zeros(shape, dtype=vals.dtype, device=vals.device)
        dense[tuple(idx.transpose(0, 1).to(dense.device))] = vals
        return cls(
            sparse,
            dense,
        )

    @property
    def values(self) -> Tensor:
        return self.sparse[0].squeeze()

    @property
    def indices(self) -> Tensor:
        return self.sparse[1].squeeze()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sparse[2]


if MAIN:
    # Test `from_dense`
    x = t.zeros(10_000)
    nonzero_indices = t.randint(0, 10_000, (10,)).sort().values
    nonzero_values = t.rand(10)
    x[nonzero_indices] = nonzero_values
    sparse_tensor = SparseTensor.from_dense(x)
    t.testing.assert_close(sparse_tensor.sparse[0], nonzero_values)
    t.testing.assert_close(sparse_tensor.sparse[1].squeeze(-1), nonzero_indices)
    t.testing.assert_close(sparse_tensor.dense, x)

    # Test `from_sparse`
    sparse_tensor = SparseTensor.from_sparse(
        (nonzero_values, nonzero_indices.unsqueeze(-1), tuple(x.shape))
    )
    t.testing.assert_close(sparse_tensor.dense, x)

    # Verify other properties
    t.testing.assert_close(sparse_tensor.values, nonzero_values)
    t.testing.assert_close(sparse_tensor.indices, nonzero_indices)

# %%

def latent_acts_to_later_latent_acts(
    latent_acts_nonzero: Float[Tensor, "nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Given some latent activations for a residual stream SAE earlier in the model, computes the
    latent activations of a later SAE. It does this by mapping the latent activations through the
    path SAE decoder -> intermediate model layers -> later SAE encoder.

    This function must input & output sparse information (i.e. nonzero values and their indices)
    rather than dense tensors, because latent activations are sparse but jacrev() doesn't support
    gradients on real sparse tensors.
    """
    latents_from = SparseTensor.from_sparse((latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape))
    reconstructed_from = sae_from.decode(latents_from.dense)

    resid_to = model.forward(
        reconstructed_from,
        start_at_layer = sae_from.cfg.hook_layer,
        stop_at_layer=sae_to.cfg.hook_layer,
    )

    latents_to = sae_to.encode(resid_to)
    latent_acts_next_recon = SparseTensor.from_dense(latents_to)

    return latent_acts_next_recon.sparse[0], (latent_acts_next_recon.dense,)

def latent_to_latent_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor, SparseTensor, SparseTensor]:
    """
    Computes the gradients between all active pairs of latents belonging to two SAEs.

    Returns:
        latent_latent_gradients:    The gradients between all active pairs of latents
        latent_acts_prev:           The latent activations of the first SAE
        latent_acts_next:           The latent activations of the second SAE
        latent_acts_next_recon:     The reconstructed latent activations of the second SAE (i.e.
                                    based on the first SAE's reconstructions)
    """
    sae_from_name = f"{sae_from.cfg.hook_name}.hook_sae_acts_post"
    sae_to_name = f"{sae_to.cfg.hook_name}.hook_sae_acts_post"
    _, cache = model.run_with_cache_with_saes(
        tokens,
        saes=[sae_from, sae_to],
        use_error_term=True,
        names_filter=[
            sae_from_name,
            sae_to_name,
        ],
    )
    latent_acts_prev = SparseTensor.from_dense(cache[sae_from_name])
    latent_acts_next = SparseTensor.from_dense(cache[sae_to_name])
    latent_acts_to_later_latent_acts_and_gradients = t.func.jacrev(
        latent_acts_to_later_latent_acts, argnums=0, has_aux=True
    )
    latent_latent_gradients, (latent_acts_next_recon_dense,) = latent_acts_to_later_latent_acts_and_gradients(
        *latent_acts_prev.sparse, 
        sae_from,
        sae_to,
        model,
    )
    latent_acts_next_recon = SparseTensor.from_dense(latent_acts_next_recon_dense)

    return (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    )

if MAIN:
    prompt = "The Eiffel tower is in Paris"
    tokens = gpt2.to_tokens(prompt)
    str_toks = gpt2.to_str_tokens(prompt)
    layer_from = 0
    layer_to = 3

    # Get latent-to-latent gradients
    t.cuda.empty_cache()
    t.set_grad_enabled(True)
    (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    ) = latent_to_latent_gradients(tokens, gpt2_saes[layer_from], gpt2_saes[layer_to], gpt2)
    t.set_grad_enabled(False)

    # Verify that ~the same latents are active in both, and the MSE loss is small
    nonzero_latents = [tuple(x) for x in latent_acts_next.indices.tolist()]
    nonzero_latents_recon = [tuple(x) for x in latent_acts_next_recon.indices.tolist()]
    alive_in_one_not_both = set(nonzero_latents) ^ set(nonzero_latents_recon)
    print(f"# nonzero latents (true): {len(nonzero_latents)}")
    print(f"# nonzero latents (reconstructed): {len(nonzero_latents_recon)}")
    print(f"# latents alive in one but not both: {len(alive_in_one_not_both)}")

    px.imshow(
        to_numpy(latent_latent_gradients.T),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[
            f"F{layer_to}.{latent}, {str_toks[seq]!r} ({seq})"
            for (_, seq, latent) in latent_acts_next_recon.indices
        ],
        y=[
            f"F{layer_from}.{latent}, {str_toks[seq]!r} ({seq})"
            for (_, seq, latent) in latent_acts_prev.indices
        ],
        labels={"x": f"To layer {layer_to}", "y": f"From layer {layer_from}"},
        title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1600,
        height=1000,
    ).show()

# %%

def tokens_to_latent_acts(
    token_scales: Float[Tensor, "batch seq"],
    tokens: Int[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Given scale factors for model's embeddings (i.e. scale factors applied after we compute the sum
    of positional and token embeddings), returns the SAE's latents.

    Returns:
        latent_acts_sparse: The SAE's latents in sparse form (i.e. the tensor of values)
        latent_acts_dense:  The SAE's latents in dense tensor, in a length-1 tuple
    """
    tok_embed = model.W_E[tokens]  # batch seq dm
    pos_embed = model.W_pos[:tokens.shape[1]]  # seq dm
    inputs = tok_embed + pos_embed
    layer0_resid = inputs * token_scales.unsqueeze(-1)  # batch seq dm
    sae_input = model(
        layer0_resid,
        start_at_layer=0,
        stop_at_layer=sae.cfg.hook_layer,
    )
    sae_latents = SparseTensor.from_dense(sae.encode(sae_input))

    return sae_latents.sparse[0], (sae_latents.dense,)


def token_to_latent_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor]:
    """
    Computes the gradients between an SAE's latents and all input tokens.

    Returns:
        token_latent_grads: The gradients between input tokens and SAE latents
        latent_acts:        The SAE's latent activations
    """
    tokens_to_latent_acts_and_grads = t.func.jacrev(
        tokens_to_latent_acts, argnums=0, has_aux=True
    )
    token_latent_grads, (latent_acts_dense,) = tokens_to_latent_acts_and_grads(
        t.ones(tokens.shape, device=device, requires_grad=True),
        tokens,
        sae,
        model,
    )

    return (
        einops.rearrange(
            token_latent_grads,
            "lat b seq -> b seq lat",
        ),
        SparseTensor.from_dense(latent_acts_dense),
    )

if MAIN:
    sae_layer = 3
    token_latent_grads, latent_acts = token_to_latent_gradients(
        tokens, sae=gpt2_saes[sae_layer], model=gpt2
    )

    px.imshow(
        to_numpy(token_latent_grads[0]),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[
            f"F{sae_layer}.{latent:05}, {str_toks[seq]!r} ({seq})"
            for (_, seq, latent) in latent_acts.indices
        ],
        y=[f"{str_toks[i]!r} ({i})" for i in range(len(str_toks))],
        labels={"x": f"To layer {sae_layer}", "y": "From tokens"},
        title=f'Gradients between input tokens and SAE latents in layer {sae_layer}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1900,
        height=450,
    )

# %%

def latent_acts_to_logits(
    latent_acts_nonzero: Float[Tensor, "nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae: SAE,
    model: HookedSAETransformer,
    token_ids: list[int] | None = None,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Computes the logits as a downstream function of the SAE's reconstructed residual stream. If we
    supply `token_ids`, it means we only compute & return the logits for those specified tokens.
    """
    latent_acts = SparseTensor.from_sparse(
        (latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape),
    ).dense

    logits_recon = model.forward(
        sae.decode(latent_acts),
        start_at_layer=sae.cfg.hook_layer,
        return_type="logits", 
    )[0, -1]
    return logits_recon[token_ids], (logits_recon,)


def latent_to_logit_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
    k: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[int] | None, SparseTensor]:
    """
    Computes the gradients between active latents and some top-k set of logits (we
    use k to avoid having to compute the gradients for all tokens).

    Returns:
        latent_logit_gradients:  The gradients between the SAE's active latents & downstream logits
        logits:                  The model's true logits
        logits_recon:            The model's reconstructed logits (i.e. based on SAE reconstruction)
        token_ids:               The tokens we computed the gradients for
        latent_acts:             The SAE's latent activations
    """
    assert tokens.shape[0] == 1, "Only supports batch size 1 for now"

    sae_acts_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    with t.no_grad():
        logits, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            names_filter=[sae_acts_name],
            use_error_term=True,
            return_type="logits",
        )

    logits = logits[0][-1]  # vocab
    latent_acts = cache[sae_acts_name]  # b seq dsae
    latent_acts_sparse_tensor = SparseTensor.from_dense(latent_acts)
    top_logits = t.topk(logits, k=k)  # k

    latent_acts_to_logits_and_grads = t.func.jacrev(
        latent_acts_to_logits, argnums=0, has_aux=True
    )
    latent_logit_gradients, (logits_recon,) = latent_acts_to_logits_and_grads(
        *latent_acts_sparse_tensor.sparse,
        sae,
        model,
        top_logits.indices,
    )

    return (
        latent_logit_gradients,
        logits,
        logits_recon,
        top_logits.indices.tolist(),
        latent_acts_sparse_tensor,
    )


if MAIN:
    layer = 9
    prompt = "The Eiffel tower is in the city of"
    answer = " Paris"

    tokens = gpt2.to_tokens(prompt, prepend_bos=True)
    str_toks = gpt2.to_str_tokens(prompt, prepend_bos=True)
    k = 25

    # Test the model on this prompt, with & without SAEs
    test_prompt(prompt, answer, gpt2)

    # How about the reconstruction? More or less; it's rank 20 so still decent
    gpt2_saes[layer].use_error_term = False
    with gpt2.saes(saes=[gpt2_saes[layer]]):
        test_prompt(prompt, answer, gpt2)

    latent_logit_grads, logits, logits_recon, token_ids, latent_acts = latent_to_logit_gradients(
        tokens, sae=gpt2_saes[layer], model=gpt2, k=k
    )

    # sort by most positive in " Paris" direction
    sorted_indices = latent_logit_grads[0].argsort(descending=True)
    latent_logit_grads = latent_logit_grads[:, sorted_indices]
    print(token_ids)

    px.imshow(
        to_numpy(latent_logit_grads),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[
            f"{str_toks[seq]!r} ({seq}), latent {latent:05}"
            for (_, seq, latent) in latent_acts.indices[sorted_indices]
        ],
        y=[f"{tok!r} ({gpt2.to_single_str_token(tok)})" for tok in token_ids],
        labels={"x": f"Features in layer {layer}", "y": "Logits"},
        title=f'Gradients between SAE latents in layer {layer} and final logits (only showing top {k} logits)<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1900,
        height=800,
        aspect="auto",
    ).show()

# %%

if MAIN:
    gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    hf_repo_id = "callummcdougall/arena-demos-transcoder"
    sae_id = "gpt2-small-layer-{layer}-mlp-transcoder-folded-b_dec_out"
    gpt2_transcoders = {
        layer: SAE.from_pretrained(
            release=hf_repo_id, sae_id=sae_id.format(layer=layer), device=str(device)
        )[0]
        for layer in tqdm(range(9))
    }

    layer = 8
    gpt2_transcoder = gpt2_transcoders[layer]
    print("Transcoder hooks (same as regular SAE hooks):", gpt2_transcoder.hook_dict.keys())

    # Load the sparsity values, and plot them
    log_sparsity_path = hf_hub_download(hf_repo_id, f"{sae_id.format(layer=layer)}/log_sparsity.pt")
    log_sparsity = t.load(log_sparsity_path, map_location="cpu", weights_only=True)
    px.histogram(
        to_numpy(log_sparsity), width=800, template="ggplot2", title="Transcoder latent sparsity"
    ).update_layout(showlegend=False).show()
    live_latents = np.arange(len(log_sparsity))[to_numpy(log_sparsity > -4)]

    # Get the activations store
    gpt2_act_store = ActivationsStore.from_sae(
        model=gpt2,
        sae=gpt2_transcoders[layer],
        streaming=True,
        store_batch_size_prompts=16,
        n_batches_in_buffer=32,
        device=str(device),
    )
    tokens = gpt2_act_store.get_batch_tokens()
    assert tokens.shape == (gpt2_act_store.store_batch_size_prompts, gpt2_act_store.context_size)

# %%

def run_with_cache_with_transcoder(
    model: HookedSAETransformer,
    transcoders: list[SAE],
    tokens: Tensor,
    use_error_term: bool = True,  # by default we don't intervene, just compute activations
) -> ActivationCache:
    """
    Runs an MLP transcoder(s) on a batch of tokens. This is quite hacky, and eventually will be
    supported in a much better way by SAELens!
    """
    assert all(
        transcoder.cfg.hook_name.endswith("ln2.hook_normalized") for transcoder in transcoders
    )
    input_hook_names = [transcoder.cfg.hook_name for transcoder in transcoders]
    output_hook_names = [
        transcoder.cfg.hook_name.replace("ln2.hook_normalized", "hook_mlp_out")
        for transcoder in transcoders
    ]

    # Hook function at transcoder input: computes its output (and all intermediate values e.g.
    # latent activations)
    def hook_transcoder_input(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        _, cache = transcoders[transcoder_idx].run_with_cache(activations)
        hook.ctx["cache"] = cache

    # Hook function at transcoder output: replaces activations with transcoder output
    def hook_transcoder_output(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        cache: ActivationCache = model.hook_dict[transcoders[transcoder_idx].cfg.hook_name].ctx[
            "cache"
        ]
        return cache["hook_sae_output"]

    # Get a list of all fwd hooks (only including the output hooks if use_error_term=False)
    fwd_hooks = []
    for i in range(len(transcoders)):
        fwd_hooks.append((input_hook_names[i], partial(hook_transcoder_input, transcoder_idx=i)))
        if not use_error_term:
            fwd_hooks.append(
                (output_hook_names[i], partial(hook_transcoder_output, transcoder_idx=i))
            )

    # Fwd pass on model, triggering all hook functions
    with model.hooks(fwd_hooks=fwd_hooks):
        _, model_cache = model.run_with_cache(tokens)

    # Return union of both caches (we rename the transcoder hooks using the same convention as
    # regular SAE hooks)
    all_transcoders_cache_dict = {}
    for i, transcoder in enumerate(transcoders):
        transcoder_cache = model.hook_dict[input_hook_names[i]].ctx.pop("cache")
        transcoder_cache_dict = {
            f"{transcoder.cfg.hook_name}.{k}": v for k, v in transcoder_cache.items()
        }
        all_transcoders_cache_dict.update(transcoder_cache_dict)

    return ActivationCache(
        cache_dict=model_cache.cache_dict | all_transcoders_cache_dict, model=model
    )


def get_k_largest_indices(
    x: Float[Tensor, "batch seq"], k: int, buffer: int = 0, no_overlap: bool = True
) -> Int[Tensor, "k 2"]:
    if buffer > 0:
        x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = t.empty((0, 2), device=x.device).long()
        while len(unique_indices) < k:
            unique_indices = t.cat(
                (unique_indices, t.tensor([[rows[0], cols[0]]], device=x.device))
            )
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices

    return t.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + t.arange(
            -buffer, buffer + 1, device=x.device
        )
    return x[rows, cols]


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, seq_pos in data:
        formatted_seq = (
            "".join(
                [
                    f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok
                    for i, str_tok in enumerate(str_toks)
                ]
            )
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


def fetch_max_activating_examples(
    model: HookedSAETransformer,
    transcoder: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
    display: bool = False,
) -> list[tuple[float, list[str], int]]:
    data = []

    for _ in tqdm(range(total_batches)):
        tokens = act_store.get_batch_tokens()
        cache = run_with_cache_with_transcoder(model, [transcoder], tokens)
        acts = cache[f"{transcoder.cfg.hook_name}.hook_sae_acts_post"][..., latent_idx]

        k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)
        tokens_with_buffer = index_with_buffer(tokens, k_largest_indices, buffer=buffer)
        str_toks = [model.to_str_tokens(toks) for toks in tokens_with_buffer]
        top_acts = index_with_buffer(acts, k_largest_indices).tolist()
        data.extend(list(zip(top_acts, str_toks, [buffer] * len(str_toks))))

    data = sorted(data, key=lambda x: x[0], reverse=True)[:k]
    if display:
        display_top_seqs(data)
    return data

# %%

if MAIN:
    latent_idx = 1
    neuronpedia_id = "gpt2-small/8-tres-dc"
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    display(IFrame(url, width=800, height=600))

    fetch_max_activating_examples(
        gpt2, gpt2_transcoder, gpt2_act_store, latent_idx=latent_idx, total_batches=200, display=True
    )

# %%

def show_top_logits(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    k: int = 10,
) -> None:
    """Displays the top & bottom logits for a particular latent."""
    logits = sae.W_dec[latent_idx] @ model.W_U

    pos_logits, pos_token_ids = logits.topk(k)
    pos_tokens = model.to_str_tokens(pos_token_ids)
    neg_logits, neg_token_ids = logits.topk(k, largest=False)
    neg_tokens = model.to_str_tokens(neg_token_ids)

    print(
        tabulate(
            zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )


print(f"Top logits for transcoder latent {latent_idx}:")
show_top_logits(gpt2, gpt2_transcoder, latent_idx=latent_idx)


def show_top_deembeddings(
    model: HookedSAETransformer, sae: SAE, latent_idx: int, k: int = 10
) -> None:
    """Displays the top & bottom de-embeddings for a particular latent."""
    deembeddings = model.W_E @ sae.W_enc[:, latent_idx]
    top_acts, top_token_ids = deembeddings.topk(k)
    bottom_acts, bottom_token_ids = deembeddings.topk(k, largest=False)
    print(
        tabulate(
            zip(
                map(repr, model.to_str_tokens(bottom_token_ids)),
                bottom_acts,
                map(repr, model.to_str_tokens(top_token_ids)),
                top_acts,
            ),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )


print(f"\nTop de-embeddings for transcoder latent {latent_idx}:")
show_top_deembeddings(gpt2, gpt2_transcoder, latent_idx=latent_idx)
tests.test_show_top_deembeddings(show_top_deembeddings, gpt2, gpt2_transcoder)

# %%

def create_extended_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:
    """
    Creates the extended embedding matrix using the model's layer-0 MLP, and the method described
    in the exercise above.

    You should also divide the output by its standard deviation across the `d_model` dimension
    (this is because that's how it'll be used later e.g. when fed into the MLP layer / transcoder).
    """
    W_E = model.W_E.clone()
    normed_embeds = model.blocks[0].ln2.forward(W_E)
    mlp_out = model.blocks[0].mlp.forward(normed_embeds)
    result_unnorm = W_E + mlp_out
    return (result_unnorm - result_unnorm.mean(dim=-1).unsqueeze(-1)) / (result_unnorm.std(dim=-1).unsqueeze(-1) + 1e-8)


if MAIN:
    tests.test_create_extended_embedding(create_extended_embedding, gpt2)


def show_top_deembeddings_extended(
    model: HookedSAETransformer, sae: SAE, latent_idx: int, k: int = 10
) -> None:
    """Displays the top & bottom de-embeddings for a particular latent."""
    embeddings = create_extended_embedding(model)
    deembeddings = embeddings @ sae.W_enc[:, latent_idx]
    top_acts, top_token_ids = deembeddings.topk(k)
    bottom_acts, bottom_token_ids = deembeddings.topk(k, largest=False)
    print(
        tabulate(
            zip(
                map(repr, model.to_str_tokens(bottom_token_ids)),
                bottom_acts,
                map(repr, model.to_str_tokens(top_token_ids)),
                top_acts,
            ),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )


if MAIN:
    print(f"Top de-embeddings (extended) for transcoder latent {latent_idx}:")
    show_top_deembeddings_extended(gpt2, gpt2_transcoder, latent_idx=latent_idx)

# %%

print(t.cuda.memory_allocated() / 1e9)

# %%

gc.collect()
t.cuda.empty_cache()


# %%
