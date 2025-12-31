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
RUN_INTRO_EXERCISES = False
RUN_RESID_SAE_EXERCISES = False
RUN_ATTN_SAE_EXERCISES = False

# %%
def format_value(value):
    return (
        "{{{0!r}: {1!r}, ...}}".format(*next(iter(value.items())))
        if isinstance(value, dict)
        else repr(value)
    )

if MAIN and RUN_INTRO_EXERCISES:
    print(get_pretrained_saes_directory())

    metadata_rows = [
        [data.model, data.release, data.repo_id, len(data.saes_map)]
        for data in get_pretrained_saes_directory().values()
    ]

    # Print all SAE releases, sorted by base model
    print(
        tabulate(
            sorted(metadata_rows, key=lambda x: x[0]),
            headers=["model", "release", "repo_id", "n_saes"],
            tablefmt="simple_outline",
        )
    )

    release = get_pretrained_saes_directory()["gpt2-small-res-jb"]

    print(
        tabulate(
            [[k, format_value(v)] for k, v in release.__dict__.items()],
            headers=["Field", "Value"],
            tablefmt="simple_outline",
        )
    )

    data = [[id, path, release.neuronpedia_id[id]] for id, path in release.saes_map.items()]

    print(
        tabulate(
            data,
            headers=["SAE id", "SAE path (HuggingFace)", "Neuronpedia ID"],
            tablefmt="simple_outline",
        )
    )

# %%

if MAIN:
    t.set_grad_enabled(False)

    gpt2: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    gpt2_sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.7.hook_resid_pre",
        device=str(device),
    )

    print(tabulate(gpt2_sae.cfg.__dict__.items(), headers=["name", "value"], tablefmt="simple_outline"))
#
#  %%
def display_dashboard(
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    latent_idx=0,
    width=800,
    height=600,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))

if MAIN and RUN_INTRO_EXERCISES:
    latent_idx = random.randint(0, gpt2_sae.cfg.d_sae)
    display_dashboard(latent_idx=latent_idx)

# %%
if MAIN and RUN_INTRO_EXERCISES:
    prompt = "Mitigating the risk of extinction from AI should be a global"
    answer = " priority"

    # First see how the model does without SAEs
    test_prompt(prompt, answer, gpt2)

    # Test our prompt, to see what the model says
    with gpt2.saes(saes=[gpt2_sae]):
        test_prompt(prompt, answer, gpt2)

    # Same thing, done in a different way
    gpt2.add_sae(gpt2_sae)
    test_prompt(prompt, answer, gpt2)
    gpt2.reset_saes()  # Remember to always do this!

    # Using `run_with_saes` method in place of standard forward pass
    logits = gpt2(prompt, return_type="logits")
    logits_sae = gpt2.run_with_saes(prompt, saes=[gpt2_sae], return_type="logits")
    answer_token_id = gpt2.to_single_token(answer)

    # Getting model's prediction
    top_prob, token_id_prediction = logits[0, -1].softmax(-1).max(-1)
    top_prob_sae, token_id_prediction_sae = logits_sae[0, -1].softmax(-1).max(-1)

    print(f"""Standard model:
        top prediction = {gpt2.to_string(token_id_prediction)!r}
        prob = {top_prob.item():.2%}
    SAE reconstruction:
        top prediction = {gpt2.to_string(token_id_prediction_sae)!r}
        prob = {top_prob_sae.item():.2%}
    """)
# %%
if MAIN and RUN_INTRO_EXERCISES:
    _, cache = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])

    for name, param in cache.items():
        if "hook_sae" in name:
            print(f"{name:<43}: {tuple(param.shape)}")
# %%
if MAIN and RUN_INTRO_EXERCISES:
    # Get top activations on final token
    _, cache = gpt2.run_with_cache_with_saes(
        prompt,
        saes=[gpt2_sae],
        stop_at_layer=gpt2_sae.cfg.hook_layer + 1,
    )
    sae_acts_post = cache[f"{gpt2_sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]

    # Plot line chart of latent activations
    px.line(
        sae_acts_post.cpu().numpy(),
        title=f"Latent activations at the final token position ({sae_acts_post.nonzero().numel()} alive)",
        labels={"index": "Latent", "value": "Activation"},
        width=1000,
    ).update_layout(showlegend=False).show()

    # Print the top 5 latents, and inspect their dashboards
    for act, ind in zip(*sae_acts_post.topk(3)):
        print(f"Latent {ind} had activation {act:.2f}")
        display_dashboard(latent_idx=ind)
# %%
if MAIN and RUN_INTRO_EXERCISES:
    logits_no_saes, cache_no_saes = gpt2.run_with_cache(prompt)

    gpt2_sae.use_error_term = False
    logits_with_sae_recon, cache_with_sae_recon = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])

    gpt2_sae.use_error_term = True
    logits_without_sae_recon, cache_without_sae_recon = gpt2.run_with_cache_with_saes(
        prompt, saes=[gpt2_sae]
    )

    # Both SAE caches contain the hook values
    assert f"{gpt2_sae.cfg.hook_name}.hook_sae_acts_post" in cache_with_sae_recon
    assert f"{gpt2_sae.cfg.hook_name}.hook_sae_acts_post" in cache_without_sae_recon

    # But final output will be different, because we don't use SAE reconstructions when use_error_term
    t.testing.assert_close(logits_no_saes, logits_without_sae_recon)
    logit_diff_from_sae = (logits_no_saes - logits_with_sae_recon).abs().mean()
    print(f"Average logit diff from using SAE reconstruction: {logit_diff_from_sae:.4f}")
# %%

if MAIN:
    print(gpt2_sae.cfg.dataset_path)

    gpt2_act_store = ActivationsStore.from_sae(
        model=gpt2,
        sae=gpt2_sae,
        streaming=True,
        store_batch_size_prompts=16,
        n_batches_in_buffer=32,
        device=str(device),
    )

    # Example of how you can use this:
    tokens = gpt2_act_store.get_batch_tokens()
    assert tokens.shape == (gpt2_act_store.store_batch_size_prompts, gpt2_act_store.context_size)

# %%
def show_activation_histogram(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 200,
):
    """
    Displays the activation histogram for a particular latent, computed across `total_batches`
    batches from `act_store`.
    """
    total_acts = 0
    nonzero_acts = []
    for i in tqdm(range(total_batches), "computing activations..."):
        tokens = act_store.get_batch_tokens()
        total_acts += tokens.flatten().size(0)
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[f"{sae.cfg.hook_name}.hook_sae_acts_post"],
        )
        sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][:, :, latent_idx]
        sae_acts_nonzero = sae_acts_post[sae_acts_post > 1e-8]
        nonzero_acts.append(sae_acts_nonzero.clone().cpu())
        del cache
        t.cuda.empty_cache()
    all_nonzero_acts = t.cat(nonzero_acts)

    frac_active = all_nonzero_acts.shape[0] / total_acts
    px.histogram(
        all_nonzero_acts,
        nbins=50,
        title=f"Activation density: {frac_active:.3%}",
        labels={"value": "activation"},
        width=800,
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
    ).update_layout(bargap=0.02, showlegend=False).show()


if MAIN and RUN_INTRO_EXERCISES:
    show_activation_histogram(gpt2, gpt2_sae, gpt2_act_store, latent_idx=9)

# %%
def get_k_largest_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int = 0,
    no_overlap: bool = True,
) -> Int[Tensor, "k 2"]:
    """
    Returns the tensor of (batch, seqpos) indices for each of the top k elements in the tensor x.

    Args:
        buffer:     We won't choose any elements within `buffer` from the start or end of their seq
                    (this helps if we want more context around the chosen tokens).
        no_overlap: If True, this ensures that no 2 top-activating tokens are in the same seq and
                    within `buffer` of each other.
    """
    assert buffer * 2 < x.size(1), "Buffer is too large for the sequence length"
    assert not no_overlap or k <= x.size(0), (
        "Not enough sequences to have a different token in each sequence"
    )

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


if MAIN and RUN_RESID_SAE_EXERCISES:
    x = t.arange(40, device=device).reshape((2, 20))
    x[0, 10] += 50  # 2nd highest value
    x[0, 16] += 100  # highest value
    x[1, 1] += 150  # not inside buffer (it's less than 3 from the start of the sequence)
    top_indices = get_k_largest_indices(x, k=2, buffer=3)
    assert top_indices.tolist() == [[0, 16], [0, 10]]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:
    """
    Indexes into `x` with `indices` (which should have come from the `get_k_largest_indices`
    function), and takes a +-buffer range around each indexed element. If `indices` are less than
    `buffer` away from the start of a sequence then we just take the first `2*buffer+1` elems (same
    for at the end of a sequence).

    If `buffer` is None, then we don't add any buffer and just return the elements at the given indices.
    """
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + t.arange(
            -buffer, buffer + 1, device=cols.device
        )
    return x[rows, cols]

if MAIN and RUN_RESID_SAE_EXERCISES:
    x_top_values_with_context = index_with_buffer(x, top_indices, buffer=3)
    assert x_top_values_with_context[0].tolist() == [
        13,
        14,
        15,
        16 + 100,
        17,
        18,
        19,
    ]  # highest value in the middle
    assert x_top_values_with_context[1].tolist() == [
        7,
        8,
        9,
        10 + 50,
        11,
        12,
        13,
    ]  # 2nd highest value in the middle


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    """
    Given a list of (activation: float, str_toks: list[str], seq_pos: int), displays a table of
    these sequences, with the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens � (usually weird quotation marks)
    for readability.
    """
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


if MAIN and RUN_RESID_SAE_EXERCISES:
    example_data = [
        (0.5, [" one", " two", " three"], 0),
        (1.5, [" one", " two", " three"], 1),
        (2.5, [" one", " two", " three"], 2),
    ]
    display_top_seqs(example_data)

# %%

def fetch_max_activating_examples(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
) -> list[tuple[float, list[str], int]]:
    """
    Returns the max activating examples across a number of batches from the activations store.
    """
    all_acts_list = []
    all_tokens_list = []
    for i in tqdm(range(total_batches), "computing activations..."):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[f"{sae.cfg.hook_name}.hook_sae_acts_post"],
        )
        sae_acts_post = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][:, :, latent_idx]
        all_acts_list.append(sae_acts_post.clone())
        all_tokens_list.append(tokens)
        del cache
        t.cuda.empty_cache()
    all_acts = t.cat(all_acts_list)
    all_tokens = t.cat(all_tokens_list)
    max_idx = get_k_largest_indices(all_acts, k, buffer=0)
    with_buffer = index_with_buffer(all_tokens, max_idx.clone(), buffer=buffer)
    data = []
    for i in range(k):
        tokens = all_tokens[max_idx[i][0]]
        seq = with_buffer[i]
        act = all_acts[tuple(max_idx[i])]
        if max_idx[i][1] < buffer:
            idx_in_seq = max_idx[i][1]
        elif max_idx[i][1] > len(tokens) - 1 - buffer:
            idx_in_seq = 2 * buffer + 1 - (len(tokens) - max_idx[i][1])
        else:
            idx_in_seq = buffer
        if idx_in_seq >= 0 and idx_in_seq < len(seq):
            data.append([act, model.to_str_tokens(seq), idx_in_seq])
        else:
            raise Exception()
    return data


if MAIN and RUN_RESID_SAE_EXERCISES:
    # Fetch & display the results
    buffer = 10
    data = fetch_max_activating_examples(
        gpt2, gpt2_sae, gpt2_act_store, latent_idx=9, buffer=buffer, k=5
    )
    display_top_seqs(data)

    # Test one of the results, to see if it matches the expected output
    first_seq_str_tokens = data[0][1]
    assert first_seq_str_tokens[buffer] == " new"

# %%

def show_top_logits(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    k: int = 10,
) -> None:
    """
    Displays the top & bottom logits for a particular latent.
    """
    logits = sae.W_dec[latent_idx] @ model.W_U
    largest = logits.topk(k=k)
    smallest = logits.topk(k=k, largest=False)
    table = Table("Bottom tokens", "Value", "Top tokens", "Value")
    for i in range(k):
        bot_token = model.to_single_str_token(int(smallest.indices[i]))
        bot_val = float(smallest.values[i])
        top_token = model.to_single_str_token(int(largest.indices[i]))
        top_val = float(largest.values[i])
        table.add_row(bot_token, f"{bot_val:.4f}", top_token, f"{top_val:.4f}")
    rprint(table)

show_top_logits(gpt2, gpt2_sae, latent_idx=9)
# tests.test_show_top_logits(show_top_logits, gpt2, gpt2_sae)

# %%

if MAIN and RUN_ATTN_SAE_EXERCISES:
    attn_saes = {
        layer: SAE.from_pretrained(
            "gpt2-small-hook-z-kk",
            f"blocks.{layer}.hook_z",
            device=str(device),
        )[0]
        for layer in [9]  # range(gpt2.cfg.n_layers)  NB - just load one to save memory
    }

    layer = 9

    display_dashboard(
        sae_release="gpt2-small-hook-z-kk",
        sae_id=f"blocks.{layer}.hook_z",
        latent_idx=2,  # or you can try `random.randint(0, attn_saes[layer].cfg.d_sae)`
    )
# %%

@dataclass
class AttnSeqDFA:
    act: float
    str_toks_dest: list[str]
    str_toks_src: list[str]
    dest_pos: int
    src_pos: int


def display_top_seqs_attn(data: list[AttnSeqDFA]):
    """
    Same as previous function, but we now have 2 str_tok lists and 2 sequence positions to
    highlight, the first being for top activations (destination token) and the second for top DFA
    (src token). We've given you a dataclass to help keep track of this.
    """
    table = Table(
        "Top Act",
        "Src token DFA (for top dest token)",
        "Dest token",
        title="Max Activating Examples",
        show_lines=True,
    )
    for seq in data:
        formatted_seqs = [
            repr(
                "".join(
                    [
                        f"[b u {color}]{str_tok}[/]" if i == seq_pos else str_tok
                        for i, str_tok in enumerate(str_toks)
                    ]
                )
                .replace("�", "")
                .replace("\n", "↵")
            )
            for str_toks, seq_pos, color in [
                (seq.str_toks_src, seq.src_pos, "dark_orange"),
                (seq.str_toks_dest, seq.dest_pos, "green"),
            ]
        ]
        table.add_row(f"{seq.act:.3f}", *formatted_seqs)
    rprint(table)

if MAIN and RUN_ATTN_SAE_EXERCISES:
    str_toks = [" one", " two", " three", " four"]
    example_data = [
    AttnSeqDFA(
        act=0.5, str_toks_dest=str_toks[1:], str_toks_src=str_toks[:-1], dest_pos=0, src_pos=0
    ),
    AttnSeqDFA(
        act=1.5, str_toks_dest=str_toks[1:], str_toks_src=str_toks[:-1], dest_pos=1, src_pos=1
    ),
    AttnSeqDFA(
        act=2.5, str_toks_dest=str_toks[1:], str_toks_src=str_toks[:-1], dest_pos=2, src_pos=0
    ),
    ]
    display_top_seqs_attn(example_data)

def fetch_max_activating_examples_attn(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 250,
    k: int = 10,
    buffer: int = 10,
) -> list[AttnSeqDFA]:
    """
    Returns the max activating examples across a number of batches from the activations store.
    """
    all_acts_list = []
    all_tokens_list = []
    all_attn_v_list = []
    all_attn_pattern_list = []
    sae_acts_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    attn_v_name = f"blocks.{sae.cfg.hook_layer}.attn.hook_v"
    attn_pattern_name = f"blocks.{sae.cfg.hook_layer}.attn.hook_pattern"

    for i in tqdm(range(total_batches), "computing activations..."):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[
                sae_acts_name,
                attn_v_name,
                attn_pattern_name,
            ],
        )
        sae_acts_post = cache[sae_acts_name][:, :, latent_idx]
        all_acts_list.append(sae_acts_post.clone())
        all_tokens_list.append(tokens.clone())
        all_attn_v_list.append(cache[attn_v_name].clone())
        all_attn_pattern_list.append(cache[attn_pattern_name].clone())
        del cache
        t.cuda.empty_cache()
    all_acts = t.cat(all_acts_list)
    all_tokens = t.cat(all_tokens_list)
    all_attn_v = t.cat(all_attn_v_list)
    all_attn_pattern = t.cat(all_attn_pattern_list)

    print(all_acts.shape)
    print(all_tokens.shape)
    print(all_attn_v.shape)
    print(all_attn_pattern.shape)

    max_dest_idx = get_k_largest_indices(all_acts, k, buffer=0)
    dest_with_buffer = index_with_buffer(all_tokens, max_dest_idx.clone(), buffer=buffer)

    vals = einops.rearrange(
        all_attn_v[max_dest_idx[:, 0]],
        "k ctx h dh -> k h ctx dh",
    )
    pattern = all_attn_pattern[max_dest_idx[:, 0], :, max_dest_idx[:, 1], :].unsqueeze(-1)  # k h ctx 1
    print(vals.shape)
    print(pattern.shape)
    vals_weighted = vals * pattern
    vals_flattened = einops.rearrange(vals_weighted, "k h ctx dh -> k (h dh) ctx")
    dfa = einops.einsum(
        vals_flattened,
        sae.W_dec[latent_idx],
        "k hdh ctx, hdh -> k ctx"
    )
    max_source_idx_ctx = dfa.argmax(dim=-1)
    max_source_idx = einops.rearrange(
        t.stack([max_dest_idx[:, 0], t.Tensor(max_source_idx_ctx).to(device)]),
        "idx k -> k idx",
    ).int()
    source_with_buffer = index_with_buffer(all_tokens, max_source_idx, buffer=buffer)

    data = []
    for i in range(k):
        dest_tokens = all_tokens[max_dest_idx[i][0]]
        dest_seq = dest_with_buffer[i]
        source_tokens = all_tokens[max_source_idx[i][0]]
        source_seq = source_with_buffer[i]

        act = all_acts[tuple(max_dest_idx[i])]

        if max_dest_idx[i][1] < buffer:
            dest_idx_in_seq = max_dest_idx[i][1]
        elif max_dest_idx[i][1] > len(dest_tokens) - 1 - buffer:
            dest_idx_in_seq = 2 * buffer + 1 - (len(dest_tokens) - max_dest_idx[i][1])
        else:
            dest_idx_in_seq = buffer

        if max_source_idx[i][1] < buffer:
            source_idx_in_seq = max_source_idx[i][1]
        elif max_source_idx[i][1] > len(source_tokens) - 1 - buffer:
            source_idx_in_seq = 2 * buffer + 1 - (len(source_tokens) - max_source_idx[i][1])
        else:
            source_idx_in_seq = buffer

        if dest_idx_in_seq >= 0 and dest_idx_in_seq < len(dest_seq) and source_idx_in_seq >= 0 and source_idx_in_seq < len(source_seq):
            data.append(AttnSeqDFA(
                act=act,
                str_toks_dest=model.to_str_tokens(dest_seq),
                str_toks_src=model.to_str_tokens(source_seq),
                dest_pos=dest_idx_in_seq,
                src_pos=source_idx_in_seq,
            ),)
        else:
            raise Exception()
    return data


if MAIN and RUN_ATTN_SAE_EXERCISES:
    # Test your function: compare it to dashboard above
    # (max DFA should come from sourcs tokens like " guns", " firearms")
    gc.collect()
    t.cuda.empty_cache()
    layer = 9
    sae = attn_saes[layer]
    data = fetch_max_activating_examples_attn(gpt2, attn_saes[layer], gpt2_act_store, latent_idx=2)
    display_top_seqs_attn(data)


# %%
