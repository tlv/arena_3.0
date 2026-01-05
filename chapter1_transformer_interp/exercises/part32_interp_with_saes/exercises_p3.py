
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
from exercises_p1 import display_dashboard
from exercises_p2 import get_k_largest_indices
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
    sae_release = "gpt2-small-res-jb-feature-splitting"

    widths = [768 * (2**n) for n in range(7)]  # Note, you can increase to 8 if it fits on your GPU
    sae_ids = [f"blocks.8.hook_resid_pre_{width}" for width in widths]

    splitting_saes = {
        width: SAE.from_pretrained(sae_release, sae_id, device=str(device))[0]
        for width, sae_id in zip(widths, sae_ids)
    }

    gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

# %%

def load_and_process_autointerp_dfs(width: int):
    # Load in dataframe
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[f"blocks.8.hook_resid_pre_{width}"]
    url = "https://www.neuronpedia.org/api/explanation/export?modelId={}&saeId={}".format(
        *neuronpedia_id.split("/")
    )
    headers = {"Content-Type": "application/json"}
    data = requests.get(url, headers=headers).json()
    df = pd.DataFrame(data)

    # Drop duplicate latent descriptions
    df["index"] = df["index"].astype(int)
    df = df.drop_duplicates(subset=["index"], keep="first").sort_values("index", ignore_index=True)

    # Fill in missing latent descriptions with empty strings
    full_index = pd.DataFrame({"index": range(width)})
    df = full_index.merge(df, on="index", how="left")
    df["description"] = df["description"].fillna("")
    print(f"Loaded autointerp df for {width=}")
    if (n_missing := (df["description"] == "").sum()) > 0:
        print(f"  Warning: {n_missing}/{len(df)} latents missing descriptions")

    return df


if MAIN:
    autointerp_dfs = {width: load_and_process_autointerp_dfs(width) for width in widths}
    display(autointerp_dfs[768].head())

# %%

def find_top_related_latents(
    model,
    sae1,
    sae2,
    sae1_latent_idx,
    act_store,
    n_batches=100,
):
    sae1_acts_name = f"{sae1.cfg.hook_name}.hook_sae_acts_post"
    sae2_acts_name = f"{sae2.cfg.hook_name}.hook_sae_acts_post"
    sae1_acts_list = []  # list[batch seq dsae1]
    tokens_list = []  # list[batch seq]

    for _ in tqdm(range(n_batches)):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae1],
            names_filter = [
                sae1_acts_name,
            ],
            use_error_term=True,
        )
        sae1_acts_list.append(cache[sae1_acts_name])
        tokens_list.append(tokens)

    all_sae1_acts = t.cat(sae1_acts_list)[:, :, sae1_latent_idx]  # BATCH seq
    sae1_topk_indices = get_k_largest_indices(all_sae1_acts, k=25)  # k 2

    all_tokens = t.cat(tokens_list)  # BATCH seq
    topk_token_seqs = all_tokens[sae1_topk_indices[:, 0]]  # k seq

    _, cache2 = model.run_with_cache_with_saes(
        topk_token_seqs,
        saes=[sae2],
        names_filter = [sae2_acts_name],
        use_error_term=True,
    )
    sae2_acts = cache2[sae2_acts_name]  # k seq dsae2
    sae2_relevant_acts = sae2_acts[range(sae2_acts.shape[0]), sae1_topk_indices[:, 1], :]  # k dsae2
    top_latents = sae2_relevant_acts.sum(dim=0).topk(k=10)
    return top_latents.indices

# %%

if MAIN:
    gpt2_768_act_store = ActivationsStore.from_sae(
        model=gpt2,
        sae=splitting_saes[768],
        streaming=True,
        store_batch_size_prompts=16,
        n_batches_in_buffer=32,
        device=str(device),
    )

# %%

if MAIN:
    print(find_top_related_latents(
        gpt2,
        splitting_saes[768],
        splitting_saes[3072],
        5,
        gpt2_768_act_store,
    ))  # [ 731, 1500, 2336, 1439,  658,  903, 1058, 2886, 1045,  302]

# %%

if MAIN:
    display_dashboard(
        sae_release=sae_release,
        sae_id=f"blocks.8.hook_resid_pre_3072",
        latent_idx=658,
    )

# %%

if MAIN:
    gc.collect()
    t.cuda.empty_cache()

# %%
