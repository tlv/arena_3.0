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
                      LanguageModelSAERunnerConfig, SAETrainingRunner)
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
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
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

# We start by emptying memory of all large tensors & objects (since we'll be loading in a lot of different models in the coming sections)
if MAIN:
    THRESHOLD = 0.1  # GB
    for obj in gc.get_objects():
        try:
            if isinstance(obj, t.nn.Module) and utils.get_tensors_size(obj) / 1024**3 > THRESHOLD:
                if hasattr(obj, "cuda"):
                    obj.cpu()
                if hasattr(obj, "reset"):
                    obj.reset()
        except:
            pass

# %%

if MAIN:
    tinystories_model = HookedSAETransformer.from_pretrained("tiny-stories-1L-21M")

    completions = [
        (i, tinystories_model.generate("Once upon a time", temperature=1, max_new_tokens=50))
        for i in range(5)
    ]

    print(tabulate(completions, tablefmt="simple_grid", maxcolwidths=[None, 100]))# %%

# %%

if MAIN:
    test_prompt(
        "Once upon a time, there was a little girl named Lily. She lived in a big, happy little girl. On her big adventure,",
        [" Lily", " she", " he"],
        tinystories_model,
    )

# %%

if MAIN:
    completion = tinystories_model.generate(
        "Once upon a time", temperature=2.5, verbose=False, max_new_tokens=200
    )

    cv.logits.token_log_probs(
        tinystories_model.to_tokens(completion),
        tinystories_model(completion).squeeze(0).log_softmax(dim=-1),
        tinystories_model.to_string,
    )

# %%

if MAIN:
    total_training_steps = 30_000  # probably we should do more
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = l1_warm_up_steps = total_training_steps // 10  # 10% of training
    lr_decay_steps = total_training_steps // 5  # 20% of training

    cfg = LanguageModelSAERunnerConfig(
        #
        # Data generation
        model_name="tiny-stories-1L-21M",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=tinystories_model.cfg.d_model,
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # tokenized language dataset on HF for the Tiny Stories corpus.
        is_dataset_tokenized=True,
        prepend_bos=True,  # you should use whatever the base model was trained with
        streaming=True,  # we could pre-download the token dataset if it was small.
        train_batch_size_tokens=batch_size,
        context_size=512,  # larger is better but takes longer (for tutorial we'll use a short one)
        #
        # SAE architecture
        architecture="gated",
        expansion_factor=16,
        b_dec_init_method="zeros",
        apply_b_dec_to_input=True,
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        #
        # Activations store
        n_batches_in_buffer=64,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        #
        # Training hyperparameters (standard)
        lr=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # controls how the LR warmup / decay works
        lr_warm_up_steps=lr_warm_up_steps,  # avoids large number of initial dead features
        lr_decay_steps=lr_decay_steps,  # helps avoid overfitting
        #
        # Training hyperparameters (SAE-specific)
        l1_coefficient=4,
        l1_warm_up_steps=l1_warm_up_steps,
        use_ghost_grads=False,  # we don't use ghost grads anymore
        feature_sampling_window=2000,  # how often we resample dead features
        dead_feature_window=1000,  # size of window to assess whether a feature is dead
        dead_feature_threshold=1e-4,  # threshold for classifying feature as dead, over window
        #
        # Logging / evals
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="arena-demos-tinystories",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        #
        # Misc.
        device=str(device),
        seed=42,
        n_checkpoints=5,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    # print("Comment this code out to train! Otherwise, it will load in the already trained model.")
    t.set_grad_enabled(True)
    runner = SAETrainingRunner(cfg)
    sae = runner.run()

    # hf_repo_id = "callummcdougall/arena-demos-tinystories"
    # sae_id = cfg.hook_name

    # upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

    # tinystories_sae = SAE.from_pretrained(release=hf_repo_id, sae_id=sae_id, device=str(device))[0]

# %%
