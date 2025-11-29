# %%

import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (ActivationCache, FactoredMatrix,
                              HookedTransformer, HookedTransformerConfig,
                              utils)
from transformer_lens.hook_points import HookPoint

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
print(Path.cwd())
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (hist, imshow, plot_comp_scores,
                          plot_logit_attribution, plot_loss_difference)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

# %%
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

# %%
MAIN = __name__ == "__main__"


# %%
if MAIN:
    # ---------------------------------------------------------------------------
    # PART 1
    # ---------------------------------------------------------------------------

    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
    print(gpt2_small.cfg)

    # %%

    model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)

    # %%

    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]

    tokens = gpt2_small.to_tokens(model_description_text)
    print(sum(prediction == tokens[0, 1:]))

    # %%

    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

    print(type(gpt2_logits), type(gpt2_cache))

    attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
    attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

    t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)

    # %%
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    layer0_q = gpt2_cache["q", 0]
    layer0_k = gpt2_cache["k", 0]
    n_tokens = layer0_q.size(0)
    scaled_scores = einops.einsum(layer0_q, layer0_k, "tq h d, tk h d -> h tq tk") / (layer0_q.size(-1) ** 0.5)
    mask = (t.tril(t.ones((n_tokens, n_tokens))).unsqueeze(0) == 0).to(device)
    scaled_scores = scaled_scores.masked_fill(mask, -float("inf"))
    scores = t.softmax(scaled_scores, dim=-1)

    layer0_pattern_from_q_and_k = scores

    # YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the
    # steps of the attention calculation (dot product, masking, scaling, softmax)
    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")

    # %%
    attention_pattern = gpt2_cache["pattern", 0]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_str_tokens,
            attention=attention_pattern,
            attention_head_names=[f"L0H{i}" for i in range(12)],
        )
    )

    # ---------------------------------------------------------------------------
    # PART 2
    # ---------------------------------------------------------------------------

    # %%
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True,  # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer",
    )

    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained_weights)

    # %%
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    attention_pattern_0 = cache["pattern", 0]
    attention_pattern_1 = cache["pattern", 1]
    tokens = model.to_str_tokens(text)

    print("Layer 0 Head Attention Patterns:")
    display(
        cv.attention.attention_patterns(
            tokens=tokens,
            attention=attention_pattern_0,
            attention_head_names=[f"L0H{i}" for i in range(12)],
        )
    )
    print("Layer 1 Head Attention Patterns:")
    display(
        cv.attention.attention_patterns(
            tokens=tokens,
            attention=attention_pattern_1,
            attention_head_names=[f"L1H{i}" for i in range(12)],
        )
    )

    # %%
    print(attention_pattern_0.size())

    # %%
    def current_attn_detector(cache: ActivationCache) -> list[str]:
        """
        Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
        """
        def _is_current_attn(scores):
            return scores.trace() >= 0.5 * scores.sum()

        result = []

        for layer in [0, 1]:
            pattern = cache["pattern", layer]
            n_tokens = pattern.size(1)
            mask = t.triu(t.tril(t.ones((n_tokens, n_tokens)))).to(device)
            pattern_diags = pattern * mask
            traces = t.sum(pattern_diags, dim=(1, 2))
            scores_sum = t.sum(pattern, dim=(1, 2))
            print(traces / scores_sum)
            result += [
                f"{layer}.{head}"
                for head in range(traces.size(0))
                if traces[head] >= scores_sum[head] * 0.2
            ]

        return result

    def prev_attn_detector(cache: ActivationCache) -> list[str]:
        """
        Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
        """
        result = []

        for layer in [0, 1]:
            pattern = cache["pattern", layer]
            n_tokens = pattern.size(1)
            mask = t.triu(
                t.tril(
                    t.ones((n_tokens, n_tokens)),
                    diagonal=1,
                ),
                diagonal=-1,
            ).to(device)
            pattern_diags = pattern * mask
            offset_traces = t.sum(pattern_diags, dim=(1, 2))
            scores_sum = t.sum(pattern, dim=(1, 2))
            print(offset_traces / scores_sum)
            result += [
                f"{layer}.{head}"
                for head in range(offset_traces.size(0))
                if offset_traces[head] >= scores_sum[head] * 0.5
            ]

        return result


    def first_attn_detector(cache: ActivationCache) -> list[str]:
        """
        Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
        """
        result = []

        for layer in [0, 1]:
            pattern = cache["pattern", layer]
            first_scores = t.sum(pattern[:, :, 0], dim=1)
            scores_sum = t.sum(pattern, dim=(1, 2))
            print(first_scores / scores_sum)
            result += [
                f"{layer}.{head}"
                for head in range(first_scores.size(0))
                if first_scores[head] >= scores_sum[head] * 0.5
            ]

        return result

    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

    # %%
    def generate_repeated_tokens(
        model: HookedTransformer, seq_len: int, batch_size: int = 1
    ) -> Int[Tensor, "batch_size full_seq_len"]:
        """
        Generates a sequence of repeated random tokens

        Outputs are:
            rep_tokens: [batch_size, 1+2*seq_len]
        """
        t.manual_seed(0)  # for reproducibility
        prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
        seq = (model.cfg.d_vocab * t.rand(batch_size, seq_len)).int()
        return t.cat((prefix, seq, seq), dim=1).to(device)


    def run_and_cache_model_repeated_tokens(
        model: HookedTransformer, seq_len: int, batch_size: int = 1
    ) -> tuple[Tensor, Tensor, ActivationCache]:
        """
        Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
        logits, cache). This function should use the `generate_repeated_tokens` function above.

        Outputs are:
            rep_tokens: [batch_size, 1+2*seq_len]
            rep_logits: [batch_size, 1+2*seq_len, d_vocab]
            rep_cache: The cache of the model run on rep_tokens
        """
        tokens = generate_repeated_tokens(model, seq_len, batch_size)
        logits, cache = model.run_with_cache(tokens)
        return tokens, logits, cache


    def get_log_probs(
        logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
    ) -> Float[Tensor, "batch posn-1"]:
        logprobs = logits.log_softmax(dim=-1)
        # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
        correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
        return correct_logprobs


    seq_len = 50
    batch_size = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(
        model, seq_len, batch_size
    )
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)

    # %%

    display(
        cv.attention.attention_patterns(
            tokens=rep_str,
            attention=rep_cache["pattern", 0],
            attention_head_names=[f"L0H{i}" for i in range(12)],
        )
    )

    # %%

    display(
        cv.attention.attention_patterns(
            tokens=rep_str,
            attention=rep_cache["pattern", 1],
            attention_head_names=[f"L1H{i}" for i in range(12)],
        )
    )

    # %%

    def induction_attn_detector(cache: ActivationCache) -> list[str]:
        """
        Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

        Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
        """
        pattern = rep_cache["pattern", 1]
        seq_len = (pattern.size(1) - 1) // 2
        print(seq_len)
        result = []
        for head in range(pattern.size(0)):
            score = t.diagonal(pattern[head], offset=-seq_len + 1).mean()
            print(score)
            if score > 0.5:
                result.append(f"1.{head}")
        return result

    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%
