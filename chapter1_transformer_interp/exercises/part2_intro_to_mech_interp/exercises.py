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

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

MAIN = __name__ == "__main__"


if MAIN:
    # ---------------------------------------------------------------------------
    # PART 1
    # ---------------------------------------------------------------------------

    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
    print(gpt2_small.cfg)


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
        seq = t.randint(model.cfg.d_vocab, (batch_size, seq_len))
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
        result = []
        for head in range(pattern.size(0)):
            score = t.diagonal(pattern[head], offset=-seq_len + 1).mean()
            if score > 0.5:
                result.append(f"1.{head}")
        return result

    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

    # %%

    seq_len = 50
    batch_size = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU,
    # which can be slow.
    induction_score_store = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
    )
    print(induction_score_store.shape)


    def induction_score_hook(
        pattern: Float[Tensor, "batch head_index source_pos dest_pos"], hook: HookPoint
    ):
        """
        Calculates the induction score, and stores it in the [layer, head] position of the
        `induction_score_store` tensor.
        """
        seq_len = (pattern.size(2) - 1) // 2
        layer_induction_scores = einops.reduce(
            pattern.diagonal(offset=-(seq_len - 1), dim1=2, dim2=3),
            "b h t -> h",
            "mean",
        )
        induction_score_store[hook.layer(), :] = layer_induction_scores


    # We make a boolean filter on activation names, that's true only on attention pattern names
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=350,
    )

    # %%

    def visualize_pattern_hook(
        pattern: Float[Tensor, "batch head_index source_pos dest_pos"], hook: HookPoint
    ):
        seq_len = (pattern.size(2) - 1) // 2
        layer_induction_scores = einops.reduce(
            pattern.diagonal(offset=-(seq_len - 1), dim1=2, dim2=3),
            "b h t -> h",
            "mean",
        )
        if layer_induction_scores.max() >= 0.5:
            print("Layer: ", hook.layer())
            display(
                cv.attention.attention_patterns(
                    tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)
                )
            )


    gpt2_small.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, visualize_pattern_hook)],
    )

    # %%
    gpt2_small_induction_scores_store =  t.zeros(
        (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
    )

    def find_induction_heads_hook(
        pattern: Float[Tensor, "batch head_index source_pos dest_pos"], hook: HookPoint
    ):
        seq_len = (pattern.size(2) - 1) // 2
        layer_induction_scores = einops.reduce(
            pattern.diagonal(offset=-(seq_len - 1), dim1=2, dim2=3),
            "b h t -> h",
            "mean",
        )
        gpt2_small_induction_scores_store[hook.layer(), :] = layer_induction_scores
        if layer_induction_scores.max() >= 0.5:
            heads = []
            for i in range(len(layer_induction_scores)):
                if layer_induction_scores[i] >= 0.3:
                    heads.append(i)
            print(f"Found induction heads in layer {hook.layer()}: {', '.join(str(head) for head in heads)}")

    gpt2_small.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, find_induction_heads_hook)],
    )

    imshow(
        gpt2_small_induction_scores_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=900,
    )

    # %%
    def logit_attribution(
        embed: Float[Tensor, "seq d_model"],
        l1_results: Float[Tensor, "seq nheads d_model"],
        l2_results: Float[Tensor, "seq nheads d_model"],
        W_U: Float[Tensor, "d_model d_vocab"],
        tokens: Int[Tensor, "seq"],
    ) -> Float[Tensor, "seq-1 n_components"]:
        """
        Inputs:
            embed: the embeddings of the tokens (i.e. token + position embeddings)
            l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
            l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
            W_U: the unembedding matrix
            tokens: the token ids of the sequence

        Returns:
            Tensor of shape (seq_len-1, n_components)
            represents the concatenation (along dim=-1) of logit attributions from:
                the direct path (seq-1,1)
                layer 0 logits (seq-1, n_heads)
                layer 1 logits (seq-1, n_heads)
            so n_components = 1 + 2*n_heads
        """
        W_U_correct_tokens = W_U[:, tokens[1:]]  # d_model seq

        embed_attr = einops.rearrange(
            einops.einsum(embed[:-1, :], W_U_correct_tokens, 't d, d t -> t'),
            't -> t ()',
        )
        l1_attr = einops.einsum(l1_results[:-1, :], W_U_correct_tokens, 't nh d, d t -> t nh')
        l2_attr = einops.einsum(l2_results[:-1, :], W_U_correct_tokens, 't nh d, d t -> t nh')

        result = t.cat(
            [embed_attr, l1_attr, l2_attr],
            dim=1,
        )
        return result


    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
        print("Tests passed!")

    # %%

    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens.squeeze())

    plot_logit_attribution(model, logit_attr, tokens, title="Logit attribution (demo prompt)")

    plot_logit_attribution(
        model,
        logit_attribution(
            rep_cache["embed"],
            rep_cache["result", 0],
            rep_cache["result", 1],
            model.W_U,
            rep_tokens.squeeze(),
        ),
        rep_tokens,
        title="Logit attribution (demo prompt)",
    )

    # %%

    def head_zero_ablation_hook(
        z: Float[Tensor, "batch seq n_heads d_head"],
        hook: HookPoint,
        head_index_to_ablate: int,
    ) -> None:
        z[:, :, head_index_to_ablate, :] = t.zeros([z.shape[0], z.shape[1], z.shape[3]])
        return z

    def get_ablation_scores(
        model: HookedTransformer,
        tokens: Int[Tensor, "batch seq"],
        ablation_function: Callable = head_zero_ablation_hook,
    ) -> Float[Tensor, "n_layers n_heads"]:
        """
        Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
        from ablating the output of each head.
        """
        # Initialize an object to store the ablation scores
        ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

        # Calculating loss without any ablation, to act as a baseline
        model.reset_hooks()
        seq_len = (tokens.shape[1] - 1) // 2
        logits = model(tokens, return_type="logits")
        loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

        for layer in tqdm(range(model.cfg.n_layers)):
            for head in range(model.cfg.n_heads):
                logits = model.run_with_hooks(
                    tokens,
                    return_type="logits",                    
                    fwd_hooks=[(
                        utils.get_act_name("z", layer), 
                        functools.partial(ablation_function, head_index_to_ablate=head),
                    )],
                )
                ablated_loss = -get_log_probs(logits, tokens)[:, -(seq_len - 1):].mean()
                ablation_scores[layer][head] = ablated_loss - loss_no_ablation

        return ablation_scores

    ablation_scores = get_ablation_scores(model, rep_tokens)
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

    imshow(
        ablation_scores,
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Loss Difference After Ablating Heads",
        text_auto=".2f",
        width=900,
        height=350,
    )
    
    # %%
    def head_mean_ablation_hook(
        z: Float[Tensor, "batch seq n_heads d_head"],
        hook: HookPoint,
        head_index_to_ablate: int,
    ) -> None:
        mean_score = z[:, :, head_index_to_ablate, :].mean(dim=(0, 1))
        z[:, :, head_index_to_ablate, :] = mean_score
        return z


    rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)[0]
    mean_ablation_scores = get_ablation_scores(
        model, rep_tokens_batch, ablation_function=head_mean_ablation_hook
    )

    imshow(
        mean_ablation_scores,
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Loss Difference After Ablating Heads",
        text_auto=".2f",
        width=900,
        height=350,
    )


# %%
# %%
