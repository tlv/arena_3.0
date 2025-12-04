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
from regex import W
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


def head_mean_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    mean_score = z[:, :, head_index_to_ablate, :].mean(dim=(0, 1))
    z[:, :, head_index_to_ablate, :] = mean_score
    return z


def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
    """
    Return the fraction of the time that the maximum value is on the circuit diagonal.
    """
    batch_num = 0
    tot = 0
    acc = 0
    while batch_num * (batch_size + 1) < full_OV_circuit.ldim:
        start = batch_num * batch_size
        batch = full_OV_circuit[start : min(start + batch_size, full_OV_circuit.ldim), :]
        num_samples = batch.ldim
        max_idxs = t.argmax(batch.AB, dim=1)
        tot += num_samples
        acc += (max_idxs == t.arange(start, start + num_samples).to(device)).sum()
        batch_num += 1
    return acc/tot


def decompose_qk_input(cache: ActivationCache) -> Float[Tensor, "n_heads+2 posn d_model"]:
    """
    Retrieves all the input tensors to the first attention layer, and concatenates them along the
    0th dim.

    The [i, :, :]th element is y_i (from notation above). The sum of these tensors along the 0th
    dim should be the input to the first attention layer.
    """

    e = cache["embed"]
    pe = cache["pos_embed"]
    att_heads = []
    for head in range(12):
        att_heads.append(
            cache["result", 0][:, head, :]
        )
    all_tensors = [e, pe] + att_heads
    return einops.rearrange(all_tensors, 'i t d -> i t d')


def decompose_q(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of query vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values).
    """
    return decomposed_qk_input @ model.W_Q[1][ind_head_index]


def decompose_k(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of key vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    """
    return decomposed_qk_input @ model.W_K[1][ind_head_index]


def decompose_attn_scores(
    decomposed_q: Float[Tensor, "q_comp q_pos d_head"],
    decomposed_k: Float[Tensor, "k_comp k_pos d_head"],
    model: HookedTransformer,
) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the
    attention scores)
    """
    return einops.einsum(
        decomposed_q, 
        decomposed_k,
        "hq tq d, hk tk d -> hq hk tq tk",
    ) / (decomposed_q.size(-1) ** 0.5)


def find_K_comp_full_circuit(
    model: HookedTransformer, prev_token_head_index: int, ind_head_index: int
) -> FactoredMatrix:
    """
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side
    (direct from token embeddings) and the second dimension being the key side (going via the
    previous token head).
    """
    W_E = model.W_E  # dv dm
    W_V = model.W_V[0][prev_token_head_index]  # dm dh
    W_O = model.W_O[0][prev_token_head_index]  # dh dm
    W_Q = model.W_Q[1][ind_head_index]  # dm dh
    W_K = model.W_K[1][ind_head_index]  # dm dh

    W_OVT = FactoredMatrix(W_O.transpose(0, 1), W_V.transpose(0, 1))
    W_QK = FactoredMatrix(W_Q, W_K.transpose(0, 1))

    return W_E @ W_QK @ (W_OVT) @ (W_E.transpose(0, 1))


def get_comp_score(W_A: Float[Tensor, "in_A out_A"], W_B: Float[Tensor, "out_A out_B"]) -> float:
    """
    Return the composition score between W_A and W_B.
    """
    result = t.linalg.matrix_norm(W_A @ W_B) / (t.linalg.matrix_norm(W_A) * t.linalg.matrix_norm(W_B))
    return result.item()


def generate_single_random_comp_score() -> float:
    """
    Write a function which generates a single composition score for random matrices
    """
    W = t.empty((4, 768, 64))
    nn.init.kaiming_uniform_(W, a=np.sqrt(5))
    return get_comp_score(
        W[0] @ W[1].transpose(0, 1),
        W[2] @ W[3].transpose(0, 1),
    )


# %%

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

    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

    # %%

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

    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

    # ---------------------------------------------------------------------------
    # PART 3
    # ---------------------------------------------------------------------------

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

    gpt2_small.run_with_hooks(
        rep_tokens_10,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(pattern_hook_names_filter, visualize_pattern_hook)],
    )

    # %%

    gpt2_small_induction_scores_store =  t.zeros(
        (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
    )

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

    # ---------------------------------------------------------------------------
    # PART 4
    # ---------------------------------------------------------------------------

    # %%

    A = t.randn(5, 2)
    B = t.randn(2, 5)
    AB = A @ B
    AB_factor = FactoredMatrix(A, B)
    print("Norms:")
    print(AB.norm())
    print(AB_factor.norm())

    print(f"Right dim: {AB_factor.rdim}, Left dim: {AB_factor.ldim}, Hidden dim: {AB_factor.mdim}")

    print("Eigenvalues:")
    print(t.linalg.eig(AB).eigenvalues)
    print(AB_factor.eigenvalues)

    print("\nSingular Values:")
    print(t.linalg.svd(AB).S)
    print(AB_factor.S)

    print("\nFull SVD:")
    print(AB_factor.svd())

    C = t.randn(5, 300)
    ABC = AB @ C
    ABC_factor = AB_factor @ C

    print(f"Unfactored: shape={ABC.shape}, norm={ABC.norm()}")
    print(f"Factored: shape={ABC_factor.shape}, norm={ABC_factor.norm()}")
    print(f"\nRight dim: {ABC_factor.rdim}, Left dim: {ABC_factor.ldim}, Hidden dim: {ABC_factor.mdim}")

    AB_unfactored = AB_factor.AB
    t.testing.assert_close(AB_unfactored, AB)

    # %%

    head_index = 4
    layer = 1

    # W_E: d_vocab x d_model
    # W_U: d_model x d_vocab
    # W_V: layers x heads x d_model x d_head
    # W_O: layers x heads x d_head x d_model

    # full circuit is W_E W_OV W_U
    # W_OV is W_V W_O

    full_OV_circuit = FactoredMatrix(
        model.W_E @ model.W_V[layer][head_index],
        model.W_O[layer][head_index] @ model.W_U,
    )

    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

    # %%

    indices = t.randint(0, model.cfg.d_vocab, (200,))
    full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

    imshow(
        full_OV_circuit_sample,
        labels={"x": "Logits on output token", "y": "Input token"},
        title="Full OV circuit for copying head",
        width=700,
        height=600,
    )

    print(f"Fraction of time that the best logit is on diagonal: {top_1_acc(full_OV_circuit):.4f}")

    combined_OV_circuit = (
        model.W_E @ FactoredMatrix(
            einops.rearrange([model.W_V[1][4], model.W_V[1][10]], 'h dm dh -> dm (h dh)'),
            einops.rearrange([model.W_O[1][4], model.W_O[1][10]], 'h dh dm -> (h dh) dm')
        ) @ model.W_U
    )

    print(f"Fraction of time that the best logit is on diagonal: {top_1_acc(combined_OV_circuit):.4f}")

    # %%

    layer = 0
    head_index = 7

    # Compute full QK matrix (for positional embeddings)
    W_pos = model.W_pos
    W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
    pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

    # Mask, scale and softmax the scores
    mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
    pos_by_pos_pattern = t.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

    # Plot the results
    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
    imshow(
        utils.to_numpy(pos_by_pos_pattern[:200, :200]),
        labels={"x": "Key", "y": "Query"},
        title="Attention patterns for prev-token QK circuit, first 100 indices",
        width=700,
        height=600,
    )

    # %%

    # Recompute rep tokens/logits/cache, if we haven't already
    seq_len = 50
    batch_size = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(
        model, seq_len, batch_size
    )
    rep_cache.remove_batch_dim()

    ind_head_index = 4

    # First we get decomposed q and k input, and check they're what we expect
    decomposed_qk_input = decompose_qk_input(rep_cache)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)
    t.testing.assert_close(
        decomposed_qk_input.sum(0),
        rep_cache["resid_pre", 1] + rep_cache["pos_embed"],
        rtol=0.01,
        atol=1e-05,
    )
    t.testing.assert_close(
        decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001
    )
    t.testing.assert_close(
        decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01
    )

    # Second, we plot our results
    component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
        imshow(
            utils.to_numpy(decomposed_input.pow(2).sum([-1])),
            labels={"x": "Position", "y": "Component"},
            title=f"Norms of components of {name}",
            y=component_labels,
            width=800,
            height=400,
        )

    # %%

    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)

    # %%

    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7), you can replace this
    # with any other pair and see that the values are generally much smaller, i.e. this pair dominates the attention score
    # calculation
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k, model)

    q_label = "Embed"
    k_label = "0.7"
    decomposed_scores_from_pair = decomposed_scores[
        component_labels.index(q_label), component_labels.index(k_label)
    ]

    imshow(
        utils.to_numpy(t.tril(decomposed_scores_from_pair)),
        title=f"Attention score contributions from query = {q_label}, key = {k_label}<br>(by query & key sequence positions)",
        width=700,
    )


    # Second plot: std dev over query and key positions, shown by component. This shows us that the other pairs of
    # (query_component, key_component) are much less important, without us having to look at each one individually like we
    # did in the first plot!
    decomposed_stds = einops.reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )
    imshow(
        utils.to_numpy(decomposed_stds),
        labels={"x": "Key Component", "y": "Query Component"},
        title="Std dev of attn score contributions across sequence positions<br>(by query & key comp)",
        x=component_labels,
        y=component_labels,
        width=700,
    )

    # %%

    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

    print(f"Token frac where max-activating key = same token: {top_1_acc(K_comp_circuit.T):.4f}")

    # %%

    tests.test_get_comp_score(get_comp_score)

    # %%

    W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
    W_OV = model.W_V @ model.W_O

    # Define tensors to hold the composition scores
    composition_scores = {
        "Q": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "K": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "V": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
    }

    # YOUR CODE HERE - fill in values of the `composition_scores` dict, using `get_comp_score`
    for head_0 in range(12):
        for head_1 in range(12):
            composition_scores["Q"][head_0][head_1] = get_comp_score(
                W_OV[0][head_0],
                W_QK[1][head_1],
            )
            composition_scores["K"][head_0][head_1] = get_comp_score(
                W_OV[0][head_0],
                W_QK[1][head_1].transpose(0, 1),
            )
            composition_scores["V"][head_0][head_1] = get_comp_score(
                W_OV[0][head_0],
                W_OV[1][head_1],
            )


    # Plot the composition scores
    for comp_type in ["Q", "K", "V"]:
        plot_comp_scores(model, composition_scores[comp_type], f"{comp_type} Composition Scores")

    # %%

    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()

    print("\nMean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())

    hist(
        comp_scores_baseline,
        nbins=50,
        width=800,
        labels={"x": "Composition score"},
        title="Random composition scores",
    )

    # %%

    baseline = comp_scores_baseline.mean()
    for comp_type, comp_scores in composition_scores.items():
        plot_comp_scores(model, comp_scores, f"{comp_type} Composition Scores", baseline=baseline)

