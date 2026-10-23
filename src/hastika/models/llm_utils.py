"""Shared quantized model loading and length-bucketed minibatches."""
import numpy as np
import torch

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def text_backbone(m):
    """The decoder stack of a text-only or multimodal checkpoint."""
    if hasattr(m, "layers"):
        return m
    for path in ("language_model", "model.language_model", "model", "text_model"):
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if hasattr(obj, "layers") or hasattr(obj, "embed_tokens"):
            return obj
    raise ValueError(f"no decoder stack found in {type(m).__name__}")

def load_backbone(name, token=None, cache_dir=None, compute_dtype=torch.float16, revision=None):
    import transformers
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    kw = dict(quantization_config=q, dtype=compute_dtype, device_map={"": 0}, token=token,
              cache_dir=cache_dir, revision=revision)
    err = None
    for cls in ("AutoModel", "AutoModelForCausalLM", "AutoModelForMultimodalLM"):
        auto = getattr(transformers, cls, None)
        if auto is None:
            continue
        try:
            return text_backbone(auto.from_pretrained(name, **kw))
        except (ValueError, KeyError, TypeError) as e:
            err = e
    raise err

def bucketed_order(lens, bs, rng, pool=50):
    """Shuffled, but batches hold similar lengths: shuffle, sort within pools of
    bs * pool rows, cut into batches, shuffle the batches. Padding is wasted compute,
    and on a T4 it was most of it."""
    order = rng.permutation(len(lens))
    out = []
    for i in range(0, len(order), bs * pool):
        chunk = sorted(order[i:i + bs * pool], key=lambda j: lens[j])
        out += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    return np.concatenate([out[k] for k in rng.permutation(len(out))])

def batches(ids, bs, pad_id, order):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(ids[j]) for j in idx)
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, :len(ids[j])] = torch.tensor(ids[j])
            m[r, :len(ids[j])] = 1
        yield idx, x, m

def quant_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True)

def load_lm(name, cache_dir, token=None, revision=None):
    """4-bit model with an LM head. Text-only and multimodal (Gemma-4) checkpoints
    register under different auto classes, so try them in turn."""
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(name, token=token, cache_dir=cache_dir, revision=revision)
    err = None
    for cls in ("AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        auto = getattr(transformers, cls, None)
        if auto is None:
            continue
        try:
            m = auto.from_pretrained(name, quantization_config=quant_config(), dtype=torch.float16,
                                     device_map={"": 0}, token=token, cache_dir=cache_dir, revision=revision)
            return tok, m.eval()
        except (ValueError, KeyError, TypeError) as e:
            err = e
    raise err
