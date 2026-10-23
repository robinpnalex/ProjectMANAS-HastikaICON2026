"""LoRA next-token adaptation on training-side and permitted external text."""
import argparse
import math
import os

# Reduce fragmentation-type CUDA OOMs ("reserved but unallocated" memory on a 15 GB T4).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hastika.common.preprocessing import clean
from hastika.models.llm_utils import LORA_TARGETS, batches, bucketed_order, text_backbone
from hastika.models.llm_utils import load_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--texts", nargs="+", required=True, help="CSVs with a Comment column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    # The most memory-safe settings by default, at the same effective batch (16). Micro-batch
    # 4 ran out of memory on a T4 (Task A Run 31): TAPT holds the LM head and its gradients
    # on top of everything the classifier holds. 128 tokens covers > 99% of comments.
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--logit-chunk", type=int, default=16,
                    help="hidden-state positions projected through the LM head at once")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    # Accepted and ignored. A generic out-of-memory retry once appended the classifier's
    # --eval-bs here, and argparse killed the retry instantly (Task A Run 31).
    ap.add_argument("--eval-bs", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("TAPT requires a CUDA GPU and the llm extra")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    token = os.environ.get("HF_TOKEN") or None

    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False     # see llm_classifier.py
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    texts = []
    for path in args.texts:
        texts += [clean(t) for t in pd.read_csv(path)["Comment"].astype(str)]
    texts = list(dict.fromkeys(t for t in texts if len(t.split()) >= 2))   # dedupe, keep order
    print(f"{len(texts)} unique comments of 2+ words from {len(args.texts)} files", flush=True)

    t0 = time.time()
    tok, model = load_lm(args.model, args.cache, token, revision=args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm_head = model.get_output_embeddings()
    cfg = getattr(model.config, "text_config", model.config)
    softcap = getattr(cfg, "final_logit_softcapping", None)
    backbone = text_backbone(model)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, target_modules=LORA_TARGETS,
        bias="none"))
    backbone.print_trainable_parameters()
    for p in lm_head.parameters():
        p.requires_grad_(False)
    print(f"loaded in {(time.time() - t0) / 60:.1f} min, softcap {softcap}", flush=True)

    ids = [tok(t, truncation=True, max_length=args.max_len)["input_ids"] for t in texts]
    lens = np.array([len(x) for x in ids])
    params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(ids) / args.bs / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    rng = np.random.default_rng(args.seed)
    step, bad, t_train = 0, 0, time.time()
    for epoch in range(args.epochs):
        run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, (_, x, m) in enumerate(batches(ids, args.bs, pad_id, bucketed_order(lens, args.bs, rng))):
            x, m = x.cuda(), m.cuda()
            target = x[:, 1:].masked_fill(m[:, 1:] == 0, -100)
            with torch.autocast("cuda", dtype=dtype):
                h = backbone(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
            flat_h = h.reshape(-1, h.size(-1))
            flat_target = target.reshape(-1)
            keep = flat_target.ne(-100)
            n_valid = int(keep.sum())
            if n_valid == 0:
                continue
            hv, tv = flat_h[keep], flat_target[keep]
            # The LM head has a ~262k vocabulary, so all positions' logits at once do not
            # fit next to the 12B model on a T4. The head is applied in chunks to a
            # *detached* copy of the hidden states: each chunk's backward only reaches
            # that copy (the head is frozen), and its logits are freed before the next.
            # Then one backward carries the accumulated gradient through the decoder.
            # Same gradients as the unchunked loss, one decoder backward per batch
            # instead of one per chunk.
            hd = hv.detach().requires_grad_(True)
            batch_loss, ok = 0.0, True
            for start in range(0, n_valid, args.logit_chunk):
                end = min(start + args.logit_chunk, n_valid)
                with torch.autocast("cuda", dtype=dtype):
                    logits = lm_head(hd[start:end]).float()
                if softcap:
                    logits = torch.tanh(logits / softcap) * softcap
                chunk_loss = nn.functional.cross_entropy(logits, tv[start:end],
                                                         reduction="sum") / n_valid
                if not torch.isfinite(chunk_loss):
                    ok = False
                    break
                scaler.scale(chunk_loss / args.grad_accum).backward()
                batch_loss += chunk_loss.item()
                del logits, chunk_loss
            if ok:
                hv.backward(hd.grad)               # gradient is already loss-scaled
                run, n = run + batch_loss, n + 1
            else:
                bad += 1
                if bad > 20:
                    print("ABORT: more than 20 non-finite losses -- fp16 overflow", flush=True)
                    sys.exit(3)
                opt.zero_grad(set_to_none=True)
            del h, flat_h, hv, hd
            if (b + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step == 1:
                    print(f"first step done; peak GPU memory "
                          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB of "
                          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB",
                          flush=True)
                if step % 25 == 0:
                    el = time.time() - t_train
                    print(f"epoch {epoch + 1} step {step}/{steps} lm loss {run / n:.4f} "
                          f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min, "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        print(f"== epoch {epoch + 1}: mean lm loss {run / max(n, 1):.4f}", flush=True)

    out = pathlib.Path(args.out)
    backbone.save_pretrained(out)
    print(f"saved adapter to {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
