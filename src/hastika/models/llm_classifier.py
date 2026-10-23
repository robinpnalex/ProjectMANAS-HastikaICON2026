"""Shared QLoRA classifier for Task A and Task B. See docs/reproducibility.md."""
import argparse
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from hastika.common.preprocessing import clean

from hastika.common.tasks import TASKS
from hastika.common.validation import read_frame, check_disjoint
from hastika.evaluation.ensemble import write_bundle
from hastika.models.llm_utils import LORA_TARGETS, text_backbone, load_backbone, bucketed_order, batches

class LLMClassifier(nn.Module):
    def __init__(self, backbone, n_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)      # fp32, trained from scratch

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1                # right padding
        pooled = h[torch.arange(h.size(0), device=h.device), last].float()
        return self.head(self.drop(pooled))


def encode(tok, texts, max_len, max_chars=700, prompt=None):
    """Preserve the historical 700-character cap and tokenizer truncation behavior."""
    prompt = prompt or TASKS["b"][2]
    return [tok(prompt.format(t[:max_chars]), truncation=True, max_length=max_len)["input_ids"]
            for t in texts]


@torch.no_grad()
def predict(model, ids, pad_id, bs, dtype):
    model.eval()
    out = np.zeros((len(ids), model.head.out_features), dtype=np.float32)
    order = np.argsort([len(x) for x in ids])          # length-sorted: less padding
    for idx, x, m in batches(ids, bs, pad_id, order):
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x.cuda(), m.cuda())
        out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="explicit base model/tokenizer revision")
    ap.add_argument("--task", choices=["a", "b"], default="b")
    ap.add_argument("--prompt", choices=["short"], default="short")
    ap.add_argument("--train", required=True, help="CSV with Comment and Hate Category")
    ap.add_argument("--eval", default="", help="labelled CSV, printed per epoch, diagnostic only")
    ap.add_argument("--predict", nargs="*", default=[], help="CSVs to write probabilities for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lora-targets", default=",".join(LORA_TARGETS),
                    help="comma-separated module suffixes to adapt; default is attention + MLP")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weight", choices=["balanced"], default="balanced")
    ap.add_argument("--rdrop", type=float, default=0.0,
                    help="R-Drop weight: two dropout passes tied by symmetric KL")
    ap.add_argument("--init-adapter", default="",
                    help="start from a LoRA adapter saved by Gemma TAPT "
                         "instead of a fresh one")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    args = ap.parse_args()
    lora_targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    if not lora_targets:
        ap.error("--lora-targets must contain at least one module suffix")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU and the llm extra")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    token = os.environ.get("HF_TOKEN") or None

    import peft.import_utils
    # peft probes torchao for every target layer and *raises* when an old one is
    # installed (Kaggle ships 0.10). It is never used here, so report it as absent.
    # This is what crashed Gemma-4-12B in Run 20 after three minutes.
    peft.import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    labels, label_col, prompt = TASKS[args.task]
    tok = AutoTokenizer.from_pretrained(args.model, token=token, cache_dir=args.cache, revision=args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    tr = read_frame(args.train, args.task, labelled=True)
    y = torch.tensor(tr[label_col].map(labels.index).to_numpy())
    X = encode(tok, [clean(t) for t in tr["Comment"]], args.max_len, prompt=prompt)
    lens = np.array([len(x) for x in X])
    print(f"{len(X)} training rows; prompt+comment tokens median {int(np.median(lens))}, "
          f"p99 {int(np.percentile(lens, 99))}, truncated {(lens >= args.max_len).sum()}", flush=True)
    ev = None
    if args.eval:
        ev = read_frame(args.eval, args.task, labelled=True)
        check_disjoint(tr, ev, check_text=True)
        Xev = encode(tok, [clean(t) for t in ev["Comment"]], args.max_len, prompt=prompt)
        yev = ev[label_col].map(labels.index).to_numpy()

    t0 = time.time()
    backbone = load_backbone(args.model, token, args.cache, dtype, revision=args.revision)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_adapter:
        # TAPT applies LoRA to the same decoder stack, so the keys line up
        backbone = PeftModel.from_pretrained(backbone, args.init_adapter, is_trainable=True)
        print("initialised from TAPT adapter", args.init_adapter, flush=True)
    else:
        backbone = get_peft_model(backbone, LoraConfig(
            r=args.r, lora_alpha=2 * args.r, lora_dropout=args.lora_dropout,
            target_modules=lora_targets, bias="none"))
    print(f"LoRA rank={args.r}; target modules={lora_targets}", flush=True)
    if hasattr(backbone, "print_trainable_parameters"):
        backbone.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,} || all params: {total:,}", flush=True)
    model = LLMClassifier(backbone, len(labels))
    model.head.cuda()       # never .cuda() the whole wrapper: 4-bit weights refuse to move
    print(f"loaded in {(time.time() - t0) / 60:.1f} min; "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GB on GPU", flush=True)

    counts = np.bincount(y.numpy(), minlength=len(labels))
    w = len(y) / (len(labels) * counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32).cuda(),
                                  label_smoothing=args.label_smoothing)

    lora = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n.split(".")[0]]
    groups = ([{"params": lora, "lr": args.lr}] if lora else [])
    groups.append({"params": model.head.parameters(), "lr": args.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = math.ceil(len(X) / args.bs / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    torch.cuda.reset_peak_memory_stats()

    bad, step, history = 0, 0, []
    rng = np.random.default_rng(args.seed)
    t_train = time.time()
    model.train()
    for epoch in range(args.epochs):
        order = bucketed_order(lens, args.bs, rng)
        run_loss, n = 0.0, 0
        opt.zero_grad(set_to_none=True)     # drop any partial accumulation window
        for b, (idx, x, m) in enumerate(batches(X, args.bs, pad_id, order)):
            with torch.autocast("cuda", dtype=dtype):
                logits = model(x.cuda(), m.cuda())
            target = y[idx].cuda()
            loss = loss_fn(logits.float(), target)
            if args.rdrop:
                # second pass sees different dropout masks (LoRA and head); the symmetric
                # KL pulls the two predictive distributions together
                with torch.autocast("cuda", dtype=dtype):
                    logits2 = model(x.cuda(), m.cuda())
                lp, lq = torch.log_softmax(logits.float(), -1), torch.log_softmax(logits2.float(), -1)
                kl = 0.5 * (nn.functional.kl_div(lp, lq, log_target=True, reduction="batchmean")
                            + nn.functional.kl_div(lq, lp, log_target=True, reduction="batchmean"))
                loss = 0.5 * (loss + loss_fn(logits2.float(), target)) + args.rdrop * kl
            loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                bad += 1
                if bad > 20:
                    print("ABORT: more than 20 non-finite losses -- fp16 overflow in this model",
                          flush=True)
                    sys.exit(3)
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.grad_accum
            n += 1
            if (b + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 25 == 0:
                    el = time.time() - t_train
                    print(f"epoch {epoch + 1} step {step}/{steps} loss {run_loss / max(n, 1):.4f} "
                          f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min, "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(n, 1), "non_finite": bad}
        if epoch == 0:
            # Task A Run 28, seed 43: the loss spiked to 1.08 in the first steps, then sat at
            # chance (~0.70) for three epochs and every row came out Non-Hate. Catch that
            # after one epoch on a sample of training rows, rather than at the end -- and
            # rather than never, in a full fit with no evaluation rows. Exit code 4 tells
            # the notebook to rerun the job with another seed.
            sample = np.random.default_rng(0).choice(len(X), min(400, len(X)), replace=False)
            share = np.bincount(predict(model, [X[i] for i in sample], pad_id, args.eval_bs,
                                        dtype).argmax(1), minlength=len(labels)) / len(sample)
            prior = counts / counts.sum()
            top = int(share.argmax())
            rec["train_pred_share"] = share.round(3).tolist()
            if share[top] > 0.95 and prior[top] < 0.9:
                print(f"COLLAPSED: after epoch 1, {share[top]:.0%} of training rows are "
                      f"predicted '{labels[top]}' (prior {prior[top]:.0%}); exiting with code 4",
                      flush=True)
                sys.exit(4)
        if ev is not None:
            p = predict(model, Xev, pad_id, args.eval_bs, dtype)
            rec["eval_macro_f1"] = float(f1_score(yev, p.argmax(1), average="macro"))
            print(f"== epoch {epoch + 1}: eval macro-F1 {rec['eval_macro_f1']:.4f} "
                  "(diagnostic only, not used for selection)", flush=True)
        history.append(rec)

    # New release artifact: save trained adapters and the independent fp32 head.
    model.backbone.save_pretrained(out / "adapter")
    torch.save({k: v.detach().cpu() for k, v in model.head.state_dict().items()},
               out / "classifier_head.pt")
    tok.save_pretrained(out / "tokenizer")
    json.dump({"args": vars(args), "classes": labels, "prompt_text": prompt,
               "resolved_model_revision": getattr(model.backbone.config, "_commit_hash", None)},
              open(out / "model.json", "w"), indent=2)

    for path in args.predict:
        df = read_frame(path, args.task)
        p = predict(model, encode(tok, [clean(t) for t in df["Comment"]], args.max_len,
                                  prompt=prompt), pad_id, args.eval_bs, dtype)
        stem = pathlib.Path(path).stem
        write_bundle(out / stem, p, df["id"].tolist(), args.task,
                     provenance={"model": args.model, "seed": args.seed, "train": str(args.train)})
        print(f"wrote {out / (stem + '_probs.npy')} {p.shape}", flush=True)
    json.dump({"args": vars(args), "history": history, "minutes": (time.time() - t0) / 60,
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30},
              open(out / "metrics.json", "w"), indent=2)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
