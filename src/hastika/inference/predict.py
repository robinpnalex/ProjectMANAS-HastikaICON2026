"""Reload a release checkpoint and write ID-aware probabilities."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output bundle prefix")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    import torch
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as compat
        compat.is_torchao_available = lambda: False
    except ImportError:
        pass
    from peft import PeftModel
    from transformers import AutoTokenizer
    from hastika.common.preprocessing import clean
    from hastika.common.validation import read_frame
    from hastika.models.llm_classifier import LLMClassifier, encode, predict
    from hastika.models.llm_utils import load_backbone
    from hastika.evaluation.ensemble import write_bundle
    if not torch.cuda.is_available():
        raise RuntimeError("Quantized inference requires a CUDA GPU and the llm extra")
    meta = json.loads((args.checkpoint / "model.json").read_text())
    saved = meta["args"]
    task = saved["task"]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    revision = meta.get("resolved_model_revision") or saved.get("revision")
    backbone = load_backbone(saved["model"], os.environ.get("HF_TOKEN"), compute_dtype=dtype, revision=revision)
    backbone = PeftModel.from_pretrained(backbone, args.checkpoint / "adapter")
    model = LLMClassifier(backbone, len(meta["classes"]))
    model.head.load_state_dict(torch.load(args.checkpoint / "classifier_head.pt", weights_only=True, map_location="cpu"))
    model.head.cuda()
    tok = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer")
    frame = read_frame(args.input, task)
    ids = encode(tok, [clean(t) for t in frame.Comment], saved["max_len"], prompt=meta["prompt_text"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    probs = predict(model, ids, pad, args.batch_size, dtype)
    write_bundle(args.out, probs, frame.id.tolist(), task, provenance={"checkpoint": str(args.checkpoint)})


if __name__ == "__main__":
    main()
