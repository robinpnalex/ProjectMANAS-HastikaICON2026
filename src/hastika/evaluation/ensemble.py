"""ID- and class-order-aware probability bundles and submission packaging."""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hastika.common.tasks import TASKS


def bundle_file(prefix, suffix):
    prefix = Path(prefix)
    return prefix.parent / (prefix.name + suffix)


def validate_probs(probs, rows, classes):
    if probs.shape != (rows, classes) or not np.isfinite(probs).all():
        raise ValueError("Invalid probability shape or non-finite values")
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-3):
        raise ValueError("Probabilities must be non-negative and sum to one")


def write_bundle(prefix, probs, ids, task, member_count=1, provenance=None):
    ids = list(map(str, ids))
    if not ids or len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ValueError("Prediction IDs must be nonempty and unique")
    if type(member_count) is not int or member_count < 1:
        raise ValueError("member_count must be a positive integer")
    probs = np.asarray(probs)
    labels = TASKS[task][0]
    validate_probs(probs, len(ids), len(labels))
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(bundle_file(prefix, "_probs.npy"), probs)
    pd.DataFrame({"id": ids}).to_csv(bundle_file(prefix, "_ids.csv"), index=False)
    bundle_file(prefix, "_metadata.json").write_text(json.dumps({
        "schema_version": 1, "task": task, "classes": labels,
        "member_count": member_count, "provenance": provenance or {},
    }, indent=2) + "\n")


def read_bundle(prefix, task=None):
    meta = json.loads(bundle_file(prefix, "_metadata.json").read_text())
    task = task or meta["task"]
    if meta["task"] != task or meta["classes"] != TASKS[task][0]:
        raise ValueError("Wrong task or probability-column order")
    count = meta["member_count"]
    if type(count) is not int or count < 1:
        raise ValueError("Invalid member count")
    frame = pd.read_csv(bundle_file(prefix, "_ids.csv"), dtype=str, keep_default_na=False)
    if list(frame.columns) != ["id"] or frame.empty or frame.id.duplicated().any() or frame.id.eq("").any():
        raise ValueError("Invalid prediction ID sidecar")
    probs = np.load(bundle_file(prefix, "_probs.npy"), allow_pickle=False)
    validate_probs(probs, len(frame), len(meta["classes"]))
    return probs, frame.id.tolist(), meta


def combine(prefixes, task, out, expected_members=None, reject_collapsed=True):
    total, ids, count = None, None, 0
    constituents = []
    for prefix in prefixes:
        p, member_ids, meta = read_bundle(prefix, task)
        if ids is not None and member_ids != ids:
            raise ValueError("Ensemble member IDs are reordered or refer to different rows")
        if reject_collapsed and np.bincount(p.argmax(1), minlength=p.shape[1]).max() / len(p) > .9:
            raise ValueError(f"Collapsed predictions in {prefix}")
        ids = member_ids
        n = meta["member_count"]
        total = p.astype(np.float64) * n if total is None else total + p * n
        count += n
        constituents.append({"prefix": str(prefix), "members": n, "provenance": meta.get("provenance", {})})
    if not count or (expected_members is not None and count != expected_members):
        raise ValueError(f"Ensemble has {count} members; expected {expected_members}")
    write_bundle(out, total / count, ids, task, count, {"constituents": constituents})
    return out


def package(prefix, reference_csv, out_zip):
    probs, ids, meta = read_bundle(prefix)
    ref = pd.read_csv(reference_csv, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)
    if "id" not in ref or ref.id.duplicated().any() or ids != ref.id.tolist():
        raise ValueError("Prediction IDs must exactly match reference input order")
    frame = pd.DataFrame({"id": ids, "label": [meta["classes"][i] for i in probs.argmax(1)]})
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("predictions.csv", frame.to_csv(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--task", choices=["a", "b"], required=True)
    c.add_argument("--bundles", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--expected-members", type=int)
    p = sub.add_parser("package")
    p.add_argument("--bundle", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "combine":
        combine(args.bundles, args.task, args.out, args.expected_members)
    else:
        package(args.bundle, args.reference, args.out)


if __name__ == "__main__":
    main()
