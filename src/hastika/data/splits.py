"""Regenerate the same-task combined labelled sets and fixed 15% holdouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hastika.common.paths import PROJECT_ROOT, RAW_DATA_DIR
from hastika.common.preprocessing import dedupe_index
from hastika.common.tasks import FILES, TASKS, SPLIT_SEED
from hastika.common.validation import read_frame, check_disjoint, overlap_counts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(ids):
    return hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()[:10]


def prepare(task, raw_dir, out_dir, verify=True):
    spec = FILES[task]
    raw, out = Path(raw_dir), Path(out_dir)
    manifest = json.loads((PROJECT_ROOT / "data/sources.json").read_text())
    names = [spec["train"], spec["validation"]]
    if (raw / spec["test"]).is_file():
        names.append(spec["test"])
    if verify:
        for name in names:
            if sha256(raw / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"{name}: source checksum differs from recorded release")
    combined = pd.concat([read_frame(raw / name, task, labelled=True)
                          for name in names[:2]], ignore_index=True)
    combined = combined.drop_duplicates("id", keep="first").reset_index(drop=True)
    column = TASKS[task][1]
    keep = dedupe_index(combined.Comment.tolist(), combined[column].tolist(), task)
    combined = combined.iloc[keep].reset_index(drop=True)
    y = combined[column].map(TASKS[task][0].index).to_numpy()
    train_i, held_i = train_test_split(np.arange(len(y)), test_size=0.15,
                                     stratify=y, random_state=SPLIT_SEED)
    train, held = combined.iloc[train_i], combined.iloc[held_i]
    check_disjoint(train, held)
    fp = fingerprint(held.id)
    if verify and (len(combined) != spec["rows"] or len(held) != spec["holdout"] or fp != spec["fingerprint"]):
        raise ValueError("Prepared split differs from the historical row counts or ID order")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [("all", combined), ("train", train), ("holdout", held)]:
        df.to_csv(out / f"{name}.csv", index=False)
    report = {"task": task, "split_seed": SPLIT_SEED, "labelled_rows": len(combined),
              "train_rows": len(train), "holdout_rows": len(held),
              "holdout_fingerprint": fp, "sources": {n: sha256(raw / n) for n in names}}
    if len(names) == 3:
        test = read_frame(raw / spec["test"], task)
        # Preserve organizer-defined labelled rows. Report intrinsic text duplicates
        # rather than silently changing the historical fitting set based on test text.
        check_disjoint(combined, test, check_text=False)
        report["labelled_vs_test_overlap"] = overlap_counts(combined, test)
        report["test_rows"] = len(test)
    (out / "split.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b"], required=True)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or PROJECT_ROOT / f"data/derived/task_{args.task}_combined_holdout"
    print(json.dumps(prepare(args.task, args.raw_dir, out), indent=2))


if __name__ == "__main__":
    main()
