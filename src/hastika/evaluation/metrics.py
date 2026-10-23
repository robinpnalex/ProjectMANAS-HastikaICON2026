"""Evaluate aligned probabilities on labelled data; full fits have no local score."""
import argparse
import json

from sklearn.metrics import accuracy_score, f1_score

from hastika.common.tasks import TASKS
from hastika.common.validation import read_frame
from hastika.evaluation.ensemble import read_bundle


def evaluate(task, gold_csv, prefix):
    gold = read_frame(gold_csv, task, labelled=True)
    p, ids, meta = read_bundle(prefix, task)
    if ids != gold.id.tolist():
        raise ValueError("Gold and prediction ID order differ")
    classes, col, _ = TASKS[task]
    y = gold[col].map(classes.index).to_numpy()
    pred = p.argmax(1)
    return {"task": task, "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(classes)), zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "members": meta["member_count"],
            "per_class_f1": dict(zip(classes, f1_score(y, pred, average=None, labels=range(len(classes)), zero_division=0).tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b"], required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.task, args.gold, args.bundle), indent=2))


if __name__ == "__main__":
    main()
