"""CSV contracts and defensive overlap checks. IDs are never model features."""
from pathlib import Path

import pandas as pd

from .preprocessing import clean
from .tasks import TASKS


def read_frame(path, task, labelled=False):
    frame = pd.read_csv(Path(path), encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)
    required = {"id", "Comment"}
    if labelled:
        required.add(TASKS[task][1])
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: required columns are {sorted(required)}")
    if frame.empty or frame["id"].eq("").any() or frame["id"].duplicated().any():
        raise ValueError(f"{path}: empty dataset, empty IDs, or duplicate IDs")
    if labelled:
        if not set(frame[TASKS[task][1]]).issubset(TASKS[task][0]):
            raise ValueError(f"{path}: labels do not belong to task {task}")
    return frame


def text_keys(frame):
    return {clean(text).casefold() for text in frame["Comment"]}


def overlap_counts(left, right):
    return {"ids": len(set(left["id"].astype(str)) & set(right["id"].astype(str))),
            "texts": len(text_keys(left) & text_keys(right))}


def check_disjoint(left, right, check_text=True):
    counts = overlap_counts(left, right)
    if counts["ids"] or (check_text and counts["texts"]):
        raise ValueError(f"Training/evaluation overlap: {counts}")
    return counts
