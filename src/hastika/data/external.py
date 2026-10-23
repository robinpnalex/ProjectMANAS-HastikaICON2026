"""Read only external comment text and exclude held-out/final-test matches."""
import pandas as pd

from hastika.common.preprocessing import clean
from hastika.common.validation import text_keys


def prepare_external(source, exclude_frames, out):
    frame = pd.read_csv(source, encoding="utf-8-sig", keep_default_na=False)
    if "Comment" not in frame:
        raise ValueError("External source requires a Comment column")
    blocked = set().union(*(text_keys(f) for f in exclude_frames))
    # Filtering is an exclusion audit. No evaluation labels are read or inferred.
    mask = frame.Comment.map(lambda t: clean(t).casefold() not in blocked)
    texts = frame.loc[mask, ["Comment"]]
    texts.to_csv(out, index=False)
    return {"source_rows": len(frame), "excluded_rows": int((~mask).sum()), "remaining_rows": len(texts)}
