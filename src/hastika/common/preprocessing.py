"""Shared text cleaning for HASTIKA. The released CSVs are mojibake:
UTF-8 bytes were decoded as latin-1, so Kannada script and emoji are mangled."""
import html
import re

import emoji as emoji_lib
import ftfy

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+")
# 3+ of the same character is emphatic lengthening (sulllle, sooooo). Collapse to
# 2, NOT to 1: doubling is phonemic in Kannada romanization (halu != hallu).
REPEAT3 = re.compile(r"(.)\1{2,}")
LONGVOWEL = re.compile(r"([aeiou])\1")


def fix_mojibake(s: str) -> str:
    """Undo the latin-1/utf-8 round trip, then let ftfy catch the rest."""
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return ftfy.fix_text(s)


def clean(s: str, demojize: bool = False, normalize: str = "") -> str:
    """demojize=True turns 🤣 into :rolling_on_the_floor_laughing:.

    Needed for wordpiece models whose vocab has no emoji at all (MuRIL maps every
    emoji to [UNK]); harmless-to-helpful for sentencepiece models that keep them.
    """
    s = fix_mojibake(str(s))
    s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = URL.sub(" <url> ", s)
    s = MENTION.sub(" <user> ", s)
    if normalize:
        s = REPEAT3.sub(r"\1\1", s)
        if normalize == "aggressive":
            s = LONGVOWEL.sub(r"\1", s)
    if demojize:
        # Bare space delimiters, then underscores to spaces, so the emoji name
        # becomes ordinary English words. Measured on MuRIL over binary_train:
        #   no demojize                151,494 tok / 0.791% UNK
        #   :snake_case: delimiters    174,054 tok / 0.050% UNK
        #   bare, underscores kept     167,946 tok / 0.052% UNK
        #   bare, underscores -> space 161,842 tok / 0.054% UNK  <- this
        # All three fix the UNKs; this one spends the fewest tokens doing it and
        # emits real wordpieces ("rolling on the floor laughing") instead of
        # "_"-fragmented junk.
        s = emoji_lib.demojize(s, delimiters=(" ", " ")).replace("_", " ")
    return WS.sub(" ", s).strip()


def dedupe_index(comments, labels=None, name=""):
    """Row indices to keep after collapsing comments with identical text.

    The released files carry the same comment under several different ids: 35
    groups / 42 redundant rows in binary_train, 12 / 14 in multiclass_train. Most
    are byte-identical before any cleaning. Left in, they get double weight in the
    loss and can straddle a fold boundary, which inflates OOF scores.

    Groups whose labels DISAGREE are dropped whole rather than resolved by
    first-wins or majority -- 3 such groups in binary_train (`Hate` vs `Non-Hate`
    on the same string), 2 in multiclass_train. The disagreement is the annotation
    telling you the comment is contestable without its thread; picking a side
    teaches a boundary the annotators themselves could not agree on.

    The key is clean() WITHOUT demojize, so every model gets the same indices
    whatever its own emoji setting. That matters: ensemble.py stacks OOF arrays by
    row position, so muril.py, baseline_svm.py and train_xlmr.py must drop exactly
    the same rows or the blend silently misaligns.

    Pass labels=None (validation/test inputs) and every row is kept -- submissions
    need one row per released id, duplicates included.
    """
    if labels is None:
        return list(range(len(comments)))

    groups = {}
    for i, c in enumerate(comments):
        groups.setdefault(clean(c).casefold(), []).append(i)

    keep, n_dup, n_conflict = [], 0, 0
    for idxs in groups.values():
        if len(idxs) == 1:
            keep.append(idxs[0])
        elif len({labels[i] for i in idxs}) > 1:
            n_conflict += len(idxs)
        else:
            keep.append(idxs[0])
            n_dup += len(idxs) - 1

    keep.sort()
    if n_dup or n_conflict:
        print(f"dedupe{' ' + name if name else ''}: {len(comments)} -> {len(keep)} rows "
              f"({n_dup} duplicate, {n_conflict} in label-conflict groups)", flush=True)
    return keep
