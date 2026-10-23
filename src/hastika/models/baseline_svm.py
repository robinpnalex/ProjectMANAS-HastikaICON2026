import argparse
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC


class SoftmaxSVC(LinearSVC):
    """LinearSVC with a predict_proba that is just a softmax of the margins.

    Not calibrated probabilities and not claimed to be: the point is a blendable
    score matrix whose argmax still equals the SVM's own decision.
    """

    def predict_proba(self, X):
        d = self.decision_function(X)
        if d.ndim == 1:                       # binary: one margin, two columns
            d = np.column_stack([-d, d])
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

def build(C=0.5, proba="calibrated"):
    feats = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )
    if proba == "calibrated":
        clf = CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced"), cv=3)
    else:
        clf = SoftmaxSVC(C=C, class_weight="balanced")
    return make_pipeline(feats, clf)


def main():
    import json
    from pathlib import Path
    import joblib
    from hastika.common.preprocessing import clean
    from hastika.common.tasks import TASKS
    from hastika.common.validation import read_frame, check_disjoint
    from hastika.evaluation.ensemble import write_bundle
    parser = argparse.ArgumentParser(description="Fit the word/character TF-IDF comparator on an explicit split")
    parser.add_argument("--task", choices=["a", "b"], required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--eval", help="optional independent labelled evaluation file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demojize", action="store_true")
    args = parser.parse_args()
    tr = read_frame(args.train, args.task, labelled=True)
    target = read_frame(args.predict, args.task)
    classes, col, _ = TASKS[args.task]
    text = lambda frame: frame.Comment.map(lambda s: clean(s, demojize=args.demojize))
    model = build(proba="calibrated" if args.task == "a" else "softmax")
    model.fit(text(tr), tr[col].map(classes.index))
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "svm.joblib")
    write_bundle(args.out / "predictions", model.predict_proba(text(target)), target.id, args.task,
                 provenance={"model": "word_char_tfidf_svm", "train": args.train})
    if args.eval:
        ev = read_frame(args.eval, args.task, labelled=True)
        check_disjoint(tr, ev)
        score = f1_score(ev[col].map(classes.index), model.predict(text(ev)), average="macro")
        (args.out / "metrics.json").write_text(json.dumps({"macro_f1": float(score)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
