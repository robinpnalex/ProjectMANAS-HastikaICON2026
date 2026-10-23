"""Task definitions, label order and recorded data contracts."""

TASKS = {
    "b": (["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"],
          "Hate Category",
          "Kannada-English (romanized) hateful YouTube comment: {}\n"
          "Category (Gender, Political, Religion, Geo-political, Violence, Others):"),
    "a": (["Non-Hate", "Hate"], "Label",
          "Kannada-English (romanized) YouTube comment: {}\n"
          "Is this hate speech (Hate or Non-Hate)?:"),
}

FILES = {
    "a": {"train": "binary_train.csv", "validation": "hastika_binary_validation.csv", "test": "hastika_binary_test.csv", "rows": 7193, "holdout": 1079, "fingerprint": "815110ff24"},
    "b": {"train": "multiclass_train.csv", "validation": "hastika_multiclass_validation.csv", "test": "hastika_multiclass_test.csv", "rows": 3532, "holdout": 530, "fingerprint": "f85f4f049b"},
}

SPLIT_SEED = 42
