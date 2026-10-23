# Reproduction guide

## Environment and availability

Use Python 3.10–3.12. Install the base package for CPU preprocessing and tests, or `.[llm]` after a suitable CUDA PyTorch build for quantized model work. The historical logs identify Transformers 5.16.1 and PEFT 0.20.0, now pinned. The complete original dependency environment and base-model revision were not recorded. The new `uv.lock` represents this release's resolution rather than a recovered historical environment. CPU contracts were checked in a fresh Python 3.12 environment; a separate optional PyTorch CPU check covers the classifier's pooling/head contract.

Set `HF_TOKEN` in your environment or the notebook secret store if model access requires it. No credentials should be written into configs or committed output. The requested model identifier `google/gemma-4-12B` is preserved from research code; future availability/access and architecture support must be checked with the model provider. Set `model_revision` in a copied config to pin an authorized model/tokenizer revision, and keep that new recipe with its output.

Historical full-data fits have no independent local target labels. Official scores are author-reported; reusing them is not evidence that a new environment or modified data policy reproduces those scores.

## Data and planning

Obtain authorized inputs as explained in `data/README.md`. The runner validates source hashes and reconstructs task-specific splits before starting GPU work. Its default action only prints the planned subprocess commands and needs no data or model download:

```bash
hastika-run --config configs/task_a/submitted.json
hastika-run --config configs/task_b/holdout.json --groups run37
```

`submitted.json` means all labelled same-task rows. `holdout.json` fits only the 85% training-side partition, with training-side TAPT. The runner removes held-out/final-test text matches from the external corpus and writes `data_audit.json` with counts and source hashes.

## Execute groups

Each group trains a fresh TAPT adapter and classifiers continuing from that adapter. GPU jobs run sequentially so the runner also works on one GPU. Select one group per Kaggle session if needed. Task A has `run33`, `run37`, `run38_session1`, `run38_session2`, `run38_session3`; Task B has the same groups except `run33`.

```bash
hastika-run --config configs/task_a/submitted.json --groups run33 --execute
hastika-run --config configs/task_a/submitted.json --groups run37 --execute
hastika-run --config configs/task_a/submitted.json --groups run38_session1 --execute
hastika-run --config configs/task_a/submitted.json --groups run38_session2 --execute
hastika-run --config configs/task_a/submitted.json --groups run38_session3 --execute
hastika-run --config configs/task_a/submitted.json --combine-only
```

The shared output root defaults to `artifacts/task_a/full`. Use `--out`, `--raw-dir` and `--external` to work elsewhere. When running separate sessions, collect each completed group under the same root before `--combine-only`.

Failed groups stop the run. No truncated/partial ensemble is labelled final. A classifier reporting first-epoch collapse (exit 4) is retried once with seed + 1000, and effective seeds and exact commands are saved. Other failures stop the runner. Existing group directories are not reused; inspect incomplete jobs and choose a new output path for a rerun. Historical OOM fallback settings are not automatically applied, since they can alter the input stream or effective recipe.

The optimizer arithmetic, R-Drop and detached hidden-gradient chunked TAPT come from the research code. In particular, historical loops update only complete gradient-accumulation windows. This release does not silently change that arithmetic during file reorganization.

## Saved models and prediction

New classifier runs save `adapter/`, `classifier_head.pt`, `tokenizer/`, `model.json`, metrics and probability sidecars. The original historical trainers did not save all of these; their old weights are not bundled. Checkpoint export/reload is new release functionality and still needs a real CUDA train/reload smoke run.

```bash
hastika-predict --checkpoint artifacts/task_a/full/run33/s42 \
  --input data/raw/hastika_binary_test.csv --out artifacts/reloaded_a
```

Probability prefixes have `_probs.npy`, `_ids.csv`, and `_metadata.json` companions. Ensemble helpers reject invalid probabilities, shuffled IDs, incompatible class order, incorrect task, collapsed members and incomplete model counts. Packaging additionally requires the exact reference input ID order:

```bash
hastika-ensemble package --bundle artifacts/task_a/full/final \
  --reference data/raw/hastika_binary_test.csv --out artifacts/submissions/task_a.zip
```

For local holdout evaluation:

```bash
hastika-evaluate --task a --gold artifacts/task_a/holdout/data/holdout.csv \
  --bundle artifacts/task_a/holdout/final
```

## Comparator and checks

`hastika-svm` fits a word/character TF-IDF plus LinearSVC comparator from explicit training/evaluation CSVs. It uses sigmoid calibration for A and softmax margins for B. This is not the character-only variant used in some historical logs, and no stored score should be assigned to it without measuring the matching split.

```bash
hastika-svm --task a --train data/derived/task_a_combined_holdout/train.csv \
  --eval data/derived/task_a_combined_holdout/holdout.csv \
  --predict data/derived/task_a_combined_holdout/holdout.csv --out artifacts/svm_a
python -m unittest discover -s tests -v
```

CPU tests cover synthetic split boundaries, recipe composition, invalid source labels, external text filtering, probability alignment, averaging and packaging. They do not measure GPU training, base-model availability or official accuracy. The paper should state those limits plainly.

## Historical experiment notebooks

`notebooks/experiments/` preserves the original experimental record separately from the supported top-level launchers. Each original notebook has a companion result file, status, code/source hashes and related-log summary. Code cells in the 72 included historical snapshots are unchanged from the source export, while saved outputs and execution counters are cleared. Original clone paths and old experimental APIs remain as historical context; no current-package compatibility or fresh GPU validation is claimed for those notebooks.

For nine excluded cases the notebook is a documentation-only record. It preserves the reported outcome and exclusion reason without restoring label-recovery or excluded evaluation-adaptation implementations. The catalogue accounts for all 81 originals. Historical score claims remain tied to the source log/documentation; modifying a historical notebook to run against current code would be a new reproduction attempt, not proof of its old measurement.
