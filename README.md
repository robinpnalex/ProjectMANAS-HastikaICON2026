# Project MANAS at HASTIKA / ICON 2026

Training and evaluation code for Kannada–English code-mixed hate-speech classification.

| Task | Official final macro-F1 | Accuracy | Ensemble |
|---|---:|---:|---:|
| A: Hate / Non-Hate | **0.825049** | **0.878412** | 12 classifiers |
| B: six hate categories | **0.736678008** | **0.787879** | 8 classifiers |

These official results were supplied by the project author. They describe the competition systems; the curated release has not been GPU-retrained or officially rescored. See [structured scores](results/final_scores.json) and [result provenance](docs/results.md).

## Final systems

Both tasks use Gemma-4-12B with NF4 4-bit QLoRA, a classification head over the last prompt-token state, and LoRA next-token task-adaptive pretraining (TAPT). Task A uses one TAPT epoch and three classifier epochs; Task B uses two and four. R-Drop models use a symmetric KL penalty of 0.5. Probabilities are averaged with equal weight per classifier.

| Training group | Task A | Task B |
|---|---:|---:|
| Run 33: TAPT | 4 | — |
| Run 37: TAPT + R-Drop | 2 | 2 |
| Run 38 session 1: TAPT | 2 | 2 |
| Run 38 session 2: TAPT | 2 | 2 |
| Run 38 session 3: TAPT | 2 | 2 |
| **Total** | **12** | **8** |

The training math comes from the research implementation. This repository provides named recipes, split reconstruction, explicit data boundaries, checkpoint export, and ID-aware probability averaging. It also includes a TF-IDF/LinearSVC comparator. Historical exploratory pipelines and prediction CSVs are outside this release.

## Install

Python 3.10–3.12 is supported. For data preparation and CPU checks:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

For training, first install a PyTorch build compatible with your CUDA driver, then:

```bash
python -m pip install -e '.[llm]'
```

Historical logs identify Transformers **5.16.1** and PEFT **0.20.0**. These are pinned in the LLM extra. Other historical package versions and the exact base-model revision were not fully recorded; see [environment limitations](docs/reproducibility.md). A CUDA GPU and access to the base model are needed for QLoRA/TAPT. Historical jobs used individual Tesla T4 GPUs; one complete ensemble requires several long jobs.

`uv.lock` records this release's dependency resolution, including the optional LLM stack; `uv sync --locked --extra llm` installs that stack. The lockfile is not a recovered historical training environment.

## Obtain data

Get organizer files under their terms and place them in `data/raw/`; place the prepared external Kannada corpus at `data/external/offenseval_kn.csv`. This repository contains no comment data. [Data instructions and checksums](data/README.md) explain the expected files, permissions, and external-corpus availability.

```bash
hastika-prepare --task a
hastika-prepare --task b
```

The historical local holdouts have 1,079 rows for A and 530 for B. Final test inputs have 806 and **396** rows respectively. The split command preserves recorded row order/fingerprints and reports intrinsic source overlap without inferring labels across tasks.

## Plan and run

The runner prints commands without starting training by default:

```bash
hastika-run --config configs/task_a/submitted.json
hastika-run --config configs/task_b/submitted.json
```

Run one group to fit a session-sized part of the ensemble:

```bash
hastika-run --config configs/task_a/submitted.json --groups run38_session1 --execute
```

Run the other groups into the same output root, then combine their completed bundles:

```bash
hastika-run --config configs/task_a/submitted.json --combine-only
```

Omit `--groups` to execute all groups sequentially and combine them automatically. `holdout.json` uses the same recipe on the training-side partition and produces local evaluation bundles. The runner stops on failed jobs and refuses incomplete final ensembles. It retries a collapsed classifier once with seed + 1000 and records effective seeds. See [full reproduction instructions](docs/reproducibility.md).

## Code map

```text
configs/               task-specific full-fit and local-holdout recipes
src/hastika/common/    preprocessing, task definitions, CSV contracts
src/hastika/data/      same-task splits and external text exclusions
src/hastika/models/    QLoRA classifier, chunked TAPT, TF-IDF/SVM
src/hastika/evaluation/ probability alignment, averaging, metrics, ZIP packaging
src/hastika/inference/ checkpoint reload and prediction
notebooks/             small Kaggle launchers pinned to release tag v0.1.0
results/               official scores and historical local result records
docs/                  competition, data, results, reproduction and release notes
paper/                 manuscript status and bibliography
tests/                 CPU data/provenance/alignment checks using synthetic fixtures
```

Raw/derived comments, checkpoints, probability arrays and generated submissions stay in ignored local directories. Cross-task label-recovery experiments, derived-label datasets, and their outputs are excluded. The source audit and changes from the historical implementation are described in [release notes](docs/release_notes.md).

## Citation and terms

Please cite the HASTIKA dataset paper and the shared-task overview when available. See [paper_refs.bib](paper/paper_refs.bib), [CITATION.cff](CITATION.cff), and [NOTICE.md](NOTICE.md). Dataset and base-model terms remain separate from project code rights. The authors have not yet selected a general code redistribution license.
