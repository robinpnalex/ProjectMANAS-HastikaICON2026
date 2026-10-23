# Local data setup

The [organizer starting kit](https://github.com/shankarb14/Hastika-ICON2026) and [competition](https://www.codabench.org/competitions/17784/) are the acquisition sources. Obtain data through their authorized route. The organizer's [data-use note](https://github.com/shankarb14/Hastika-ICON2026/blob/main/LICENSE_NOTE.md) states research/non-commercial use and no redistribution; this release does not bundle the comments or derived full-text splits.

Place the following files in `data/raw/`:

| Filename | Rows | Role |
|---|---:|---|
| `binary_train.csv` | 6,446 | A original training labels |
| `hastika_binary_validation.csv` | 806 | A released validation labels |
| `hastika_binary_test.csv` | 806 | A final test inputs |
| `multiclass_train.csv` | 3,159 | B original training labels |
| `hastika_multiclass_validation.csv` | 395 | B released validation labels |
| `hastika_multiclass_test.csv` | 396 | B final test inputs |

The earlier `multiclass_validation_inputs.csv` contains 395 rows and is a different phase from the 396-row final test file. Do not substitute it.

`sources.json` records the byte checksums of the local files used in the research archive. The split generator requires those recorded versions. A differently encoded/new organizer release requires an explicit source-manifest update and comparison of split IDs; changing a hash alone does not establish equivalence.

Run `hastika-prepare --task a` or `--task b`. Each combines only that task's labelled train/validation files, removes duplicate IDs, collapses agreeing normalized-text duplicates, drops conflicting-label groups, and makes a stratified 15% split with seed 42. Expected combined counts are 7,193 and 3,532. Expected holdout ID-order fingerprints are `815110ff24` and `f85f4f049b`.

The organizer-defined labelled sets and final test inputs have zero same-task ID overlap. Normalized text overlap exists: 6 distinct text keys for A and 4 for B. The split report preserves and reports this dataset property. No target labels are inferred from these matches. Independent local holdouts have no normalized-text overlap with their training partition.

## External Kannada text

TAPT uses the processed `offenseval_kn.csv` from the research pipeline, originally derived from `community-datasets/offenseval_dravidian` (Kannada). Put an authorized copy in `data/external/`. Its recorded checksum and 3,247-row count are in `sources.json`.

The historical processing retained targeted insults and `Not_offensive` rows, excluded untargeted/non-Kannada rows, short/native-script-dominant comments, self-duplicates and matches against then-available HASTIKA files. The exact upstream dataset revision was not recorded, and this release does not claim a byte-exact independent rebuild of that processed corpus. Obtain the authorized processed artifact from the authors or reconstruct and document a new source manifest. Public availability of the upstream dataset is not a grant to redistribute HASTIKA text.

Only `Comment` is passed to TAPT. The release runner removes external text matching local holdout or final-test inputs. One normalized comment in the historical external file matches Task A final test text; excluding it changes that data stream relative to the historical implementation. No official scores have been recomputed after that exclusion.

Raw data, external data, derived CSVs and runtime data audits are ignored in Git. They are never copied to the public source release.
