# Curated release notes

This repository starts with independent Git history. The original project remains the research archive. `results/source_provenance.json` identifies the source commit and hashes used for the reusable methods.

The release includes the final Gemma classifier and TAPT implementation, explicit 12-/8-member ensemble recipes, same-task split reconstruction, a sparse comparator, checkpoint/prediction support and structured results. The raw research chronology, broad sweeps, failed setup logs, speculative models, duplicate notebooks and unscored submission snapshots are outside the active tree. MuRIL-era training orchestration is not yet a supported reproduction path in this focused release.

Cross-task label-recovery code, augmented datasets derived from evaluation overlap, evaluation-side adaptation branches and dependent artifacts are excluded. IDs are metadata for alignment and audit, not classifier inputs. Source CSVs and raw notebook outputs are not redistributed. Excluding an experiment from the release does not change the provenance of its historical measurements; only the documented final results and selected development measurements are recorded here.

The release also adds an external-source exclusion step. The historical processed external corpus has one normalized Task A final-test text match. That match is removed by the new runner. The original organizer-defined same-task labelled rows are preserved, including intrinsic duplicate text with test inputs (6 distinct keys in A, 4 in B), and the data audit reports it. Local holdout construction avoids normalized-text overlap within each task. These observations must inform any claim about contamination or strict inductive evaluation.

Official final results are recorded from the author's clarification. Local development results retain their protocols and model counts. The larger final ensembles must not inherit the scores of smaller holdout ensembles. Historical seed replacements, the full dependency lock and immutable base-model revisions remain incompletely recorded.

The existing rough manuscript stays in the research workspace while the system-description paper is revised around this release. No historical ACL template is presented here as a confirmed ICON 2026 requirement.
