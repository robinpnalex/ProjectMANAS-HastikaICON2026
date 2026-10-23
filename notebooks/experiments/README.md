# Experiment history

This collection accounts for **81 historical notebooks**: **72 code snapshots** and **9 result-only exclusion records**. The two current final-system launchers remain in the parent `notebooks/` directory.

- [Task A notebooks and results](task_a/README.md)
- [Task B notebooks and results](task_b/README.md)
- [Machine-readable catalogue](../../results/experiment_catalog.json)
- [All notebook outcomes as CSV](../../results/notebook_results.csv)

Each notebook begins with its method/outcome, status, source identity, and a link to its own JSON result record. Result records retain numerical summary extracts from matching research logs, line numbers, log hashes, and a statement of how confidently a log can be bound to the notebook. Raw logs and comments are not redistributed.

No score is invented for an unrun notebook. Full-data fits have no independent local score unless separately evaluated. Failed attempts, collapsed seeds, negative results and unresolved results are retained. Where a notebook was regenerated, the recorded source hash refers to the exported file and does not claim to be the exact measured execution.

Historical code/bootstrap paths are preserved for inspection rather than silently rewritten to current APIs. These notebooks are **archival**, not GPU-validated current-release launchers. Reproducing a historical execution may require its original source revision and environment; the current supported final-system path is described in the repository README.

The earlier instruction to remove leakage-related implementations remains in force: excluded notebooks have documentation cells only. Their outcomes and reasons for exclusion remain visible so the experimental record is not erased. Other historical scores are not automatically certified as clean inductive evaluations; inspect their source/data protocol.
