# Competition context

HASTIKA at ICON 2026 has two independent subtasks on Kannada–English YouTube comments: binary hate detection (A), and six-way category classification of hateful comments (B). The ranking metric is macro-F1; accuracy is also reported.

- [Official ICON shared-task listing and paper submission route](https://www.icon2026.org/shared-tasks)
- [Organizer starting kit, task definitions and citation](https://github.com/shankarb14/Hastika-ICON2026)
- [CodaBench competition](https://www.codabench.org/competitions/17784/)

This repository supports the team's shared-task system-description paper. The final systems are the 12-model Task A and 8-model Task B TAPT ensembles identified in `results/final_scores.json`. Earlier Run 28 and Run 23 records in the research repository do not describe these final ensembles.

The historical final fits combine original train labels with the released same-task validation labels. Confirm phase-specific permissions and external-data terms with the organizers when reusing this protocol. Local holdout recipes use only training-side text for supervised fitting and TAPT; final test labels are not distributed here. Organizer schedules and submission formats may change; consult the competition source for current instructions.
