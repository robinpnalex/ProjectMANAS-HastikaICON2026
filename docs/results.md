# Official and development results

The author confirmed these final competition scores and ensemble membership:

| Task | Macro-F1 | Accuracy | Classifier members |
|---|---:|---:|---:|
| A | 0.825049 | 0.878412 | 12 |
| B | 0.736678008 | 0.787879 | 8 |

Both F1 scores use macro averaging. Rankings were not supplied. Final prediction CSVs are outside the required release. These figures are recorded as author-reported official results, not measurements made by this repository's tests.

A contains Run 33 × 4, Run 37 × 2, and Run 38 sessions 1–3 × 2 each. B contains Run 37 × 2 and Run 38 sessions 1–3 × 2 each. The research combination code uses equal weight per classifier. Configurations record requested seeds; all effective replacement seeds and immutable model revisions were not recovered.

The separate historical local measurements in `results/manifest.csv` must retain their evaluation protocol. In particular:

- A Run 36's 0.8721 is a four-model **holdout** mixture, not the 12-model ensemble's official score.
- B Run 39's 0.7056 is a two-model **holdout** pair, not the eight-model ensemble's official score.
- B Run 23's 0.6872 uses two holdout models; its older full-fit snapshot used four.
- B Run 36 has TAPT 0.6741, TAPT/R-Drop 0.7026, and their mixture 0.6966. Retain all arms when discussing that comparison.

No GPU training or official rescoring was performed for this source release. Its external-text exclusion policy differs from the old stream for one Task A test-text match. Historical scores therefore should not be advertised as newly verified performance of a strictly filtered rerun. See `source_provenance.json` and the data protocol.
