# Session-Aware News Recommendation

A reproducible foundation for chronological news recommendation research.
The current code builds a local interaction database and compares a frozen
popularity baseline with deterministic hash ordering. It evaluates both logged
candidate ranking and retrieval from the previously observed article catalog.

This is an initial foundation. Learned retrieval, session-aware ranking and the
browsing application are not yet implemented.

## Run

Python 3.11 or later, with no runtime package dependencies:

```sh
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -e . -r requirements/test.txt
python -m pytest -q
```

Obtain the **MIND-small training archive** following [data access](docs/data.md).
Extract its `news.tsv` and `behaviors.tsv` into a private directory. Keep the
official validation archive sealed for the final project evaluation.

```sh
newsrec ingest --news /private/mind/train/news.tsv --behaviors /private/mind/train/behaviors.tsv --database /private/news.sqlite --cutoff 2019-11-13T00:00:00
newsrec evaluate --database /private/news.sqlite --output /private/baseline-report.json
```

The cutoff was frozen before scoring after training-only date inspection.
Existing databases and reports are never overwritten. Raw data, article
text, user IDs and per-impression predictions stay outside the repository.

See [evaluation](docs/evaluation.md) for metric definitions and limitations.
Tests use authored fixtures, not benchmark data or measured model evidence.

## Measured baseline

MIND-small training archive: 95,071 fitting impressions (November 9â€“12, 2019),
61,894 chronological development impressions (November 13â€“14), and 51,282 articles.
All 2,454,102 development candidate entries and 94,786 positives are retained.
The official validation archive remains unopened.

| Logged-candidate ranking | AUC | Mean clicked RR | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Frozen popularity | 0.5015 | 0.2225 | 0.2806 | 0.4932 |
| Fixed hash order | 0.5078 | 0.2259 | 0.2829 | 0.4897 |

Popularity catalog recall is **0.24% at 20** and **3.58% at 100**. These use
the previously observed catalog, a much harder candidate pool than logged ranking.
Popularity does not consistently improve on the untrained comparison. In 37,961
development impressions, at least one clicked article was absent from fitting
candidates; this baseline gives those articles no learned content score.
These offline results make no claim about live engagement or statistical significance.

The evaluator took 25.12 seconds on CPU, with a measured 57.43 MB peak worker
working set (61.64 MB for launcher and worker together). An independent TSV audit
matched all 96 metric values across 16 groups and the complete ranked prediction
digest. See [full results and evidence](results/baseline-v1/report.json) and
[evaluation details](docs/evaluation.md). The 40 tests use authored fixtures.

```sh
python scripts/verify_evidence.py
python scripts/audit_baseline.py --news /private/mind/train/news.tsv --behaviors /private/mind/train/behaviors.tsv --database /private/news.sqlite --report /private/baseline-report.json --output /private/audit.json
```
