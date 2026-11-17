# Session-Aware News Recommendation

Chronological news recommendation research with a licensed data pipeline,
frozen baselines, learned content retrieval and an exact temporal vector index.
The learned encoder can represent new articles from titles and categories;
all model results are measured on a fixed chronological development split.
The serving application and final holdout evaluation remain unfinished. This is
a research implementation, not a complete deployed product.

See [learned retrieval setup and methods](docs/retrieval.md) for TensorFlow/TFRS,
FAISS, cold-start definitions, saved-model access and reproducible evaluation.
The [session-ranking workflow](docs/session-ranking.md) adds delayed-feedback
sequence modeling and freshness/diversity controls.

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

MIND-small training archive: 95,071 fitting impressions (November 9-12, 2019),
61,894 chronological development impressions (November 13-14), and 51,282 articles.
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
[evaluation details](docs/evaluation.md). The baseline tests use authored fixtures.

```sh
python scripts/verify_evidence.py
python scripts/audit_baseline.py --news /private/mind/train/news.tsv --behaviors /private/mind/train/behaviors.tsv --database /private/news.sqlite --report /private/baseline-report.json --output /private/audit.json
```

## Measured learned retrieval

The fixed three-epoch content model was trained on all 141,558 fitting clicks
with 2,105,408 parameters. Both content arms use the same 61,894 development
impressions and retain all 94,786 positives. No final holdout data was opened.

| Model | Logged AUC | Logged nDCG@10 | Catalog recall@20 | Catalog recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Frozen popularity | 0.5015 | 0.2806 | 0.24% | 3.58% |
| Initial content weights | 0.5558 | 0.3108 | 0.39% | 0.95% |
| Learned content | 0.6141 | 0.3602 | 0.74% | 2.25% |

Learning improves the content model's observed ranking and retrieval metrics,
but **catalog recall@100 remains below popularity**. The model retrieves only a
small fraction of clicked items from the full observed catalog. Empty-history
recall@100 is 0.067%; the stricter unseen-content-positive segment reaches 1.74%.
These weaknesses remain visible; no checkpoint was selected or retuned using
these outcomes. There is no claim of statistical significance or live engagement lift.

The exact FAISS index contains 51,282 article vectors. All returned articles
pass strict time-availability checks. An independent raw-TSV audit reproduced
every published segment and daily metric and both complete prediction digests.
Exhaustive float64 checks passed the frozen tolerances on 128 queries per arm;
the initial arm had one ordering difference within floating-point tolerance.
Saved-weight reloads were bitwise equal on the measured CPU.

Training took 60.44 seconds on CPU. The complete two-arm evaluation took
164.59 seconds internally, including prediction recording and audits; the
external worker peak was 453.10 MB. Training memory samples were lost in a
monitor-exit race, so no training memory figure is claimed. The index is
13.13 MB per arm, and learned-arm catalog-search batch p95 was 39.34 ms for
128 queries; this is not service latency. See the [full evidence](docs/retrieval.md).

```sh
python -m pip install -e . -r requirements/test.txt -r requirements/retrieval.txt
python -m pytest -q
python scripts/verify_evidence.py
python scripts/verify_retrieval.py
```

The retrieval tests use authored fixtures to cover training, index persistence,
time filtering,
new-article encoding, model reload and the independent metric audit. Lightweight
baseline environments skip the optional modeling test modules.

## Measured session ranking

Two fixed three-epoch rankers use the unchanged content encoder: a long-term-only
comparison and a 32-unit GRU with delayed session feedback. Each has 33,281 ranker
parameters. All 141,558 fitting positives are used with logged nonclicked negatives.
Five-minute feedback availability and 30-minute inactivity sessions are simulation
assumptions because click times are unavailable. Only 3,511 of 61,894 development
impressions have prior session context; all other impressions remain included.

| Logged ranking | AUC | nDCG@10, all | nDCG@10, with session | Categories@10 | First-seen age hours@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long-term only | 0.6146 | 0.3641 | 0.3553 | 4.98 | 26.94 |
| Session | 0.6202 | 0.3680 | 0.3582 | 5.00 | 26.27 |
| Session + fresh | n/a | 0.3666 | 0.3584 | 5.01 | 23.42 |
| Session + balanced | n/a | 0.3652 | 0.3569 | 5.16 | 24.25 |

AUC is reported for model scores; slate controls rearrange the top ten and are
evaluated as rankings. These are one-seed offline observations, not a claim of
statistical significance or causal engagement improvement. Freshness uses first
observation, not an actual publication timestamp. Session article-macro recall@10
was 0.3879 versus 0.3917 for long-term only, so the small aggregate improvement
does not extend to every view of the data. The fresh and balanced presets reduce
logged nDCG slightly while changing the age and category mix.

| Retrieved union, session ranking | Recall@10 | Recall@100 | Candidate-pool recall | Categories@10 | First-seen age hours@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| relevance | 0.99% | 3.39% | 5.61% | 3.16 | 65.39 |
| fresh | 1.24% | 3.39% | 5.61% | 3.18 | 49.78 |
| balanced | 1.11% | 3.39% | 5.61% | 3.80 | 53.78 |

The pool combines the frozen learned 100 and popularity 100 without injecting
clicked items. Its limited coverage remains a bottleneck. Unexposed articles
have unknown relevance: these catalog figures count known logged positives and
do not establish full-catalog relevance or live lift. Exposure-frequency strata
and macro recall across clicked articles are reported as descriptive sensitivity
checks, with no inverse-propensity or unbiased-causal claim.

Independent checks verified all 156,965 context rows, 430 metric values across
50 groups, complete prediction/pool digests and 128 sampled policy queries.
Saved-model reloads matched 256 impressions per arm bitwise. Training took
31.16 seconds; the complete offline evaluation took 374.64 seconds, including
slate policies, metrics and private prediction recording. See
[methods, evidence and limitations](docs/session-ranking.md).

The 55 authored-fixture tests cover the complete workflow. Lightweight environments
run 40 baseline tests and skip the two optional modeling modules. Run
`python scripts/verify_session.py` to verify the published session evidence.
The official validation archive remains unopened. Service integration and the
usable browsing demo remain unfinished.
