# Learned content retrieval

This retriever learns article and history representations from fitting clicks.
It uses TensorFlow 2.20, legacy tf-keras 2.20.1, TensorFlow Recommenders 0.7.7
and FAISS 1.12. CPU execution is supported; no pretrained model or paid API is
required. The [recorded environment](../requirements/retrieval-recorded.txt)
pins the packages used for the measured run. The baseline remains dependency-free.

## Run

Use Python 3.13 in a separate environment. On Windows, keep environment and
model-output paths short: TensorFlow checkpoint temporary filenames can exceed
the Windows path limit in deeply nested directories. No system settings need
to change. These commands use paths relative to a short working directory;
substitute your private paths.

```sh
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -e /path/to/session-aware-news-recommender -r /path/to/session-aware-news-recommender/requirements/retrieval.txt
python -m newsrec.train_retrieval --database /private/news.sqlite --output /private/model
python -m newsrec.evaluate_retrieval --database /private/news.sqlite --model /private/model --output /private/retrieval
python /path/to/session-aware-news-recommender/scripts/audit_retrieval.py --behaviors /private/mind/train/behaviors.tsv --report /private/retrieval/report.json --output /private/retrieval-audit.json
```

Use the database created by the [documented training-data pipeline](../README.md).
Set these process environment variables before Python for the measured CPU setup:
`TF_USE_LEGACY_KERAS=1`, `TF_NUM_INTRAOP_THREADS=4`,
`TF_NUM_INTEROP_THREADS=1`, `OMP_NUM_THREADS=4`, and
`TF_ENABLE_ONEDNN_OPTS=0`. The model module selects legacy Keras before importing
TensorFlow; callers that already imported TensorFlow must set that variable first.
The evaluator explicitly uses four FAISS threads. CI also tests a two-thread setup.

Training and evaluation refuse existing output directories. Each epoch saves
weights and an optimizer checkpoint; the completed training record hashes every
weight file. The evaluator loads those saved weights, saves and reloads each
index, and writes aggregate results plus private label-free prediction records.
The supplied training command starts a new run; it does not automatically resume
an interrupted optimizer checkpoint.

Article IDs, title-derived vectors, weights and per-impression predictions stay
private under the dataset's access terms. Public evidence contains only aggregate
metrics, configuration and hashes. To reproduce model access, obtain the licensed
training archive and run the commands above; no weight download is supplied.

## Model and chronology

The article encoder uses category, subcategory and title, with a fixed SHA256
token mapping into 32,768 buckets and at most 32 tokens. A trainable 64-dimensional
word embedding is mean-pooled, projected through a tanh layer and normalized.
The query encoder averages the last 20 supplied historical article vectors,
applies a learned linear projection and normalizes the result. An empty history
uses a learned generic query vector. There are no user-ID or article-ID embeddings.

All 141,558 fitting positive events become training examples. The TFRS objective
uses other batch targets as negatives, removes duplicate target-ID negatives and
uses temperature 0.1. Adam uses learning rate 0.001 and gradient clip norm 1.0.
Three epochs, batch size 256, dimension 64 and seed 20261116 were fixed before
training. The final epoch is evaluated without development-based stopping or
selection. The saved initial weights form an untuned content-overlap comparator.

Tokenization does not fit a vocabulary or other corpus statistics. Text for all
observed articles can be encoded, but gradients come only from fitting targets
and their histories. A new article can be embedded from its content using
`text_tokens` and `ContentRetriever.articles_from_tokens`, without changing the
embedding dimensions or retraining an item-ID table. Hash collisions and
untrained words remain limitations.

The exact FAISS inner-product index retrieves from articles first observed
**strictly before** each query. Same-time and future articles are excluded.
Search expands its candidate count until enough eligible results exist, including
all ties at the result boundary, then uses a fixed SHA256 article tie order.
Earlier unlabeled exposures can expand the catalog; labels do not update weights.
The catalog is the observed research sample, not a complete publisher inventory.

## Evaluation and verification

Both arms use all 61,894 development impressions from November 13–14 in the
training archive. The separate official validation archive remains sealed.
Logged ranking evaluates all displayed candidates. Catalog metrics use a much
larger previously observed pool. Recall@20/100 and catalog nDCG@100 retain
unavailable positive articles in the denominator. Per-impression means and their
denominators are published, including daily and cold-start results.

`cold_user` means absent from fitting impressions, with content history still
available. `cold_positive_article` means at least one clicked item was absent
from fitting candidate exposures. That item may have appeared in fitting
histories. The stricter `unseen_content_positive` segment excludes articles seen
in either fitting candidate exposures or the last 20 fitting histories used by
the encoder. These are impression-level segments: other positives in the same
impression can be warm. They are not permanently-unseen-user/item claims.

For every development query the evaluator checks strict availability, uniqueness
and returned cardinality. It separately compares 128 ID-selected queries per arm
with exhaustive float64 NumPy scores over the whole catalog. Floating-point
boundary differences are allowed only within the frozen tolerances. Saved-weight
reloads produce bitwise-equal query and article vectors on the measured CPU.
The independent auditor rereads raw TSV labels and chronology, checks the frozen
prediction digests, and recomputes all segment and daily metrics using rank-sum
AUC and accurate sums. It does not import the implementation's metric functions.

See [aggregate results](../results/retrieval-v1/report.json),
[fixed protocol](../results/retrieval-v1/protocol.json),
[training](../results/retrieval-v1/training.json),
[independent audit](../results/retrieval-v1/audit.json) and
[runtime evidence](../results/retrieval-v1/runtime.json).

## Limits

This mean-history retrieval representation does not yet model timed sessions.
The provided histories lack per-click timestamps, and article snapshots lack
content-version history. Other clicked items can still be false in-batch negatives;
observed-exposure and popularity biases remain. One seed and one configuration
do not establish statistical significance or cross-platform bitwise retraining.
Timing covers offline batches and local files, not recommendation-service latency.
Offline improvement does not establish live engagement lift. Session ranking
and freshness/diversity experiments are documented in
[session ranking](session-ranking.md). Serving and a browsing demo remain unfinished.
