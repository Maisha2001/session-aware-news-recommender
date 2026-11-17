# Data access and use

The dataset is [MIND](https://msnews.github.io/), introduced by
Wu et al., *MIND: A Large-scale Dataset for News Recommendation*, ACL 2020.
MIND-small contains a sample of users from the larger benchmark.

Read the publisher's [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf)
before downloading. They permit noncommercial research, testing and publication
of results without a material portion of the data. They prohibit redistribution
of the dataset. This project's code does not grant rights to the data.

The official website currently links to the
[yjw1029/MIND mirror](https://huggingface.co/datasets/yjw1029/MIND).
At the November 4, 2026 access check, its downloads required an authenticated
account with access. The older Azure public endpoint was unavailable.
Use the publisher-linked source and its access process; do not use unofficial
copies to bypass access requirements. Never put account tokens into commands,
reports or this repository.

Record the archive revision, byte count and SHA256 after a successful download.
Only extract training `news.tsv` and `behaviors.tsv` for development. Keep the
official validation archive unopened until the final protocol and models are
frozen. The ingestion manifest records exact input hashes; filenames alone do
not establish which archive supplied a file.

The parser retains anonymized IDs locally. No attempt is made to identify users.
URL and entity columns are not imported. Missing article metadata is counted,
without silently dropping impression outcomes. Published fixtures are authored.

MIND lacks article publication times and per-history-click timestamps. A history
is an ordered list before its impression, not an observed timed session. Later
session modeling must state any inferred boundaries and use only earlier events.

## Verified inputs

Archives supplied on November 16, 2026 match the saved publisher-linked mirror
metadata at revision `0f871bdfd1ac4029324a48722ec6d52d714ebf27`:

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| MINDsmall_train.zip | 52,953,372 | `a966e5138ad103376e9817e02395719bf1c62ec56e6e98c30d46fbb991a7fafa` |
| MINDsmall_dev.zip | 30,946,172 | `b315cde1c9b9d45008b5a7c4b2e1f87647659f09f74892ae3899c0005d5d6155` |

The training ZIP contains a `MINDsmall_train/` directory. Only `news.tsv` and
`behaviors.tsv` are extracted; entity/relation vectors are unused. The final
archive was hashed as opaque bytes, with no contents listed or decompressed.
[The input manifest](../results/baseline-v1/data-manifest.json) records extracted
file hashes. Your local database byte hash may differ across SQLite versions;
the input identities, split counts, metrics and ranked prediction digest are the
portable reproducibility checks.
