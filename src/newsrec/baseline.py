from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import time

from .data import connect_readonly, file_sha256
from .metrics import MetricMeans, auc, ranking_metrics


def tie_key(article_id: str) -> str:
    return hashlib.sha256(("newsrec-tie-v1:" + article_id).encode()).hexdigest()


def fit_popularity(con) -> tuple[dict, set]:
    counts = dict(con.execute("""SELECT news_id,sum(clicked) FROM candidates
        JOIN impressions USING(impression_id) WHERE split='fit' GROUP BY news_id"""))
    users = {row[0] for row in con.execute("SELECT DISTINCT user_id FROM impressions WHERE split='fit'")}
    return counts, users


def eligible_topk(ordered_ids: list[str], first_seen: dict, query_time: str, k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    selected = []
    for article_id in ordered_ids:
        if first_seen[article_id] < query_time:
            selected.append(article_id)
            if len(selected) == k:
                break
    return selected


def evaluate(database: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite evaluation evidence")
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    con = connect_readonly(database)
    try:
        provenance = json.loads(con.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0])
        popularity, fit_users = fit_popularity(con)
        seen = dict(con.execute("SELECT news_id,first_seen FROM observations"))
        ordered = sorted(seen, key=lambda a: (-popularity.get(a, 0), tie_key(a)))
        # Zero-score articles are tied; ranks never use candidate file position or labels.
        tie_scores = {a: int(tie_key(a), 16) / (2 ** 256) for a in seen}
        counters = Counter()
        metrics = {model: {segment: MetricMeans() for segment in ("all", "cold_user", "warm_user", "cold_positive_article")}
                   for model in ("popularity", "hash_order")}
        retrieval = {segment: MetricMeans() for segment in ("all", "cold_user", "warm_user", "cold_positive_article")}
        daily = {}
        predictions_digest = hashlib.sha256()
        query = """SELECT i.impression_id,i.user_id,i.occurred_at,i.history_json,c.news_id,c.clicked
            FROM impressions i JOIN candidates c USING(impression_id)
            WHERE i.split='development' ORDER BY i.occurred_at,i.impression_id,c.position"""
        current, items = None, []
        def consume(header, items):
            iid, user, at, history = header
            positives = {a for a, y in items if y}
            cold_positive = any(a not in popularity for a in positives)
            segments = ["all", "warm_user" if user in fit_users else "cold_user"]
            if cold_positive:
                segments.append("cold_positive_article")
            counters["impressions"] += 1
            counters["candidates"] += len(items)
            counters["positives"] += len(positives)
            counters["zero_positive_impressions"] += not positives
            counters["empty_history_impressions"] += not json.loads(history)
            counters["cold_user_impressions"] += user not in fit_users
            counters["cold_positive_article_impressions"] += cold_positive
            counters["unavailable_positive_events"] += sum(seen[a] >= at for a in positives)
            day = at[:10]
            if day not in daily:
                daily[day] = {model: MetricMeans() for model in ("popularity", "hash_order")}
            for model in metrics:
                scores = {a: float(popularity.get(a, 0)) if model == "popularity" else -tie_scores[a] for a, _ in items}
                ranked = sorted(items, key=lambda pair: (-scores[pair[0]], tie_key(pair[0])))
                result = ranking_metrics([y for _, y in ranked])
                result.pop("positive_count")
                result["auc"] = auc([y for _, y in items], [scores[a] for a, _ in items])
                for segment in segments:
                    metrics[model][segment].add(result)
                daily[day][model].add(result)
                predictions_digest.update(json.dumps([iid, model, [a for a, _ in ranked]], separators=(",", ":")).encode() + b"\n")
            top = eligible_topk(ordered, seen, at, 100)
            if positives:
                result = {f"recall@{k}": len(set(top[:k]) & positives) / len(positives) for k in (20, 100)}
                result["available_positive_fraction"] = sum(seen[a] < at for a in positives) / len(positives)
                for segment in segments:
                    retrieval[segment].add(result)
        for row in con.execute(query):
            header = tuple(row[:4])
            if current is not None and header != current:
                consume(current, items)
                items = []
            current = header
            items.append((row[4], row[5]))
        if current is not None:
            consume(current, items)
        report = {
            "format_version": 1, "dataset": provenance, "evaluation_split": "development",
            "holdout_access": "Evaluator reads only its input database; original archive identity is recorded by input hashes.", "counts": dict(counters),
            "impression_ranking": {model: {segment: value.report() for segment, value in groups.items()} for model, groups in metrics.items()},
            "catalog_retrieval": {segment: value.report() for segment, value in retrieval.items()},
            "daily_ranking": {day: {model: value.report() for model, value in groups.items()} for day, groups in daily.items()},
            "prediction_order_sha256": predictions_digest.hexdigest(),
            "database_sha256": file_sha256(database),
            "protocol": {
                "fit": "Only labeled candidates in impressions strictly before cutoff; no history-derived clicks.",
                "development": "Frozen fit counts, no development label updates. Labels used for metrics only.",
                "catalog": "All IDs observed in earlier impressions/histories; strictly earlier time, independent of clicks. New zero-score items retain fixed hash tie order.",
                "cold_user": "User absent from fitting impressions.",
                "cold_positive_article": "At least one positive article absent from all fitting candidates, not merely zero-click.",
                "ranking": "All logged candidates, no negative resampling; ties broken by SHA256(article ID) with fixed namespace.",
                "averaging": "Macro per impression; zero-positive impressions excluded from recall/nDCG/RR, one-class impressions excluded from AUC; counts accompany every mean.",
                "mrr": "mean_clicked_rr averages reciprocal ranks over all clicked candidates; first_hit_rr is reported separately.",
            },
            "limitations": [
                "Logged exposure and active-user sampling bias; no propensity correction or causal engagement claim.",
                "MIND timestamps have no documented timezone; dates are retained in dataset time.",
                "No publication timestamps or text version histories: strict prior observation is a conservative availability proxy, not publication time.",
                "The catalog is the sampled dataset's observed catalog, not the full live publisher catalog.",
                "History contains ordered earlier clicks without per-click timestamps; session IDs and boundaries are not supplied.",
                "Offline research only; popularity is a baseline, not the completed learned recommendation system.",
            ],
            "runtime": {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_seconds": time.perf_counter() - start, "python": platform.python_version(), "platform": platform.platform()},
        }
    finally:
        con.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
