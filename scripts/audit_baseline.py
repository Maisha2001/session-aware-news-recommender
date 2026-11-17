"""Recompute aggregate baseline evidence directly from the private training TSVs."""

import argparse
from bisect import insort
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(article):
    return hashlib.sha256(('newsrec-tie-v1:' + article).encode()).hexdigest()


def score(labels, scores):
    positive_ranks = [i + 1 for i, y in enumerate(labels) if y]
    npos = len(positive_ranks)
    result = {}
    if npos:
        result['mean_clicked_rr'] = math.fsum(1 / r for r in positive_ranks) / npos
        result['first_hit_rr'] = 1 / positive_ranks[0]
        for k in (5, 10):
            result[f'recall@{k}'] = sum(r <= k for r in positive_ranks) / npos
            result[f'ndcg@{k}'] = math.fsum(1 / math.log2(r + 1) for r in positive_ranks if r <= k) / math.fsum(1 / math.log2(r + 1) for r in range(1, min(k, npos) + 1))
    if 0 < npos < len(labels):
        ordered = sorted(zip(scores, labels))
        ranks = defaultdict(list)
        for rank, (value, _) in enumerate(ordered, 1):
            ranks[value].append(rank)
        average = {value: math.fsum(group) / len(group) for value, group in ranks.items()}
        positive_rank_sum = math.fsum(average[value] for value, label in ordered if label)
        result['auc'] = (positive_rank_sum - npos * (npos + 1) / 2) / (npos * (len(labels) - npos))
    return result


def audit(news, behaviors, database, report_path, output):
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    report = json.loads(report_path.read_text())
    assert sha256(news) == report['dataset']['input_sha256']['news.tsv']
    assert sha256(behaviors) == report['dataset']['input_sha256']['behaviors.tsv']
    assert sha256(database) == report['database_sha256']
    cutoff = datetime.fromisoformat(report['dataset']['cutoff'])
    fit, users, first, development = Counter(), set(), {}, []
    partitions = Counter()
    with behaviors.open(encoding='utf-8-sig') as stream:
        for line in stream:
            iid, user, raw_at, history, raw_items = line.rstrip('\r\n').split('\t')
            at = datetime.strptime(raw_at, '%m/%d/%Y %I:%M:%S %p')
            items = [(a, int(y)) for a, y in (item.rsplit('-', 1) for item in raw_items.split())]
            for article in set(history.split()) | {a for a, _ in items}:
                first[article] = min(first.get(article, at), at)
            if at < cutoff:
                partitions['fit'] += 1
                users.add(user)
                for article, y in items:
                    fit[article] += y
            else:
                partitions['development'] += 1
                development.append((at, iid, user, history, items))
    assert dict(partitions) == report['dataset']['partition_counts']
    metrics = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    digest = hashlib.sha256()
    arrivals = sorted((at, a) for a, at in first.items())
    active, cursor = [], 0
    hash_keys = {a: key(a) for a in first}

    def add(group, values):
        for name, value in values.items():
            metrics[group][name].append(value)

    for at, iid, user, history, items in sorted(development):
        while cursor < len(arrivals) and arrivals[cursor][0] < at:
            article = arrivals[cursor][1]
            insort(active, (-fit[article], hash_keys[article], article))
            if len(active) > 100:
                active.pop()
            cursor += 1
        positives = {a for a, y in items if y}
        cold = any(a not in fit for a in positives)
        segments = ['all', 'warm_user' if user in users else 'cold_user']
        if cold:
            segments.append('cold_positive_article')
        counts['impressions'] += 1
        counts['candidates'] += len(items)
        counts['positives'] += len(positives)
        counts['zero_positive_impressions'] += not positives
        counts['empty_history_impressions'] += not history.split()
        counts['cold_user_impressions'] += user not in users
        counts['cold_positive_article_impressions'] += cold
        counts['unavailable_positive_events'] += sum(first[a] >= at for a in positives)
        for model in ('popularity', 'hash_order'):
            ranked = sorted(items, key=lambda item: (-fit[item[0]] if model == 'popularity' else 0, hash_keys[item[0]]))
            values = [fit[a] if model == 'popularity' else -int(hash_keys[a], 16) for a, _ in ranked]
            result = score([y for _, y in ranked], values)
            for segment in segments:
                add(('impression_ranking', model, segment), result)
            add(('daily_ranking', str(at.date()), model), result)
            digest.update(json.dumps([iid, model, [a for a, _ in ranked]], separators=(',', ':')).encode() + b'\n')
        if positives:
            top = [a for _, _, a in active]
            result = {f'recall@{k}': len(set(top[:k]) & positives) / len(positives) for k in (20, 100)}
            result['available_positive_fraction'] = sum(first[a] < at for a in positives) / len(positives)
            for segment in segments:
                add(('catalog_retrieval', segment), result)
    assert dict(counts) == report['counts'], (counts, report['counts'])
    assert digest.hexdigest() == report['prediction_order_sha256']
    maximum, checked = 0.0, 0
    for group, values in metrics.items():
        expected = report
        for part in group:
            expected = expected[part]
        assert set(expected) == set(values)
        for metric, samples in values.items():
            assert len(samples) == expected[metric]['impressions']
            difference = abs(math.fsum(samples) / len(samples) - expected[metric]['mean'])
            assert difference <= 1e-10, (group, metric, difference)
            maximum = max(maximum, difference)
            checked += 1
    observed_groups = set(metrics)
    expected_groups = {(root, a, b) for root in ('impression_ranking', 'daily_ranking') for a, groups in report[root].items() for b in groups if groups[b]}
    expected_groups |= {('catalog_retrieval', s) for s, values in report['catalog_retrieval'].items() if values}
    assert observed_groups == expected_groups
    evidence = {'status': 'passed', 'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'development_impressions': len(development), 'metric_values_checked': checked, 'metric_groups_checked': len(metrics),
                'maximum_absolute_difference': maximum, 'absolute_tolerance': 1e-10,
                'prediction_order_sha256': digest.hexdigest(), 'report_sha256': sha256(report_path),
                'source_sha256': sha256(Path(__file__)), 'wall_seconds': time.perf_counter() - start,
                'method': 'Independent TSV parsing, rank-sum AUC, positive-rank metrics, accurate sums, and chronological active top100 catalog; no newsrec imports.',
                'final_archive_opened': False}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('news', 'behaviors', 'database', 'report', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.news, args.behaviors, args.database, args.report, args.output), indent=2))
