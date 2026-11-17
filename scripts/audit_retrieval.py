"""Audit frozen retrieval predictions against the private training behaviors TSV."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from audit_baseline import key, score, sha256


def audit(behaviors, report_path, output):
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    report = json.loads(report_path.read_text())
    assert sha256(behaviors) == report['data']['input_sha256']['behaviors.tsv']
    cutoff = datetime.fromisoformat(report['data']['cutoff'])
    users, candidates, content, first, development = set(), set(), set(), {}, []
    with behaviors.open(encoding='utf-8-sig') as stream:
        for line in stream:
            iid, user, raw_at, raw_history, raw_items = line.rstrip('\r\n').split('\t')
            at = datetime.strptime(raw_at, '%m/%d/%Y %I:%M:%S %p')
            history = raw_history.split()
            items = [(a, int(y)) for a, y in (v.rsplit('-', 1) for v in raw_items.split())]
            for article in set(history) | {a for a, _ in items}:
                first[article] = min(first.get(article, at), at)
            if at < cutoff:
                users.add(user)
                candidates.update(a for a, _ in items)
                if any(y for _, y in items):
                    content.update(history[-report['training_config']['history_length']:])
            else:
                development.append((at, iid, user, history, items))
    content.update(candidates)
    assert len(content) == report['data']['fit_content_articles']
    development.sort()
    arrival_times = sorted(first.values())
    checks, groups, maximum, hashes = 0, 0, 0.0, {}
    for arm, recorded in report['arms'].items():
        metrics = defaultdict(lambda: defaultdict(list))
        counts, digest = Counter(), hashlib.sha256()
        artifact = report_path.parent / recorded['private_prediction_artifact']['file']
        assert sha256(artifact) == recorded['private_prediction_artifact']['sha256']

        def add(group, values):
            for name, value in values.items():
                metrics[group][name].append(value)

        with gzip.open(artifact, 'rt', encoding='utf-8') as stream:
            for at, iid, user, history, items in development:
                prediction = json.loads(next(stream))
                assert prediction[0] == iid
                ranked, retrieved = prediction[1:]
                ids, values = [row[0] for row in ranked], [row[1] for row in ranked]
                assert len(ids) == len(set(ids)) == len(items)
                assert set(ids) == {a for a, _ in items}
                assert all(math.isfinite(v) for v in values)
                assert ranked == sorted(ranked, key=lambda row:(-row[1], key(row[0])))
                assert len(retrieved) == len(set(retrieved)) == min(100, bisect_left(arrival_times, at))
                assert all(first[a] < at for a in retrieved)
                labels = dict(items)
                positives = {a for a, y in items if y}
                segments = ['all', 'warm_user' if user in users else 'cold_user']
                if not history:
                    segments.append('empty_history')
                if positives - candidates:
                    segments.append('cold_positive_article')
                if positives - content:
                    segments.append('unseen_content_positive')
                for segment in segments:
                    counts[segment + '_impressions'] += 1
                counts['candidate_entries'] += len(items)
                counts['positive_events'] += len(positives)
                counts['unavailable_positive_events'] += sum(first[a] >= at for a in positives)
                counts['zero_positive_impressions'] += not positives
                counts['single_class_impressions'] += len(positives) in (0, len(items))
                logged = score([labels[a] for a in ids], values)
                catalog = {}
                if positives:
                    ranks = [i + 1 for i, a in enumerate(retrieved) if a in positives]
                    catalog = {f'recall@{k}':sum(r <= k for r in ranks) / len(positives) for k in (20, 100)}
                    catalog['ndcg@100'] = math.fsum(1 / math.log2(r + 1) for r in ranks) / math.fsum(1 / math.log2(r + 1) for r in range(1, min(100, len(positives)) + 1))
                    catalog['available_positive_fraction'] = sum(first[a] < at for a in positives) / len(positives)
                for segment in segments:
                    add(('logged_ranking', segment), logged)
                    add(('catalog_retrieval', segment), catalog)
                add(('daily', str(at.date()), 'ranking'), logged)
                add(('daily', str(at.date()), 'catalog'), catalog)
                digest.update(json.dumps([iid, ids, retrieved], separators=(',', ':')).encode() + b'\n')
            assert stream.readline() == ''
        assert dict(counts) == recorded['counts'], (counts, recorded['counts'])
        assert digest.hexdigest() == recorded['prediction_sha256']
        hashes[arm] = digest.hexdigest()
        expected_groups = {(root, segment) for root in ('logged_ranking', 'catalog_retrieval') for segment, values in recorded[root].items() if values}
        expected_groups |= {('daily', day, group) for day, children in recorded['daily'].items() for group, values in children.items() if values}
        assert set(metrics) == expected_groups
        for path, values in metrics.items():
            expected = recorded
            for part in path:
                expected = expected[part]
            assert set(expected) == set(values)
            for name, samples in values.items():
                assert len(samples) == expected[name]['impressions']
                error = abs(math.fsum(samples) / len(samples) - expected[name]['mean'])
                assert error <= 1e-10, (arm, path, name, error)
                maximum = max(maximum, error)
                checks += 1
            groups += 1
    result = {'status':'passed', 'development_impressions_per_arm':len(development), 'arms':len(report['arms']),
              'metric_values_checked':checks, 'metric_groups_checked':groups, 'maximum_absolute_difference':maximum,
              'absolute_tolerance':1e-10, 'prediction_order_sha256':hashes, 'report_sha256':sha256(report_path),
              'source_sha256':sha256(Path(__file__)), 'independent_metric_source_sha256':sha256(Path(__file__).with_name('audit_baseline.py')),
              'method':'Raw TSV labels, history and chronology; frozen label-free predictions; independent rank-sum AUC, positive-rank metrics, accurate sums, all segment/daily groups, every availability/cardinality check. No newsrec imports.',
              'started_at_utc':started, 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
              'wall_seconds':time.perf_counter()-begin, 'final_archive_opened':False}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('behaviors', 'report', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.behaviors, args.report, args.output), indent=2))
