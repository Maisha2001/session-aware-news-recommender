"""Bounded HTTP replay against preserved development predictions, without labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import httpx
import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def selected(prepared):
    arrays = {n: np.load(prepared / (n + '.npy'), mmap_mode='r', allow_pickle=False)
              for n in ('impression_ids', 'fit', 'event_mask', 'times', 'article_ids', 'first_seen')}
    eligible = np.flatnonzero(~arrays['fit'] & ~arrays['event_mask'].any(axis=1))
    rows = sorted(eligible, key=lambda r: hashlib.sha256(('news-service-parity-v1:' + str(arrays['impression_ids'][r])).encode()).hexdigest())[:128]
    return arrays, rows


class Replay:
    def __init__(self, url, log):
        self.client = httpx.Client(base_url=url, headers={'Authorization': 'Bearer ' + os.environ['NEWS_API_TOKEN']}, timeout=45)
        self.log = log

    def call(self, path, body=None, status=200):
        start = time.perf_counter()
        response = self.client.get(path) if body is None else self.client.post(path, json=body)
        value = response.json()
        seconds = time.perf_counter() - start
        with self.log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'path': path, 'body': body, 'status': response.status_code, 'response': value, 'seconds': seconds}) + '\n')
        assert response.status_code == status, (path, response.status_code, value)
        return value, seconds


def request(at, user, history=None, **kwargs):
    return {'request_id': str(uuid4()), 'user_id': user, 'at': at, 'history': history or [], 'limit': 20, 'policy': 'relevance', **kwargs}


def latency(values):
    return {'count': len(values), 'p50_ms': float(np.percentile(values, 50) * 1000),
            'p95_ms': float(np.percentile(values, 95) * 1000), 'max_ms': max(values) * 1000}


def parity(args, a):
    protocol = json.loads(args.protocol.read_text())
    assert digest(args.reference) == protocol['reference_predictions_sha256']
    arrays, rows = selected(args.prepared)
    ids = [str(arrays['impression_ids'][r]) for r in rows]
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == protocol['selected_impressions_sha256']
    wanted = set(ids)
    prior = {}
    with gzip.open(args.reference, 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['iid'] in wanted:
                prior[record['iid']] = record
    assert set(prior) == wanted
    available = dict(zip(arrays['article_ids'].tolist(), arrays['first_seen'].tolist()))
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as con:
        histories = {iid: json.loads(con.execute('SELECT history_json FROM impressions WHERE impression_id=?', (iid,)).fetchone()[0])[-20:] for iid in ids}
    elapsed, warm, errors, exact = [], [], [], {p: 0 for p in ('relevance', 'fresh', 'balanced')}
    receipts = []
    for j, (iid, row) in enumerate(zip(ids, rows)):
        for policy in exact:
            body = request(int(arrays['times'][row]), 'parity_' + str(j) + '_' + policy, histories[iid], policy=policy)
            value, duration = a.call('/recommendations', body)
            order = [x['article_id'] for x in value['items']]
            reference = prior[iid]['catalog'][policy][:20]
            exact[policy] += order == reference
            assert order == reference, ('Rank parity differs', j, policy)
            assert len(order) == len(set(order)) == 20 and value['session_groups'] == 0
            assert value['candidate_count'] == len(prior[iid]['catalog']['relevance'])
            scores = dict(zip(prior[iid]['catalog']['relevance'], prior[iid]['catalog_relevance_scores']))
            differences = [abs(item['score'] - scores[item['article_id']]) for item in value['items']]
            errors.extend(differences)
            assert max(differences) <= protocol['score_absolute_tolerance']
            assert all(available[item] < body['at'] for item in order)
            assert not value['cache_hit']
            elapsed.append(duration)
            if policy == 'relevance':
                original, _ = a.call('/recommendations', body)
                assert original == value
                cached, speed = a.call('/recommendations', {**body, 'request_id': str(uuid4())})
                assert cached['cache_hit'] and cached['items'] == value['items']
                warm.append(speed)
                receipts.append({'body': body, 'response': value})
    save(args.output / 'restart-receipts.json', receipts[:8])
    # Unique users prevent cache hits and per-user serialization from disguising concurrency.
    calls = [request(int(arrays['times'][rows[j % 128]]), 'concurrent_' + str(j), histories[ids[j % 128]]) for j in range(64)]
    def one(item):
        worker = Replay(args.url, args.output / ('concurrent-' + item['user_id'] + '.jsonl'))
        try:
            value, duration = worker.call('/recommendations', item)
            assert not value['cache_hit'] and len(value['items']) == 20
            return duration
        finally:
            worker.client.close()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(one, calls))
    wall = time.perf_counter() - start
    return {'queries': 128, 'policies_per_query': 3, 'exact_top20_by_policy': exact, 'scores_compared': len(errors),
            'maximum_absolute_score_difference': max(errors), 'strict_availability_passed': True,
            'idempotent_requests': 128, 'cache_hits_checked': 128,
            'first_request_including_lazy_index_ms': elapsed[0] * 1000,
            'serial_uncached_http': latency(elapsed[1:]), 'serial_cached_http': latency(warm),
            'concurrent_uncached_http': {**latency(concurrent), 'clients': 4, 'wall_seconds': wall, 'requests_per_second': len(calls) / wall},
            'limits': 'Label-free subset with no prior session context. Only returned top20 and union size compared; not a new full ranking evaluation. HTTP latency includes SQL, cache and model calls; startup excluded except separately reported lazy index request.'}


def consistency(args, a):
    b = Replay(args.second_url, args.output / 'second-worker.jsonl')
    at = 1573776000
    first = request(at, 'stateful_http')
    value, _ = a.call('/recommendations', first)
    names = [x['article_id'] for x in value['items']]
    click = {'feedback_id': str(uuid4()), 'request_id': first['request_id'], 'user_id': first['user_id'],
             'article_id': names[0], 'kind': 'click', 'observed_at': at}
    accepted, _ = b.call('/feedback', click)
    duplicate, _ = a.call('/feedback', click)
    assert not accepted['duplicate'] and duplicate['duplicate']
    delayed, _ = b.call('/recommendations', request(at + 300, first['user_id']))
    active_body = request(at + 301, first['user_id'])
    active, _ = a.call('/recommendations', active_body)
    assert delayed['session_groups'] == 0 and active['session_groups'] == 1
    assert active['feedback_revision'] == 1 and not active['cache_hit']
    cross, _ = b.call('/recommendations', {**active_body, 'request_id': str(uuid4())})
    assert cross['cache_hit'] and cross['items'] == active['items']
    dismiss = {**click, 'feedback_id': str(uuid4()), 'article_id': names[1], 'kind': 'dismiss', 'observed_at': at + 301}
    a.call('/feedback', dismiss)
    filtered, _ = b.call('/recommendations', request(at + 302, first['user_id']))
    assert names[1] not in [x['article_id'] for x in filtered['items']] and filtered['feedback_revision'] == 2
    save(args.output / 'stateful-restart.json', {'body': first, 'response': value, 'next_at': at + 303, 'excluded': names[1]})
    reset, _ = a.call('/recommendations', request(at + 2103, first['user_id']))
    assert reset['session_groups'] == 0
    changed = {'articles': [{'article_id': 'authored_service_cold_1', 'title': 'Authored satellite research update', 'category': 'science', 'subcategory': 'space', 'first_seen': at}]}
    update, _ = a.call('/articles', changed)
    assert update == {'inserted': 1, 'catalog_revision': 2}
    same, _ = b.call('/articles', changed)
    assert same == {'inserted': 0, 'catalog_revision': 2}
    bad = {'articles': [{**changed['articles'][0], 'article_id': 'authored_rollback_only'}, {**changed['articles'][0], 'title': 'Conflicting immutable title'}]}
    b.call('/articles', bad, 409)
    for endpoint, user in ((a, 'generation_a'), (b, 'generation_b')):
        early, _ = endpoint.call('/recommendations', request(at, user))
        assert early['catalog_revision'] == 2 and 'authored_service_cold_1' not in [x['article_id'] for x in early['items']]
        later, _ = endpoint.call('/recommendations', request(at + 1, user))
        health, _ = endpoint.call('/health')
        assert later['catalog_revision'] == health['index_revision'] == 2 and health['index_articles'] == 51283
    old, _ = b.call('/recommendations', first)
    assert old == value and old['catalog_revision'] == 1
    b.client.close()
    return {'workers': 2, 'strict_delay_seconds': 300, 'inactivity_reset_seconds': 1800,
            'feedback_and_cache_revision_checks': 'passed', 'cross_worker_cache_and_idempotency': 'passed',
            'append_only_catalog_revision': 2, 'index_articles_each_worker': 51283, 'atomic_rollback': 'passed',
            'old_request_receipts_preserved': True, 'new_article_is_authored_fixture': True,
            'limits': 'The new vector is encoded and indexed; insertion does not guarantee appearance in top20.'}


def recovery(args, a):
    if args.phase == 'postgres_down':
        a.call('/health', status=503)
        a.call('/recommendations', request(1573776000, 'pg_failure'), 503)
        return {'durable_store_fail_closed': True, 'http_status': 503}
    if args.phase == 'redis_down':
        value, elapsed = a.call('/recommendations', request(1573776000, 'redis_failure'))
        assert not value['cache_hit'] and len(value['items']) == 20
        health, _ = a.call('/health')
        assert not health['cache_available']
        return {'cache_failure_fallback': True, 'http_ms_including_failed_cache_calls': elapsed * 1000}
    for record in json.loads((args.output / 'restart-receipts.json').read_text()):
        value, _ = a.call('/recommendations', record['body'])
        assert value == record['response']
    state = json.loads((args.output / 'stateful-restart.json').read_text())
    old, _ = a.call('/recommendations', state['body'])
    assert old == state['response']
    value, _ = a.call('/recommendations', request(1573778104, 'stateful_http'))
    assert value['feedback_revision'] == 2 and state['excluded'] not in [x['article_id'] for x in value['items']]
    assert value['catalog_revision'] == 2
    return {'old_receipts_verified': 9, 'feedback_revision_preserved': 2, 'catalog_revision_preserved': 2, 'dismissal_preserved': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['parity', 'consistency', 'restart', 'redis_down', 'postgres_down', 'recovered'])
    p.add_argument('--url', default='http://127.0.0.1:8015')
    p.add_argument('--second-url', default='http://127.0.0.1:8016')
    for name in ('output', 'prepared', 'database', 'reference', 'protocol'):
        p.add_argument('--' + name, type=Path, required=name == 'output')
    args = p.parse_args()
    started = now()
    a = Replay(args.url, args.output / (args.phase + '-http.jsonl'))
    try:
        result = parity(args, a) if args.phase == 'parity' else consistency(args, a) if args.phase == 'consistency' else recovery(args, a)
        save(args.output / (args.phase + '-report.json'), {'phase': args.phase, 'started_at_utc': started, 'completed_at_utc': now(), 'status': 'passed', 'final_holdout_opened': False, **result})
        print(json.dumps({'phase': args.phase, 'status': 'passed'}))
    finally:
        a.client.close()


if __name__ == '__main__':
    main()
