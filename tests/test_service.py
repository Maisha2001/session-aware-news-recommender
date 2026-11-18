from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
pytest.importorskip('fastapi')
pytest.importorskip('psycopg')
pytest.importorskip('tensorflow')
import numpy as np
import psycopg
import redis
from fastapi.testclient import TestClient

from newsrec.api import create_app
from newsrec.data import file_sha256
from newsrec.service_engine import RecommendationCache, RecommendationEngine
from newsrec.service_store import RecommendationStore, Conflict


class FixtureEngine:
    fingerprint = 'authored-fixture'

    def __init__(self):
        self.calls = []

    def encode_articles(self, articles):
        return [[1.0] + [0.0] * 63 for _ in articles]

    def recommend(self, generation, load, history, groups, excluded, at, policy, limit):
        self.calls.append({'generation': generation, 'groups': groups, 'history': history})
        rows = [r for r in load() if r['first_seen'] < at and r['article_id'] not in excluded]
        return {'items': [{'article_id': r['article_id'], 'title': r['title'], 'category': r['category'], 'score': 0.0}
                          for r in rows[:limit]], 'candidate_count': len(rows), 'session_groups': len(groups),
                'catalog_revision': generation, 'model': self.fingerprint}


def article(name='A', first_seen=0):
    return {'article_id': name, 'title': 'Authored article ' + name, 'category': 'science',
            'subcategory': 'space', 'first_seen': first_seen, 'fit_clicks': 0, 'vector': [1.0] + [0.0] * 63}


def query(at=1000, user='reader', **kwargs):
    return {'request_id': str(uuid4()), 'user_id': user, 'at': at, 'history': [], 'policy': 'relevance', 'limit': 10, **kwargs}


def feedback(request, name='A', kind='click', at=None):
    return {'feedback_id': str(uuid4()), 'request_id': request['request_id'], 'user_id': request['user_id'],
            'article_id': name, 'kind': kind, 'observed_at': request['at'] if at is None else at}


@pytest.fixture
def store():
    dsn = os.environ.get('NEWS_TEST_DATABASE_URL')
    if not dsn:
        pytest.skip('Set NEWS_TEST_DATABASE_URL for real PostgreSQL integration tests')
    assert psycopg.conninfo.conninfo_to_dict(dsn)['dbname'] == 'portfolio_news_test'
    value = RecommendationStore(dsn)
    value.initialize()
    with value.connect() as con:
        con.execute('TRUNCATE newsrec.feedback,newsrec.requests,newsrec.readers,newsrec.articles')
        con.execute('UPDATE newsrec.catalog_state SET revision=0')
    value.seed([article('A'), article('B'), article('C')])
    return value


@pytest.fixture
def cache():
    url = os.environ.get('NEWS_REDIS_URL')
    if not url:
        pytest.skip('Set NEWS_REDIS_URL for real Redis integration tests')
    client = redis.Redis.from_url(url, socket_timeout=.5, socket_connect_timeout=.5)
    assert client.ping()
    value = RecommendationCache(client)
    original = value.key
    namespace = uuid4().hex
    keys = set()
    def key(payload):
        result = original({'test_namespace': namespace, **payload})
        keys.add(result)
        return result
    value.key = key
    yield value
    if keys:
        client.delete(*keys)
    client.close()


def test_request_idempotency_and_conflict(store, cache):
    engine = FixtureEngine()
    body = query()
    first = store.recommend(body, engine, cache)
    assert store.recommend(body, engine, cache) == first and len(engine.calls) == 1
    with pytest.raises(Conflict):
        store.recommend({**body, 'limit': 2}, engine, cache)


def test_cache_hit_and_feedback_revision(store, cache):
    engine = FixtureEngine()
    first = query()
    store.recommend(first, engine, cache)
    assert store.recommend(query(), engine, cache)['cache_hit']
    store.feedback(feedback(first))
    after = store.recommend(query(), engine, cache)
    assert not after['cache_hit'] and after['feedback_revision'] == 1
    assert after['session_groups'] == 0


def test_feedback_delay_is_strict_and_inactivity_resets(store, cache):
    engine = FixtureEngine()
    first = query()
    store.recommend(first, engine, cache)
    store.feedback(feedback(first))
    assert store.recommend(query(at=1300), engine, cache)['session_groups'] == 0
    assert store.recommend(query(at=1301), engine, cache)['session_groups'] == 1
    assert store.recommend(query(at=3102), engine, cache)['session_groups'] == 0


def test_feedback_arrival_and_equal_time_are_excluded(store, cache):
    engine = FixtureEngine()
    first = query()
    store.recommend(first, engine, cache)
    store.feedback(feedback(first, at=1500))
    assert store.recommend(query(at=1500), engine, cache)['session_groups'] == 0
    assert store.recommend(query(at=1501), engine, cache)['session_groups'] == 1
    with pytest.raises(Conflict):
        store.recommend(query(at=1500), engine, cache)


def test_feedback_idempotency_and_served_membership(store, cache):
    engine = FixtureEngine()
    first = query(limit=1)
    store.recommend(first, engine, cache)
    click = feedback(first)
    assert not store.feedback(click)['duplicate']
    assert store.feedback(click)['duplicate']
    with pytest.raises(Conflict):
        store.feedback({**click, 'kind': 'dismiss'})
    with pytest.raises(ValueError, match='served'):
        store.feedback(feedback(first, name='B'))


def test_dismiss_is_future_only_and_user_isolated(store, cache):
    engine = FixtureEngine()
    first = query()
    store.recommend(first, engine, cache)
    store.feedback(feedback(first, kind='dismiss'))
    assert 'A' in [a['article_id'] for a in store.recommend(query(), engine, cache)['items']]
    assert 'A' not in [a['article_id'] for a in store.recommend(query(at=1001), engine, cache)['items']]
    assert 'A' in [a['article_id'] for a in store.recommend(query(at=1001, user='other'), engine, cache)['items']]


def test_catalog_append_generation_and_atomic_rollback(store, cache):
    engine = FixtureEngine()
    store.recommend(query(), engine, cache)
    assert store.add_articles([article('D', 1000)]) == {'inserted': 1, 'catalog_revision': 2}
    current = store.recommend(query(), engine, cache)
    assert not current['cache_hit'] and current['catalog_revision'] == 2 and len(current['items']) == 3
    assert len(store.recommend(query(at=1001), engine, cache)['items']) == 4
    assert store.add_articles([article('D', 1000)])['inserted'] == 0
    with pytest.raises(Conflict):
        store.add_articles([article('E'), {**article(), 'title': 'Changed'}])
    with store.connect() as con:
        assert con.execute("SELECT 1 FROM newsrec.articles WHERE article_id='E'").fetchone() is None


def test_parallel_duplicate_requests_and_feedback(store, cache):
    engine = FixtureEngine()
    body = query()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.recommend(body, engine, cache), range(8)))
    assert results == [results[0]] * 8 and len(engine.calls) == 1
    event = feedback(body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.feedback(event), range(8)))
    assert sum(not x['duplicate'] for x in results) == 1


def test_restart_preserves_request_and_feedback(store, cache):
    first = query()
    expected = store.recommend(first, FixtureEngine(), cache)
    store.feedback(feedback(first))
    restarted = RecommendationStore(store.dsn)
    assert restarted.recommend(first, FixtureEngine(), RecommendationCache()) == expected
    assert restarted.recommend(query(at=1301), FixtureEngine(), RecommendationCache())['session_groups'] == 1


def test_redis_outage_and_corruption_fall_back(store, cache):
    class Offline:
        def get(self, key): raise redis.ConnectionError('Authored outage')
        def set(self, *args, **kwargs): raise redis.ConnectionError('Authored outage')
    response = store.recommend(query(), FixtureEngine(), RecommendationCache(Offline()))
    assert not response['cache_hit']
    key = cache.key({'authored': True})
    cache.client.set(key, '{bad')
    assert cache.get(key) is None
    cache.client.set(key, json.dumps({'value': response, 'digest': 'wrong'}))
    assert cache.get(key) is None


def test_http_auth_validation_and_store_failure(store, cache):
    token = 'authored-test-token-at-least24'
    with TestClient(create_app(store, FixtureEngine(), cache, token)) as client:
        assert client.get('/health').status_code == 401
        client.headers['Authorization'] = 'Bearer ' + token
        assert client.get('/health').json()['status'] == 'ready'
        assert client.post('/recommendations', json=query(limit=21)).status_code == 422
        body = query()
        assert client.post('/recommendations', json=body).status_code == 200
        assert client.post('/recommendations', json={**body, 'at': 1001}).status_code == 409
        original = store.connect
        store.connect = lambda: (_ for _ in ()).throw(psycopg.OperationalError('private connection detail'))
        result = client.post('/recommendations', json=query())
        assert result.status_code == 503 and 'private' not in result.text
        store.connect = original


def test_nonportfolio_database_is_rejected():
    with pytest.raises(ValueError, match='dedicated'):
        RecommendationStore('postgresql://localhost/course_database')


def test_engine_cold_content_and_snapshot_reload(tmp_path):
    from newsrec.retrieval_model import ContentRetriever, tf
    from newsrec.session_model import SessionRanker
    tf.keras.utils.set_random_seed(123)
    model = ContentRetriever(np.zeros((1, 32), np.int32))
    model(np.zeros((1, 20), np.int32))
    model.save_weights(str(tmp_path/'content.weights.h5'))
    ranker = SessionRanker(np.zeros((1, 64), np.float32))
    ranker({'long':np.zeros((1,64),np.float32),'events':np.zeros((1,10,65),np.float32),
            'event_mask':np.zeros((1,10),bool),'candidate_ids':np.zeros((1,1),np.int32),'ages':np.zeros((1,1),np.float32)})
    ranker.save_weights(str(tmp_path/'session.weights.h5'))
    manifest = {'models': {name: {'file': name+'.weights.h5', 'sha256':file_sha256(tmp_path/(name+'.weights.h5'))} for name in ('content','session')}}
    path = tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    engine = RecommendationEngine(path)
    rows = [article('A'), article('B', 1000)]
    vectors = engine.encode_articles(rows)
    for row, vector in zip(rows, vectors):row['vector'] = vector
    before = engine.recommend(1, lambda: rows, [], [], set(), 1000, 'relevance', 10)
    assert [a['article_id'] for a in before['items']] == ['A']
    after = engine.recommend(2, lambda: rows, ['A'], [(1000,['A'])], set(), 1301, 'balanced', 10)
    assert len(after['items']) == 2 and after['session_groups'] == 1 and engine.rebuilds == 2
    with pytest.raises(ValueError, match='future'):
        engine.recommend(2, lambda: rows, ['B'], [], set(), 999, 'relevance', 10)
    with (tmp_path/'session.weights.h5').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        RecommendationEngine(path)
