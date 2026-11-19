from pathlib import Path
from uuid import uuid4

import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from newsrec.demo import REPLAY_START, ServiceClient, comparisons, create_demo, load_catalog


class Backend:
    def __init__(self):
        self.calls = []
        self.feedback = set()

    def post(self, path, body):
        self.calls.append((path, body))
        if path == '/feedback':
            duplicate = body['feedback_id'] in self.feedback
            self.feedback.add(body['feedback_id'])
            return {'accepted': True, 'duplicate': duplicate}
        return {'request_id': body['request_id'], 'session_groups': 0, 'candidate_count': 1,
                'items': [{'article_id': 'A', 'title': '<script>authored title</script>', 'category': 'science',
                           'score': .4, 'history_similarity': .3, 'session_similarity': 0., 'first_seen_age_hours': 4.}]}


@pytest.fixture
def demo():
    backend = Backend()
    app = create_demo(backend, {'A': {'title': 'Authored science title', 'abstract': 'Authored summary', 'category': 'science'}},
                      [{'id': 'discover', 'label': 'Discover', 'history': [], 'description': 'No history'},
                       {'id': 'science', 'label': 'Science', 'history': ['A'], 'description': 'Authored profile'}], {'development': {'rows': []}})
    with TestClient(app) as client:
        yield client, backend


def begin(client):
    response = client.get('/api/session')
    assert response.status_code == 200
    state = response.json()
    client.headers['X-News-Demo'] = state['csrf']
    return state


def test_demo_static_and_private_cookie(demo):
    client, backend = demo
    page = client.get('/')
    assert page.status_code == 200 and 'Fieldnotes' in page.text
    assert "frame-ancestors 'none'" in page.headers['content-security-policy']
    assert client.get('/assets/app.js').status_code == 200
    response = client.get('/api/session')
    assert 'HttpOnly' in response.headers['set-cookie'] and 'SameSite=strict' in response.headers['set-cookie']
    state = response.json()
    assert 'user_id' not in state and 'request_id' not in state and 'NEWS_API_TOKEN' not in response.text
    assert state['items'][0]['abstract'] == 'Authored summary'
    assert len(backend.calls) == 1
    assert client.get('/api/session').json() == state and len(backend.calls) == 1


def test_demo_requires_session_csrf_and_same_origin(demo):
    client, _ = demo
    assert client.post('/api/action', json={'action': 'refresh'}).status_code == 409
    begin(client)
    assert client.post('/api/action', json={'action': 'refresh'}, headers={'X-News-Demo': 'wrong'}).status_code == 403
    assert client.post('/api/action', json={'action': 'refresh'}, headers={'Origin': 'https://unrelated.example'}).status_code == 403
    assert client.get('/', headers={'Host': 'unrelated.example'}).status_code == 400


def test_demo_feedback_is_explicit_idempotent_and_does_not_advance_time(demo):
    client, backend = demo
    state = begin(client)
    body = {'action': 'feedback', 'value': 'click', 'article_id': 'A'}
    result = client.post('/api/action', json=body).json()
    assert result['at'] == state['at'] == REPLAY_START and result['items'][0]['feedback'] == 'click'
    assert backend.calls[-1][0] == '/feedback' and backend.calls[-1][1]['observed_at'] == REPLAY_START
    assert len(client.post('/api/action', json=body).json()['activity']) == 1
    after = client.post('/api/action', json={'action': 'refresh'}).json()
    assert after['at'] == REPLAY_START + 360 and after['items'][0]['feedback'] is None
    assert backend.calls[-1][0] == '/recommendations'


def test_demo_profile_and_reset_isolate_synthetic_readers(demo):
    client, backend = demo
    begin(client)
    first = backend.calls[-1][1]['user_id']
    value = client.post('/api/action', json={'action': 'profile', 'value': 'science'}).json()
    second = backend.calls[-1][1]['user_id']
    assert first != second and value['profile'] == 'science' and len(value['history']) == 1
    assert backend.calls[-1][1]['history'] == ['A']
    reset = client.post('/api/action', json={'action': 'reset'}).json()
    assert reset['profile'] == 'discover' and backend.calls[-1][1]['user_id'] not in (first, second)


def test_demo_invalid_actions_cannot_inject_history_or_feedback(demo):
    client, backend = demo
    begin(client)
    for payload in ({'action': 'feedback', 'value': 'click', 'article_id': 'not_served'},
                    {'action': 'profile', 'value': 'unknown'}, {'action': 'policy', 'value': 'unknown'},
                    {'action': 'refresh', 'history': ['injected']}):
        assert client.post('/api/action', json=payload).status_code == 422
    assert len(backend.calls) == 1


def test_demo_proxy_cannot_forward_credentials_to_remote_host():
    for url in ('https://outside.example', 'http://outside.example', 'file:///private', 'http://user:pass@localhost'):
        with pytest.raises(ValueError, match='local HTTP'):
            ServiceClient(url, 'authored-token')


def test_demo_comparisons_preserve_separate_candidate_pools():
    reports = comparisons(Path(__file__).resolve().parents[1] / 'results')
    development = reports['development']
    assert development['impressions'] == 61894 and len(development['rows']) == 3
    values = {r['name']: r for r in development['rows']}
    assert values['Session ranking']['logged_ndcg10'] > values['Popularity']['logged_ndcg10']
    assert values['Content retrieval']['catalog_recall100'] < values['Popularity']['catalog_recall100']


def test_demo_environment_paths_and_training_catalog_guard(tmp_path):
    import hashlib
    import json
    import sqlite3
    database = tmp_path / 'fixture.sqlite'
    with sqlite3.connect(database) as con:
        con.executescript('CREATE TABLE articles(news_id TEXT,title TEXT,abstract TEXT,category TEXT); CREATE TABLE observations(news_id TEXT,first_seen TEXT);')
        con.execute('INSERT INTO articles VALUES(?,?,?,?)', ('A', 'Authored title', 'Authored summary', 'news'))
        con.execute('INSERT INTO observations VALUES(?,?)', ('A', '2019-11-13T12:00:00'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_database_sha256': hashlib.sha256(database.read_bytes()).hexdigest()}))
    catalog, profiles = load_catalog(str(database), str(manifest))
    assert catalog['A']['title'] == 'Authored title' and profiles[1]['history'] == ['A']
    manifest.write_text(json.dumps({'source_database_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='training-archive'):
        load_catalog(str(database), str(manifest))
