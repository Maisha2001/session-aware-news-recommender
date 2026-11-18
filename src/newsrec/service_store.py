from collections import defaultdict
import hashlib
import json

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class Conflict(ValueError):
    pass


class NotFound(ValueError):
    pass


SCHEMA = '''
CREATE SCHEMA IF NOT EXISTS newsrec;
CREATE TABLE IF NOT EXISTS newsrec.catalog_state (
 singleton integer PRIMARY KEY CHECK(singleton=1), revision bigint NOT NULL);
INSERT INTO newsrec.catalog_state VALUES(1,0) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS newsrec.articles (
 article_id text PRIMARY KEY, title text NOT NULL, category text NOT NULL,
 subcategory text NOT NULL, first_seen bigint NOT NULL, fit_clicks bigint NOT NULL CHECK(fit_clicks>=0),
 vector real[] NOT NULL CHECK(array_length(vector,1)=64), digest text NOT NULL);
CREATE TABLE IF NOT EXISTS newsrec.readers (
 user_id text PRIMARY KEY, revision bigint NOT NULL DEFAULT 0, last_at bigint,
 last_request_at bigint, session_start bigint);
CREATE TABLE IF NOT EXISTS newsrec.requests (
 request_id uuid PRIMARY KEY, user_id text NOT NULL REFERENCES newsrec.readers,
 at bigint NOT NULL, signature text NOT NULL, response jsonb NOT NULL);
CREATE INDEX IF NOT EXISTS requests_user_time ON newsrec.requests(user_id,at);
CREATE TABLE IF NOT EXISTS newsrec.feedback (
 feedback_id uuid PRIMARY KEY, request_id uuid NOT NULL REFERENCES newsrec.requests,
 article_id text NOT NULL REFERENCES newsrec.articles, kind text NOT NULL CHECK(kind IN ('click','dismiss')),
 observed_at bigint NOT NULL, signature text NOT NULL, UNIQUE(request_id,article_id));
'''


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class RecommendationStore:
    def __init__(self, dsn):
        self.dsn = dsn
        if psycopg.conninfo.conninfo_to_dict(dsn).get('dbname') not in ('portfolio_news', 'portfolio_news_test'):
            raise ValueError('Use a dedicated portfolio_news database')

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5,
                               options='-c statement_timeout=30000 -c lock_timeout=10000')

    def initialize(self):
        with self.connect() as con:
            con.execute(SCHEMA)

    def seed(self, articles):
        with self.connect() as con:
            state = con.execute('SELECT revision FROM newsrec.catalog_state WHERE singleton=1 FOR UPDATE').fetchone()
            if state['revision'] or con.execute('SELECT 1 FROM newsrec.articles LIMIT 1').fetchone():
                raise Conflict('Initial import requires an empty catalog')
            count = 0
            with con.cursor().copy('COPY newsrec.articles FROM STDIN') as copy:
                for article in articles:
                    copy.write_row((article['article_id'], article['title'], article['category'], article['subcategory'],
                                    article['first_seen'], article['fit_clicks'], article['vector'], signature(article)))
                    count += 1
            if not count:
                raise ValueError('Initial catalog is empty')
            con.execute('UPDATE newsrec.catalog_state SET revision=1 WHERE singleton=1')
            return {'articles': count, 'catalog_revision': 1}

    def add_articles(self, articles):
        if not articles:
            raise ValueError('No articles supplied')
        with self.connect() as con:
            state = con.execute('SELECT revision FROM newsrec.catalog_state WHERE singleton=1 FOR UPDATE').fetchone()
            inserted = 0
            for article in articles:
                fingerprint = signature(article)
                old = con.execute('SELECT digest FROM newsrec.articles WHERE article_id=%s', (article['article_id'],)).fetchone()
                if old:
                    if old['digest'] != fingerprint:
                        raise Conflict('Existing article content is immutable')
                    continue
                con.execute('INSERT INTO newsrec.articles VALUES(%s,%s,%s,%s,%s,%s,%s,%s)',
                            (article['article_id'], article['title'], article['category'], article['subcategory'],
                             article['first_seen'], article['fit_clicks'], article['vector'], fingerprint))
                inserted += 1
            revision = state['revision'] + bool(inserted)
            con.execute('UPDATE newsrec.catalog_state SET revision=%s WHERE singleton=1', (revision,))
            return {'inserted': inserted, 'catalog_revision': revision}

    @staticmethod
    def reader(con, user):
        con.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (user,))
        con.execute('INSERT INTO newsrec.readers(user_id) VALUES(%s) ON CONFLICT DO NOTHING', (user,))
        return con.execute('SELECT * FROM newsrec.readers WHERE user_id=%s FOR UPDATE', (user,)).fetchone()

    def recommend(self, body, engine, cache):
        identity = signature(body)
        with self.connect() as con:
            reader = self.reader(con, body['user_id'])
            old = con.execute('SELECT signature,response FROM newsrec.requests WHERE request_id=%s', (body['request_id'],)).fetchone()
            if old:
                if old['signature'] != identity:
                    raise Conflict('Request ID has different content')
                return old['response']
            at = body['at']
            if reader['last_at'] is not None and at < reader['last_at']:
                raise Conflict('Reader clock must not move backwards')
            start = reader['session_start']
            if reader['last_request_at'] is None or at - reader['last_request_at'] > 1800:
                start = at
            generation = con.execute('SELECT revision FROM newsrec.catalog_state WHERE singleton=1 FOR SHARE').fetchone()['revision']
            rows = con.execute('''SELECT r.at,f.article_id,f.kind FROM newsrec.feedback f
                                  JOIN newsrec.requests r USING(request_id)
                                  WHERE r.user_id=%s AND f.observed_at<%s AND r.at<%s
                                  ORDER BY r.at,f.article_id,f.feedback_id''', (body['user_id'], at, at)).fetchall()
            grouped = defaultdict(list)
            excluded = set()
            for item in rows:
                if item['kind'] == 'dismiss':
                    excluded.add(item['article_id'])
                elif item['at'] >= start and item['at'] + 300 < at:
                    grouped[item['at']].append(item['article_id'])
            groups = list(grouped.items())[-10:]
            key = cache.key({'generation': generation, 'reader_revision': reader['revision'], 'session_start': start,
                             'user': body['user_id'], 'at': at, 'history': body['history'], 'policy': body['policy'],
                             'limit': body['limit'], 'model': engine.fingerprint})
            value = cache.get(key)
            hit = value is not None and value.get('catalog_revision') == generation and value.get('model') == engine.fingerprint
            if not hit:
                load = lambda: con.execute('SELECT * FROM newsrec.articles ORDER BY article_id').fetchall()
                value = engine.recommend(generation, load, body['history'], groups, excluded, at, body['policy'], body['limit'])
                cache.set(key, value)
            response = {**value, 'request_id': body['request_id'], 'at': at, 'policy': body['policy'],
                        'feedback_revision': reader['revision'], 'cache_hit': hit}
            con.execute('INSERT INTO newsrec.requests VALUES(%s,%s,%s,%s,%s)',
                        (body['request_id'], body['user_id'], at, identity, Jsonb(response)))
            con.execute('UPDATE newsrec.readers SET last_at=%s,last_request_at=%s,session_start=%s WHERE user_id=%s',
                        (at, at, start, body['user_id']))
            return response

    def feedback(self, body):
        identity = signature(body)
        with self.connect() as con:
            reader = self.reader(con, body['user_id'])
            old = con.execute('SELECT signature FROM newsrec.feedback WHERE feedback_id=%s', (body['feedback_id'],)).fetchone()
            if old:
                if old['signature'] != identity:
                    raise Conflict('Feedback ID has different content')
                return {'accepted': True, 'duplicate': True}
            request = con.execute('SELECT * FROM newsrec.requests WHERE request_id=%s AND user_id=%s',
                                  (body['request_id'], body['user_id'])).fetchone()
            if not request:
                raise NotFound('Recommendation request not found')
            if body['article_id'] not in {a['article_id'] for a in request['response']['items']}:
                raise ValueError('Feedback must refer to a served article')
            if body['observed_at'] < max(request['at'], reader['last_at']):
                raise Conflict('Feedback observation cannot precede the reader clock')
            if con.execute('SELECT 1 FROM newsrec.feedback WHERE request_id=%s AND article_id=%s',
                           (body['request_id'], body['article_id'])).fetchone():
                raise Conflict('This served article already has feedback')
            con.execute('INSERT INTO newsrec.feedback VALUES(%s,%s,%s,%s,%s,%s)',
                        (body['feedback_id'], body['request_id'], body['article_id'], body['kind'], body['observed_at'], identity))
            con.execute('UPDATE newsrec.readers SET revision=revision+1,last_at=%s WHERE user_id=%s',
                        (body['observed_at'], body['user_id']))
            return {'accepted': True, 'duplicate': False}
