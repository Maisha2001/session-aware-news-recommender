import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .baseline import tie_key
from .data import file_sha256
from .rank_policy import rerank
from .retrieval_data import text_tokens
from .retrieval_index import TemporalIndex
from .retrieval_model import ContentRetriever, tf
from .session_data import normalized_mean
from .session_model import SessionRanker


class RecommendationEngine:
    def __init__(self, manifest_path):
        path = Path(manifest_path)
        config = json.loads(path.read_text(encoding='utf-8'))
        self.fingerprint = file_sha256(path)
        files = {}
        for name, record in config['models'].items():
            target = path.parent / record['file']
            if file_sha256(target) != record['sha256']:
                raise ValueError('Serving model hash mismatch: ' + name)
            files[name] = target
        self.content = ContentRetriever(np.zeros((1, 32), np.int32))
        self.content(np.zeros((1, 20), np.int32))
        self.content.load_weights(str(files['content']))
        self.ranker = SessionRanker(np.zeros((1, 64), np.float32))
        self.ranker({'long': np.zeros((1, 64), np.float32),
                     'events': np.zeros((1, 10, 65), np.float32),
                     'event_mask': np.zeros((1, 10), bool),
                     'candidate_ids': np.zeros((1, 1), np.int32),
                     'ages': np.zeros((1, 1), np.float32)})
        self.ranker.load_weights(str(files['session']))
        self.lock = threading.RLock()
        self.generation = None
        self.rebuilds = 0

    def encode_articles(self, articles):
        with self.lock:
            tokens = np.asarray([text_tokens(a['title'], a['category'], a['subcategory']) for a in articles], np.int32)
            vectors = self.content.articles_from_tokens(tokens).numpy()
            if not np.isfinite(vectors).all():
                raise ValueError('Nonfinite content vectors')
            return vectors.tolist()

    def refresh(self, generation, load_articles):
        if self.generation == generation:
            return
        rows = load_articles()
        if not rows:
            raise ValueError('Catalog is empty')
        vectors = np.asarray([r['vector'] for r in rows], np.float32)
        ids = [r['article_id'] for r in rows]
        index = TemporalIndex(vectors, [r['first_seen'] for r in rows], ids)
        self.rows, self.index = rows, index
        self.mapping = {article: i for i, article in enumerate(ids)}
        self.vectors = vectors
        self.ranker.vectors = tf.constant(vectors)
        self.popular = sorted(range(len(ids)), key=lambda i: (-rows[i]['fit_clicks'], tie_key(ids[i])))
        self.generation = generation
        self.rebuilds += 1

    def recommend(self, generation, load_articles, history, groups, excluded, at, policy, limit):
        with self.lock:
            self.refresh(generation, load_articles)
            if any(a not in self.mapping or self.rows[self.mapping[a]]['first_seen'] > at for a in history):
                raise ValueError('History contains an unknown or future article')
            long = normalized_mean(self.vectors[[self.mapping[a] for a in history]])
            if history:
                mean = np.mean(self.vectors[[self.mapping[a] for a in history]], axis=0)
                query = self.content.query_projection(mean[None, :])
            else:
                query = self.content.query_projection(self.content.empty_query[None, :])
            query = tf.math.l2_normalize(query, axis=-1).numpy()
            learned = [i - 1 for i, _ in self.index.search(query, [at], k=100)[0]]
            popular = [i for i in self.popular if self.rows[i]['first_seen'] < at][:100]
            pool = [i for i in dict.fromkeys(learned + popular) if self.rows[i]['article_id'] not in excluded]
            events = np.zeros((1, 10, 65), np.float32)
            mask = np.zeros((1, 10), bool)
            for j, (when, clicked) in enumerate(groups[-10:]):
                if when + 300 >= at:
                    raise ValueError('Feedback is not yet eligible')
                events[0, j, :64] = normalized_mean(self.vectors[[self.mapping[a] for a in clicked]])
                events[0, j, 64] = min(1, np.log1p(at - when) / np.log1p(1800))
                mask[0, j] = True
            if not pool:
                return {'items': [], 'candidate_count': 0, 'session_groups': int(mask.sum()), 'catalog_revision': generation, 'model': self.fingerprint}
            ages = np.asarray([max(0, (at - self.rows[i]['first_seen']) / 3600) for i in pool], np.float32)
            features = {'long': long[None, :], 'events': events, 'event_mask': mask,
                        'candidate_ids': np.asarray([pool], np.int32),
                        'ages': np.minimum(1, np.log1p(ages) / np.log1p(168))[None, :].astype(np.float32)}
            scores = self.ranker(features).numpy()[0]
            names = [self.rows[i]['article_id'] for i in pool]
            categories = [self.rows[i]['category'] for i in pool]
            selected = rerank(names, scores, self.vectors[pool], categories, ages, policy)
            selected += sorted((i for i in range(len(pool)) if i not in selected), key=lambda i: (-scores[i], tie_key(names[i])))
            recent = normalized_mean(events[0, mask[0], :64])
            items = []
            for position in selected[:limit]:
                i = pool[position]
                row = self.rows[i]
                items.append({'article_id': row['article_id'], 'title': row['title'], 'category': row['category'],
                              'score': float(scores[position]), 'first_seen_age_hours': float(ages[position]),
                              'history_similarity': float(self.vectors[i] @ long),
                              'session_similarity': float(self.vectors[i] @ recent)})
            return {'items': items, 'candidate_count': len(pool), 'session_groups': int(mask.sum()),
                    'catalog_revision': generation, 'model': self.fingerprint}


class RecommendationCache:
    def __init__(self, client=None, ttl=60):
        self.client, self.ttl = client, ttl

    @staticmethod
    def key(value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        return 'newsrec:v1:' + hashlib.sha256(raw).hexdigest()

    def get(self, key):
        if self.client is None:
            return None
        try:
            envelope = json.loads(self.client.get(key) or 'null')
            if not isinstance(envelope, dict) or set(envelope) != {'value', 'digest'}:
                return None
            value = envelope['value']
            raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if hashlib.sha256(raw).hexdigest() != envelope['digest']:
                return None
            if not isinstance(value, dict) or set(value) != {'items', 'candidate_count', 'session_groups', 'catalog_revision', 'model'}:
                return None
            return value
        except (ValueError, TypeError, ConnectionError, TimeoutError):
            return None
        except Exception as error:
            import redis
            if isinstance(error, redis.RedisError):
                return None
            raise

    def set(self, key, value):
        if self.client is None:
            return
        try:
            raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            envelope = {'value': value, 'digest': hashlib.sha256(raw).hexdigest()}
            self.client.set(key, json.dumps(envelope, allow_nan=False), ex=self.ttl)
        except Exception as error:
            import redis
            if not isinstance(error, (redis.RedisError, ConnectionError, TimeoutError)):
                raise
