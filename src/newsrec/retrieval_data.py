from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re

import numpy as np

from .data import connect_readonly, file_sha256


def text_tokens(title, category, subcategory, buckets=32768, length=32):
    if buckets < 2 or length < 1:
        raise ValueError('Invalid token dimensions')
    words = ['category:' + category.lower(), 'subcategory:' + subcategory.lower()]
    words += re.findall(r'[a-z0-9]+', title.lower())
    values = [1 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], 'big') % (buckets - 1) for word in words[:length]]
    return values + [0] * (length - len(values))


def seconds(value):
    return int((datetime.fromisoformat(value) - datetime(1970, 1, 1)).total_seconds())


def history_indices(history, mapping, length):
    values = [mapping[a] for a in history[-length:]]
    return values + [0] * (length - len(values))


@dataclass
class ContentData:
    article_ids: list
    mapping: dict
    tokens: np.ndarray
    first_seen: np.ndarray
    fit_histories: np.ndarray
    fit_targets: np.ndarray
    fit_users: set
    fit_candidates: set
    fit_content: set
    metadata: dict


def prepare_content(database, buckets=32768, token_length=32, history_length=20):
    with connect_readonly(database) as con:
        provenance = json.loads(con.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0])
        rows = con.execute('SELECT o.news_id,o.first_seen,a.title,a.category,a.subcategory FROM observations o LEFT JOIN articles a USING(news_id) ORDER BY o.news_id').fetchall()
        if any(row[2] is None for row in rows):
            raise ValueError('Content model requires metadata for every observed article; no outcomes are silently dropped')
        ids = [''] + [row[0] for row in rows]
        mapping = {a: i for i, a in enumerate(ids)}
        tokens = np.asarray([[0] * token_length] + [text_tokens(r[2], r[3], r[4], buckets, token_length) for r in rows], dtype=np.int32)
        first_seen = np.asarray([np.iinfo(np.int64).max] + [seconds(r[1]) for r in rows], dtype=np.int64)
        fit_users = {r[0] for r in con.execute("SELECT DISTINCT user_id FROM impressions WHERE split='fit'")}
        fit_candidates = {mapping[r[0]] for r in con.execute("SELECT DISTINCT news_id FROM candidates JOIN impressions USING(impression_id) WHERE split='fit'")}
        fit_content = set(fit_candidates)
        histories, targets = [], []
        query = """SELECT i.history_json,c.news_id FROM impressions i JOIN candidates c USING(impression_id)
                   WHERE i.split='fit' AND c.clicked=1 ORDER BY i.occurred_at,i.impression_id,c.position"""
        for raw_history, article in con.execute(query):
            history = json.loads(raw_history)
            encoded = history_indices(history, mapping, history_length)
            histories.append(encoded)
            targets.append(mapping[article])
            fit_content.update(i for i in encoded if i)
        if not targets:
            raise ValueError('No fitting positives')
    metadata = {'database_sha256': file_sha256(database), 'input_sha256': provenance['input_sha256'], 'cutoff': provenance['cutoff'],
                'token_buckets': buckets, 'token_length': token_length, 'history_length': history_length,
                'articles': len(ids) - 1, 'fit_positive_examples': len(targets), 'fit_users': len(fit_users),
                'fit_candidate_articles': len(fit_candidates), 'fit_content_articles': len(fit_content),
                'text': 'Category, subcategory and title only; fixed SHA256 token hashing, no corpus vocabulary adaptation.',
                'fit': 'All positive targets from fitting impressions; only their pre-impression last20 history items enter the query encoder.',
                'content_sha256': hashlib.sha256(tokens.tobytes()).hexdigest()}
    return ContentData(ids, mapping, tokens, first_seen, np.asarray(histories, dtype=np.int32),
                       np.asarray(targets, dtype=np.int32), fit_users, fit_candidates, fit_content, metadata)


def development_impressions(database, mapping, history_length=20):
    with connect_readonly(database) as con:
        query = """SELECT i.impression_id,i.user_id,i.occurred_at,i.history_json,c.news_id,c.clicked FROM impressions i
            JOIN candidates c USING(impression_id) WHERE i.split='development'
            ORDER BY i.occurred_at,i.impression_id,c.position"""
        previous, items = None, []
        for row in con.execute(query):
            header = tuple(row[:4])
            if previous is not None and header != previous:
                iid, user, at, raw_history = previous
                yield iid, user, seconds(at), history_indices(json.loads(raw_history), mapping, history_length), items
                items = []
            previous = header
            items.append((mapping[row[4]], row[5]))
        if previous is not None:
            iid, user, at, raw_history = previous
            yield iid, user, seconds(at), history_indices(json.loads(raw_history), mapping, history_length), items
