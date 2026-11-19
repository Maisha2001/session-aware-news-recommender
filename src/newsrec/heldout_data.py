from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import numpy as np

from .data import NEWS_ID, connect_readonly, file_sha256, parse_behavior, timestamp
from .retrieval_data import history_indices, seconds, text_tokens
from .retrieval_model import ContentRetriever, tf
from .session_data import SessionData, SessionHistory, impression_rows, load_session, normalized_mean
from .train_retrieval import write_json


def read_final(database, archive, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    opened = datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(archive) as source:
        for name in ('news.tsv', 'behaviors.tsv'):
            members = [item for item in source.infolist() if item.filename in (name, 'MINDsmall_dev/' + name)]
            if len(members) != 1:
                raise ValueError('Expected one root or MINDsmall_dev member: ' + name)
            with source.open(members[0]) as incoming, (directory / name).open('xb') as target:
                shutil.copyfileobj(incoming, target)
    with connect_readonly(database) as con:
        articles = {r[0]: tuple(r[1:]) for r in con.execute('SELECT * FROM articles')}
        first_seen = {r[0]: seconds(r[1]) for r in con.execute('SELECT * FROM observations')}
        last_training = seconds(con.execute('SELECT max(occurred_at) FROM impressions').fetchone()[0])
        fit_users = {r[0] for r in con.execute("SELECT DISTINCT user_id FROM impressions WHERE split='fit'")}
        fit_content = {r[0] for r in con.execute("SELECT DISTINCT news_id FROM candidates JOIN impressions USING(impression_id) WHERE split='fit'")}
        for (history,) in con.execute("SELECT DISTINCT history_json FROM impressions i WHERE split='fit' AND EXISTS (SELECT 1 FROM candidates c WHERE c.impression_id=i.impression_id AND c.clicked=1)"):
            fit_content.update(json.loads(history)[-20:])
    counts = Counter()
    seen = set()
    with (directory / 'news.tsv').open(encoding='utf-8-sig') as stream:
        for line in stream:
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 8 or not NEWS_ID.fullmatch(fields[0]) or fields[0] in seen:
                raise ValueError('Malformed or duplicate final article metadata')
            seen.add(fields[0]); article = tuple(fields[1:5])
            if fields[0] in articles:
                counts['shared_metadata_ids'] += 1
                counts['changed_shared_metadata_ids'] += articles[fields[0]] != article
            else:
                articles[fields[0]] = article
                counts['new_metadata_ids'] += 1
    rows, identities = [], set()
    labels_opened = datetime.now(timezone.utc).isoformat()
    with (directory / 'behaviors.tsv').open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = parse_behavior(line)
            at = seconds(timestamp(row.time))
            if at <= last_training:
                raise ValueError('Final archive must be strictly later than the entire training archive')
            if row.impression_id in identities:
                raise ValueError('Duplicate final impression ID')
            identities.add(row.impression_id)
            for article in row.history + row.candidates:
                if article not in articles:
                    raise ValueError('Observed article has no metadata; no impressions excluded')
                first_seen[article] = min(first_seen.get(article, at), at)
            rows.append(row)
    if not rows:
        raise ValueError('Empty final partition')
    rows.sort(key=lambda row: (row.time, row.impression_id))
    return articles, first_seen, fit_users, fit_content, rows, {
        'archive_content_opened_at_utc': opened, 'labels_first_parsed_at_utc': labels_opened,
        'labels_access': 'Labels are parsed for strictly delayed causal replay. Current, equal-time and future outcomes never enter a query; metrics wait until all predictions are frozen.',
        'metadata_policy': 'Existing training-archive metadata wins on shared IDs, including any changed text. New metadata is encoded with the frozen hashing/encoder. Availability depends only on observations.',
        'metadata_counts': dict(counts), 'train_last_time': last_training,
        'input_sha256': {name: file_sha256(directory / name) for name in ('news.tsv','behaviors.tsv')},
        'archive_sha256': file_sha256(Path(archive)), 'database_sha256': file_sha256(Path(database)),
        'date_range': [timestamp(rows[0].time), timestamp(rows[-1].time)],
        'timezone': 'Naive dataset timestamps; no documented geographic timezone.'}


def final_contexts(database, rows, mapping, vectors):
    sessions = SessionHistory()
    for at, group in itertools.groupby(impression_rows(database), key=lambda row: row[2]):
        updates = defaultdict(list)
        for _, user, _, _, _, items in group:
            sessions.before(user, at)
            updates[user].extend(mapping[article] for article, label in items if label)
        for user, clicked in updates.items():
            if clicked:
                sessions.after(user, at, normalized_mean(vectors[clicked]))
    count = len(rows)
    events = np.zeros((count, 10, 65), np.float32)
    masks = np.zeros((count, 10), bool)
    position = 0
    for at, group in itertools.groupby(rows, key=lambda row: seconds(timestamp(row.time))):
        updates = defaultdict(list)
        for row in group:
            context = sessions.before(row.user_id, at)
            for j, (when, vector) in enumerate(context):
                events[position, j, :64] = vector
                events[position, j, 64] = min(1, np.log1p(at - when) / np.log1p(1800))
                masks[position, j] = True
            updates[row.user_id].extend(mapping[a] for a, y in zip(row.candidates, row.labels) if y)
            position += 1
        for user, clicked in updates.items():
            if clicked:
                sessions.after(user, at, normalized_mean(vectors[clicked]))
    return events, masks


def prepare_final(database, archive, original_prepared, content_model, output):
    output = Path(output)
    articles, observed, fit_users, fit_content, rows, metadata = read_final(database, archive, output / 'input')
    original = load_session(original_prepared)
    ids = [''] + sorted(observed)
    mapping = {article: i for i, article in enumerate(ids)}
    original_mapping = {article: i for i, article in enumerate(original.arrays['article_ids'])}
    tokens = np.asarray([[0] * 32] + [text_tokens(articles[a][2], articles[a][0], articles[a][1]) for a in ids[1:]], np.int32)
    content = ContentRetriever(tokens)
    content(np.zeros((1,20), np.int32)); content.load_weights(str(content_model))
    vectors = np.zeros((len(ids), 64), np.float32)
    new = []
    fit_clicks = np.zeros(len(ids), np.int64); fit_exposures = np.zeros(len(ids), np.int64)
    for i, article in enumerate(ids[1:], 1):
        if article in original_mapping:
            old = original_mapping[article]
            vectors[i] = original.arrays['vectors'][old]
            fit_clicks[i] = original.arrays['fit_clicks'][old]
            fit_exposures[i] = original.arrays['fit_exposures'][old]
        else:
            new.append(i)
    for start in range(0,len(new),512):
        batch = new[start:start+512]
        vectors[batch] = content.articles(np.asarray(batch,np.int32)).numpy()
    histories = np.asarray([history_indices(row.history, mapping, 20) for row in rows], np.int32)
    long = np.asarray([normalized_mean(vectors[h[h != 0]]) for h in histories], np.float32)
    events, event_mask = final_contexts(database, rows, mapping, vectors)
    candidates = np.asarray([mapping[a] for row in rows for a in row.candidates], np.int32)
    labels = np.asarray([y for row in rows for y in row.labels], np.uint8)
    offsets = np.r_[0, np.cumsum([len(row.candidates) for row in rows])].astype(np.int64)
    arrays = {'article_ids': np.asarray(ids), 'vectors': vectors, 'tokens': tokens, 'histories': histories,
              'first_seen': np.asarray([np.iinfo(np.int64).max] + [observed[a] for a in ids[1:]],np.int64),
              'categories': np.asarray([''] + [articles[a][0] for a in ids[1:]]),
              'fit_clicks': fit_clicks, 'fit_exposures': fit_exposures,
              'fit_content': np.asarray([a in fit_content for a in ids]),
              'long': long, 'events': events, 'event_mask': event_mask,
              'times': np.asarray([seconds(timestamp(row.time)) for row in rows],np.int64),
              'impression_ids': np.asarray(['final:' + row.impression_id for row in rows]),
              'users': np.asarray([row.user_id for row in rows]),
              'cold_user': np.asarray([row.user_id not in fit_users for row in rows]),
              'candidate_ids': candidates, 'labels': labels, 'offsets': offsets}
    prepared = output / 'prepared'; prepared.mkdir(exist_ok=False)
    for name, array in arrays.items():
        np.save(prepared / (name + '.npy'), array, allow_pickle=False)
    metadata.update({'format_version':1,'counts': {'impressions':len(rows),'candidate_entries':len(labels),
        'positive_events':int(labels.sum()),'catalog_articles':len(ids)-1,'new_observed_articles':len(new)},
        'original_prepared_manifest_sha256':file_sha256(Path(original_prepared)/'manifest.json'),
        'content_weights_sha256':file_sha256(Path(content_model)),
        'sha256':{p.name:file_sha256(p) for p in prepared.glob('*.npy')},
        'prepared_at_utc':datetime.now(timezone.utc).isoformat()})
    write_json(prepared / 'manifest.json',metadata)
    return SessionData(prepared, arrays, metadata)
