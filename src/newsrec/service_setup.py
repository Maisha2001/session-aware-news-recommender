import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .data import connect_readonly, file_sha256
from .service_store import RecommendationStore


def initialize_service(database, prepared, content_model, session_model, output, dsn):
    database, prepared, content_model, session_model, output = map(Path, (database, prepared, content_model, session_model, output))
    metadata = json.loads((prepared / 'manifest.json').read_text())
    if file_sha256(database) != metadata['database_sha256']:
        raise ValueError('Source database does not match prepared evidence')
    for name, expected in metadata['sha256'].items():
        if file_sha256(prepared / name) != expected:
            raise ValueError('Prepared data changed: ' + name)
    content = json.loads((content_model / 'training.json').read_text())
    session = json.loads((session_model / 'training.json').read_text())
    if session['config']['data_manifest_sha256'] != file_sha256(prepared / 'manifest.json'):
        raise ValueError('Ranker was fitted to a different prepared catalog')
    weights = {'content': (content_model / 'epoch-3.weights.h5', content['weights_sha256']['epoch-3.weights.h5']),
               'session': (session_model / 'session-epoch-3.weights.h5', session['arms']['session']['weights_sha256'])}
    for source, expected in weights.values():
        if file_sha256(source) != expected:
            raise ValueError('Source model changed')
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    for name, (source, expected) in weights.items():
        target = output / (name + '.weights.h5')
        shutil.copyfile(source, target)
        models[name] = {'file': target.name, 'sha256': expected}
    arrays = {name: np.load(prepared / (name + '.npy'), allow_pickle=False, mmap_mode='r')
              for name in ('article_ids', 'vectors', 'first_seen', 'fit_clicks', 'categories')}
    with connect_readonly(database) as con:
        articles = {row[0]: row[1:] for row in con.execute('SELECT news_id,title,category,subcategory FROM articles')}
    def records():
        for i, article in enumerate(arrays['article_ids'][1:], 1):
            title, category, subcategory = articles[str(article)]
            if category != arrays['categories'][i]:
                raise ValueError('Catalog category mismatch')
            yield {'article_id': str(article), 'title': title, 'category': category, 'subcategory': subcategory,
                   'first_seen': int(arrays['first_seen'][i]), 'fit_clicks': int(arrays['fit_clicks'][i]),
                   'vector': arrays['vectors'][i].tolist()}
    store = RecommendationStore(dsn)
    store.initialize()
    imported = store.seed(records())
    manifest = {'format_version': 1, 'models': models, 'initial_catalog': imported,
                'source_database_sha256': metadata['database_sha256'],
                'prepared_manifest_sha256': file_sha256(prepared / 'manifest.json'),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'feedback_delay_seconds': 300, 'session_gap_seconds': 1800, 'weights_frozen': True}
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize a private service catalog and frozen model package. Existing catalogs are never overwritten.')
    for name in ('database', 'prepared', 'content-model', 'session-model', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = initialize_service(args.database, args.prepared, args.content_model, args.session_model, args.output, os.environ['NEWS_DATABASE_URL'])
    print(json.dumps({'articles': result['initial_catalog']['articles'], 'weights_frozen': True}))
