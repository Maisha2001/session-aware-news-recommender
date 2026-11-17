import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from .data import file_sha256


class TemporalIndex:
    """Exact inner-product retrieval with strict first-observation filtering."""

    def __init__(self, vectors, first_seen, article_ids):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0 or not np.isfinite(vectors).all():
            raise ValueError('Expected finite nonempty embedding matrix')
        if len(first_seen) != len(vectors) or len(article_ids) != len(vectors) or len(set(article_ids)) != len(article_ids):
            raise ValueError('Index metadata must match unique article IDs')
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.first_seen = np.asarray(first_seen, dtype=np.int64)
        self.article_ids = list(article_ids)
        self.ties = [hashlib.sha256(('newsrec-tie-v1:' + a).encode()).hexdigest() for a in article_ids]

    def search(self, queries, times, k=100):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        times = np.asarray(times, dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != self.index.d or len(times) != len(queries) or not np.isfinite(queries).all() or k <= 0:
            raise ValueError('Invalid query shape, time or k')
        wanted = np.minimum(k, np.asarray([(self.first_seen < at).sum() for at in times]))
        found = [None] * len(queries)
        pending = list(range(len(queries)))
        limit = min(self.index.ntotal, max(k + 1, 256))
        while pending:
            distances, neighbors = self.index.search(queries[pending], limit)
            retry = []
            for row, query in enumerate(pending):
                candidates = [(float(score), int(index)) for score, index in zip(distances[row], neighbors[row])
                              if index >= 0 and self.first_seen[index] < times[query]]
                candidates.sort(key=lambda item: (-item[0], self.ties[item[1]]))
                enough = len(candidates) >= wanted[query]
                # Expand tied boundaries too; FAISS's arbitrary tie subset must not determine results.
                tied = wanted[query] > 0 and enough and candidates[wanted[query] - 1][0] <= float(distances[row, -1])
                if limit < self.index.ntotal and (not enough or tied):
                    retry.append(query)
                else:
                    found[query] = [(index + 1, score) for score, index in candidates[:wanted[query]]]
            pending = retry
            limit = min(self.index.ntotal, limit * 2)
        return found

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        faiss.write_index(self.index, str(directory / 'vectors.faiss'))
        np.save(directory / 'first_seen.npy', self.first_seen, allow_pickle=False)
        (directory / 'articles.json').write_text(json.dumps(self.article_ids), encoding='utf-8')
        manifest = {'format_version':1, 'dimension':self.index.d, 'articles':self.index.ntotal,
                    'sha256':{name:file_sha256(directory / name) for name in ('vectors.faiss','first_seen.npy','articles.json')}}
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, expected in manifest['sha256'].items():
            if file_sha256(directory / name) != expected:
                raise ValueError('Index artifact hash mismatch: ' + name)
        index = faiss.read_index(str(directory / 'vectors.faiss'))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError('Expected exact inner-product index')
        vectors = index.reconstruct_n(0, index.ntotal)
        return cls(vectors, np.load(directory / 'first_seen.npy', allow_pickle=False), json.loads((directory / 'articles.json').read_text()))
