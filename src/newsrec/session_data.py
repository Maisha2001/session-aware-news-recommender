from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import numpy as np

from .data import connect_readonly, file_sha256
from .retrieval_data import seconds
from .retrieval_index import TemporalIndex
from .train_retrieval import write_json


def normalized_mean(vectors):
    if len(vectors) == 0:
        return np.zeros(64, dtype=np.float32)
    mean = np.mean(vectors, axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def impression_rows(database):
    with connect_readonly(database) as con:
        query = '''SELECT i.impression_id,i.user_id,i.occurred_at,i.split,i.history_json,c.news_id,c.clicked
                   FROM impressions i JOIN candidates c USING(impression_id)
                   ORDER BY i.occurred_at,i.impression_id,c.position'''
        for header, group in itertools.groupby(con.execute(query), key=lambda row:tuple(row[:5])):
            iid, user, at, split, history = header
            yield iid, user, seconds(at), split, json.loads(history), [(row[5],row[6]) for row in group]


class SessionHistory:
    def __init__(self, gap_seconds=1800, feedback_delay_seconds=300, length=10):
        if gap_seconds <= 0 or feedback_delay_seconds < 0 or length < 1:
            raise ValueError('Invalid session configuration')
        self.gap, self.delay, self.length = gap_seconds, feedback_delay_seconds, length
        self.states = {}

    def before(self, user, at):
        last, events = self.states.get(user, (None, []))
        if last is None or at - last > self.gap:
            events = []
        if last is not None and at < last:
            raise ValueError('Session events must be chronological')
        self.states[user] = (at, events)
        return [(time, vector) for time, vector in events if time + self.delay < at][-self.length:]

    def after(self, user, at, vector):
        last, events = self.states[user]
        assert last == at
        events.append((at, vector))


@dataclass
class SessionData:
    root: Path
    arrays: dict
    metadata: dict

    def items(self, row):
        start, end = self.arrays['offsets'][row:row+2]
        return self.arrays['candidate_ids'][start:end], self.arrays['labels'][start:end]


def prepare_session(database, index_directory, directory, gap_seconds=1800, feedback_delay_seconds=300, length=10):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    index = TemporalIndex.load(index_directory)
    if index.index.d != 64:
        raise ValueError('Expected the frozen 64-dimensional article encoder')
    vectors = np.vstack([np.zeros((1,64),dtype=np.float32),index.index.reconstruct_n(0,index.index.ntotal)])
    ids = [''] + index.article_ids
    mapping = {article:i for i,article in enumerate(ids)}
    with connect_readonly(database) as con:
        count = con.execute('SELECT count(*) FROM impressions').fetchone()[0]
        entries = con.execute('SELECT count(*) FROM candidates').fetchone()[0]
        provenance = json.loads(con.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0])
        categories = dict(con.execute('SELECT news_id,category FROM articles'))
    arrays = {'vectors':vectors, 'first_seen':np.asarray([np.iinfo(np.int64).max]+index.first_seen.tolist(),dtype=np.int64),
              'article_ids':np.asarray(ids), 'categories':np.asarray(['']+[categories[a] for a in ids[1:]]),
              'long':np.zeros((count,64),np.float32),'events':np.zeros((count,length,65),np.float32),
              'event_mask':np.zeros((count,length),bool),'fit':np.zeros(count,bool),'times':np.zeros(count,np.int64),
              'candidate_ids':np.zeros(entries,np.int32),'labels':np.zeros(entries,np.uint8),'offsets':np.zeros(count+1,np.int64),
              'fit_exposures':np.zeros(len(ids),np.int64),'fit_clicks':np.zeros(len(ids),np.int64)}
    impression_ids, users, cursor, position = [], [], 0, 0
    sessions = SessionHistory(gap_seconds, feedback_delay_seconds, length)
    counts = Counter()
    fit_users = set()
    for at, group in itertools.groupby(impression_rows(database), key=lambda row:row[2]):
        updates = defaultdict(list)
        for iid,user,_,split,history,items in group:
            encoded = [mapping[a] for a in history[-20:]]
            arrays['long'][position] = normalized_mean(vectors[encoded])
            context = sessions.before(user,at)
            for j,(event_time,vector) in enumerate(context):
                arrays['events'][position,j,:64] = vector
                arrays['events'][position,j,64] = min(1.0,np.log1p(at-event_time)/np.log1p(gap_seconds))
                arrays['event_mask'][position,j] = True
            candidate_ids = np.asarray([mapping[a] for a,_ in items],dtype=np.int32)
            labels = np.asarray([y for _,y in items],dtype=np.uint8)
            end = cursor + len(items)
            arrays['candidate_ids'][cursor:end], arrays['labels'][cursor:end] = candidate_ids,labels
            arrays['offsets'][position] = cursor
            arrays['times'][position], arrays['fit'][position] = at,split=='fit'
            if split == 'fit':
                np.add.at(arrays['fit_exposures'],candidate_ids,1)
                np.add.at(arrays['fit_clicks'],candidate_ids,labels)
                fit_users.add(user)
            updates[user].extend(candidate_ids[labels==1].tolist())
            counts[split+'_impressions'] += 1
            counts[split+'_with_session_context'] += bool(context)
            counts[split+'_positive_events'] += int(labels.sum())
            counts[split+'_without_negatives'] += bool(labels.all())
            impression_ids.append(iid);users.append(user)
            cursor,position = end,position+1
        # All equal-time impressions see the same past; their clicks enter as one unordered group.
        for user,clicked in updates.items():
            if clicked:
                sessions.after(user,at,normalized_mean(vectors[clicked]))
    assert position==count and cursor==entries
    arrays['offsets'][-1]=cursor
    arrays['impression_ids']=np.asarray(impression_ids)
    arrays['users']=np.asarray(users)
    arrays['cold_user']=np.asarray([u not in fit_users for u in users])
    for name,array in arrays.items():
        np.save(directory/(name+'.npy'),array,allow_pickle=False)
    metadata={'format_version':1,'database_sha256':file_sha256(database),'input_sha256':provenance['input_sha256'],
              'cutoff':provenance['cutoff'],'index_manifest_sha256':file_sha256(Path(index_directory)/'manifest.json'),
              'counts':dict(counts),'configuration':{'gap_seconds':gap_seconds,'feedback_delay_seconds':feedback_delay_seconds,'length':length},
              'source_sha256':file_sha256(Path(__file__)),
              'session':'Past same-user impression-time groups; clicks available strictly after five minutes; gap >30min resets. Assumed feedback delay, not measured click timing.',
              'sha256':{p.name:file_sha256(p) for p in directory.glob('*.npy')}}
    write_json(directory/'manifest.json',metadata)
    return load_session(directory)


def load_session(directory):
    directory=Path(directory)
    metadata=json.loads((directory/'manifest.json').read_text())
    for name,sha in metadata['sha256'].items():
        if file_sha256(directory/name)!=sha:
            raise ValueError('Prepared session data hash mismatch: '+name)
    arrays={Path(name).stem:np.load(directory/name,mmap_mode='r',allow_pickle=False) for name in metadata['sha256']}
    return SessionData(directory,arrays,metadata)


def candidate_ages(data, rows, candidates):
    at=data.arrays['times'][rows]
    first=data.arrays['first_seen'][candidates]
    hours=np.maximum(0,(at[...,None]-first)/3600)
    return np.minimum(1,np.log1p(hours)/np.log1p(168)).astype(np.float32)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Prepare chronological delayed-session contexts using a frozen content index.')
    for name in ('database','index','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    result=prepare_session(args.database,args.index,args.output)
    print(json.dumps(result.metadata,indent=2))
