from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import faiss
import numpy as np

from .baseline import tie_key
from .data import file_sha256
from .evaluate_retrieval import exhaustive_check
from .evaluate_session import catalog_values, exposure_bin, policy_order, scores_for
from .metrics import MetricMeans, auc, ranking_metrics
from .retrieval_index import TemporalIndex
from .retrieval_model import ContentRetriever, tf
from .session_model import SessionRanker
from .train_retrieval import write_json


LOGGED = ('popularity','content','long_only','session','session_fresh','session_balanced')
CATALOG = ('popularity','content','session','session_fresh','session_balanced')
SEGMENTS = ('all','with_session','without_session','cold_user','warm_user','empty_long_history',
            'cold_positive_article','unseen_content_positive')


def load_models(data, paths):
    content = ContentRetriever(data.arrays['tokens'])
    content(np.zeros((1,20),np.int32)); content.load_weights(str(paths['content']))
    models = {}
    for arm, enabled in (('long_only',False),('session',True)):
        model = SessionRanker(data.arrays['vectors'],enabled)
        model({'long':np.zeros((1,64),np.float32),'events':np.zeros((1,10,65),np.float32),
               'event_mask':np.zeros((1,10),bool),'candidate_ids':np.ones((1,1),np.int32),'ages':np.zeros((1,1),np.float32)})
        model.load_weights(str(paths[arm])); models[arm] = model
    return content, models


def queries_for(content, data, rows):
    histories = data.arrays['histories'][rows]
    raw = data.arrays['vectors'][histories]
    count = (histories != 0).sum(axis=1)[:,None]
    mean = raw.sum(axis=1) / np.maximum(count,1)
    mean = np.where(count > 0,mean,content.empty_query.numpy()[None,:]).astype(np.float32)
    return tf.math.l2_normalize(content.query_projection(mean),axis=-1).numpy()


def predict_final(data, paths, directory, batch_size=128, audit_queries=128):
    directory = Path(directory); directory.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(); begin = time.perf_counter()
    content, models = load_models(data,paths)
    ids = data.arrays['article_ids']; vectors = data.arrays['vectors']
    index = TemporalIndex(vectors[1:],data.arrays['first_seen'][1:],ids[1:].tolist())
    index.save(directory/'index'); index = TemporalIndex.load(directory/'index')
    faiss.omp_set_num_threads(4)
    count = len(data.arrays['times'])
    audit_order = sorted(range(count),key=lambda r:hashlib.sha256(('newsrec-final-audit-v1:'+str(data.arrays['impression_ids'][r])).encode()).hexdigest())
    selected = set(audit_order[:audit_queries])
    reload_rows = np.asarray(sorted(audit_order[:256]),np.int32)
    reload_items = [data.arrays['candidate_ids'][slice(*data.arrays['offsets'][r:r+2])] for r in reload_rows]
    other_content, other_models = load_models(data,paths)
    np.testing.assert_array_equal(queries_for(content,data,reload_rows),queries_for(other_content,data,reload_rows))
    for arm in models:
        for left,right in zip(scores_for(models[arm],data,reload_rows,reload_items),scores_for(other_models[arm],data,reload_rows,reload_items)):
            np.testing.assert_array_equal(left,right)
    del other_content, other_models
    popular = np.asarray(sorted(range(1,len(ids)),key=lambda i:(-data.arrays['fit_clicks'][i],tie_key(ids[i]))),np.int32)
    query_vectors = np.zeros((count,64),np.float32)
    digest = hashlib.sha256(); timing = []; exact = checked = 0; maximum = 0.0
    with gzip.open(directory/'predictions.jsonl.gz','xt',encoding='utf-8',newline='\n') as stream:
        for start in range(0,count,batch_size):
            before = time.perf_counter(); rows = np.arange(start,min(count,start+batch_size))
            queries = queries_for(content,data,rows); query_vectors[rows] = queries
            found = index.search(queries,data.arrays['times'][rows],100)
            candidates = [data.arrays['candidate_ids'][slice(*data.arrays['offsets'][r:r+2])] for r in rows]
            pop, pools = [], []
            for r, hits in zip(rows,found):
                at = data.arrays['times'][r]
                top = popular[data.arrays['first_seen'][popular]<at][:100]
                pop.append(top)
                pools.append(np.asarray(list(dict.fromkeys([i for i,_ in hits]+top.tolist())),np.int32))
            logged_scores = {arm:scores_for(model,data,rows,candidates) for arm,model in models.items()}
            pool_scores = scores_for(models['session'],data,rows,pools)
            for j,r in enumerate(rows):
                at = int(data.arrays['times'][r]); items = candidates[j]
                scores = {'popularity':data.arrays['fit_clicks'][items].astype(float),
                          'content':vectors[items] @ queries[j],
                          'long_only':logged_scores['long_only'][j],'session':logged_scores['session'][j]}
                record = {'iid':str(data.arrays['impression_ids'][r]),'logged':{},'catalog':{},
                          'session_groups':int(data.arrays['event_mask'][r].sum())}
                for arm in LOGGED:
                    base = 'session' if arm.startswith('session_') else arm
                    preset = arm.split('_',1)[1] if arm.startswith('session_') else 'relevance'
                    order,_ = policy_order(data,items,scores[base],at,preset)
                    result = {'ids':ids[items[order]].tolist()}
                    if arm in scores: result['scores'] = [float(scores[base][i]) for i in order]
                    record['logged'][arm] = result
                record['catalog']['popularity'] = ids[pop[j]].tolist()
                record['catalog']['content'] = [str(ids[i]) for i,_ in found[j]]
                for arm,preset in (('session','relevance'),('session_fresh','fresh'),('session_balanced','balanced')):
                    order,_ = policy_order(data,pools[j],pool_scores[j],at,preset)
                    record['catalog'][arm] = ids[pools[j][order]].tolist()
                    if arm=='session': record['catalog_session_scores'] = [float(pool_scores[j][i]) for i in order]
                for order in record['catalog'].values():
                    assert len(order)==len(set(order))
                assert all(data.arrays['first_seen'][pools[j]]<at)
                if r in selected:
                    match,error = exhaustive_check(vectors[1:],queries[j],at,index,found[j])
                    checked += 1; exact += match; maximum = max(maximum,error)
                encoded = json.dumps(record,separators=(',',':'),allow_nan=False)+'\n'
                stream.write(encoded); digest.update(encoded.encode())
            timing.append(time.perf_counter()-before)
    assert checked == len(selected)
    np.save(directory/'queries.npy',query_vectors,allow_pickle=False)
    manifest = {'status':'predictions_frozen','started_at_utc':started,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                'impressions':count,'predictions_sha256':file_sha256(directory/'predictions.jsonl.gz'),
                'uncompressed_sha256':digest.hexdigest(),'queries_sha256':file_sha256(directory/'queries.npy'),
                'prepared_manifest_sha256':file_sha256(data.root/'manifest.json'),
                'model_sha256':{name:file_sha256(Path(path)) for name,path in paths.items()},
                'audit':{'exhaustive_float64_queries':checked,'exact_order_matches':exact,'maximum_score_difference':maximum,
                         'score_tolerance':1e-5,'boundary_tie_tolerance':1e-6,'reload_impressions_per_arm':len(reload_rows),
                         'reload_predictions_bitwise_equal':True,'all_catalog_entries_unique_and_strictly_earlier':True},
                'runtime':{'wall_seconds':time.perf_counter()-begin,'batch_size':batch_size,
                           'prediction_batch_p95_seconds':float(np.percentile(timing,95)),
                           'timing_scope':'Encoding, exact retrieval, scoring, policies, inline exhaustive audit and prediction serialization. Model/index loading included in wall time; metric aggregation and external audit excluded.'},
                'labels':'Prediction code never accesses current labels. Prepared contexts already include only earlier eligible feedback under the frozen causal replay; this is not an entirely label-free test simulation.'}
    write_json(directory/'manifest.json',manifest)
    return manifest


def final_metrics(data, predictions, output):
    if Path(output).exists():raise FileExistsError('Refusing to overwrite final scores')
    predictions=Path(predictions); manifest=json.loads((predictions/'manifest.json').read_text())
    path=predictions/'predictions.jsonl.gz'
    assert file_sha256(path)==manifest['predictions_sha256']
    started=datetime.now(timezone.utc).isoformat(); begin=time.perf_counter()
    logged={arm:{s:MetricMeans() for s in SEGMENTS} for arm in LOGGED}
    catalog={arm:{s:MetricMeans() for s in SEGMENTS} for arm in CATALOG}
    daily=defaultdict(lambda:{arm:MetricMeans() for arm in LOGGED})
    counts=Counter(); exposure={arm:defaultdict(lambda:[0,0]) for arm in LOGGED}
    attempts=Counter(); hits={arm:Counter() for arm in LOGGED}; coverage={arm:set() for arm in CATALOG}
    ids=data.arrays['article_ids']; mapping={a:i for i,a in enumerate(ids)}
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for r,line in enumerate(stream):
            record=json.loads(line); assert record['iid']==str(data.arrays['impression_ids'][r])
            items,labels=data.items(r); positive=set(items[labels==1].tolist()); at=int(data.arrays['times'][r])
            present=bool(data.arrays['event_mask'][r].any())
            segments=['all','with_session' if present else 'without_session','cold_user' if data.arrays['cold_user'][r] else 'warm_user']
            if not np.any(data.arrays['histories'][r]):segments.append('empty_long_history')
            if any(data.arrays['fit_exposures'][a]==0 for a in positive):segments.append('cold_positive_article')
            if any(not data.arrays['fit_content'][a] for a in positive):segments.append('unseen_content_positive')
            counts['impressions']+=1; counts['candidate_entries']+=len(items); counts['positive_events']+=len(positive)
            counts['zero_positive_impressions']+=not positive;counts['single_class_impressions']+=len(positive) in (0,len(items))
            counts['unavailable_positive_events']+=sum(data.arrays['first_seen'][a]>=at for a in positive)
            for s in segments:counts[s+'_impressions']+=1
            attempts.update(positive)
            for arm in LOGGED:
                ranked=record['logged'][arm]; order=[mapping[a] for a in ranked['ids']]
                assert set(order)==set(items.tolist()) and len(order)==len(items)
                binary=[int(a in positive) for a in order]
                values=ranking_metrics(binary);values.pop('positive_count')
                if 'scores' in ranked:values['auc']=auc(binary,ranked['scores'])
                from .rank_policy import slate_metrics
                ages=np.maximum(0,(at-data.arrays['first_seen'][order])/3600)
                values.update(slate_metrics(list(range(min(10,len(order)))),data.arrays['vectors'][order],data.arrays['categories'][order],ages))
                for s in segments:logged[arm][s].add(values)
                day=datetime.fromtimestamp(at,timezone.utc).date().isoformat();daily[day][arm].add(values)
                selected=set(order[:10]);hits[arm].update(positive&selected)
                for article in positive:
                    bucket=exposure_bin(int(data.arrays['fit_exposures'][article]))
                    exposure[arm][bucket][0]+=int(article in selected);exposure[arm][bucket][1]+=1
            for arm in CATALOG:
                order=[mapping[a] for a in record['catalog'][arm]]
                assert len(order)==len(set(order)) and all(data.arrays['first_seen'][a]<at for a in order)
                values=catalog_values(order,positive)
                ages=np.maximum(0,(at-data.arrays['first_seen'][order])/3600)
                values.update(slate_metrics(list(range(min(10,len(order)))),data.arrays['vectors'][order],data.arrays['categories'][order],ages))
                for s in segments:catalog[arm][s].add(values)
                coverage[arm].update(order[:10])
    assert counts['impressions']==manifest['impressions']
    result={'format_version':1,'evaluation_split':'official MIND-small dev archive, reserved as final chronological holdout',
            'counts':{k:int(v) for k,v in counts.items()},'data':{k:v for k,v in data.metadata.items() if k!='sha256'},
            'logged_ranking':{a:{s:m.report() for s,m in groups.items()} for a,groups in logged.items()},
            'catalog_ranking':{a:{s:m.report() for s,m in groups.items()} for a,groups in catalog.items()},
            'daily':{d:{a:m.report() for a,m in groups.items()} for d,groups in daily.items()},
            'exposure_analysis':{a:{'positive_recall@10':{b:{'hits':v[0],'events':v[1],'recall':v[0]/v[1]} for b,v in groups.items()},
                'macro_clicked_article_recall@10':math.fsum(hits[a][article]/n for article,n in attempts.items())/len(attempts) if attempts else None,
                'distinct_clicked_articles':len(attempts)} for a,groups in exposure.items()},
            'catalog_coverage':{a:{'distinct_top10_articles':len(v),'observed_catalog_articles':len(ids)-1,'fraction':len(v)/(len(ids)-1)} for a,v in coverage.items()},
            'prediction_manifest':manifest,'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'metric_wall_seconds':time.perf_counter()-begin,
            'comparison_note':'All final impressions retained. Logged AUC/nDCG rank the observed slate; catalog recall counts only known logged positives across all earlier observed articles. Session retrieval reranks the union of content100 and popularity100; the content and popularity rows use their own100. These different candidate pools limit direct attribution. Exposure bias prevents causal engagement claims.'}
    result['comparison_rows']=[{'name':label,'logged_auc':result['logged_ranking'][arm]['all']['auc']['mean'],
        'logged_ndcg10':result['logged_ranking'][arm]['all']['ndcg@10']['mean'],'catalog_recall100':result['catalog_ranking'][arm]['all']['recall@100']['mean']}
        for arm,label in [('popularity','Popularity'),('content','Content retrieval'),('session','Session ranking')]]
    write_json(Path(output),result)
    return result


if __name__=='__main__':
    import argparse
    from .heldout_data import prepare_final
    from .session_data import load_session
    parser=argparse.ArgumentParser(description='Prepare, predict and score a reserved chronological holdout with frozen model files. Each stage creates new evidence only.')
    parser.add_argument('stage',choices=['prepare','predict','metrics'])
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--models',type=Path,help='JSON mapping content, long_only and session to local weight paths')
    for name in ('database','archive','original-prepared'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    paths={k:Path(v) for k,v in json.loads(args.models.read_text()).items()} if args.models else {}
    if args.stage=='prepare':
        if not all((args.database,args.archive,args.original_prepared,paths.get('content'))):parser.error('prepare requires database, archive, original-prepared and models')
        result=prepare_final(args.database,args.archive,args.original_prepared,paths['content'],args.run).metadata
    elif args.stage=='predict':
        if set(paths)!={'content','long_only','session'}:parser.error('predict requires all three frozen model paths')
        result=predict_final(load_session(args.run/'prepared'),paths,args.run/'predictions')
    else:result=final_metrics(load_session(args.run/'prepared'),args.run/'predictions',args.run/'report.json')
    print(json.dumps({'status':'complete','stage':args.stage,'counts':result.get('counts'),'completed_at_utc':datetime.now(timezone.utc).isoformat()}))
