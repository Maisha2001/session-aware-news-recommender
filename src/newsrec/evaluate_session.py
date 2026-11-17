import argparse
from collections import Counter, defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .baseline import tie_key
from .data import file_sha256
from .metrics import MetricMeans,auc,ranking_metrics
from .rank_policy import rerank,slate_metrics
from .session_data import load_session
from .session_model import SessionRanker,tf
from .train_session import inputs_for
from .train_retrieval import write_json


SEGMENTS=('all','with_session','without_session','cold_user','empty_long_history','cold_positive_article')


def catalog_values(order,positives):
    if not positives:return {}
    result={f'recall@{k}':len(set(order[:k])&positives)/len(positives) for k in (10,100)}
    dcg=math.fsum(1/math.log2(i+2) for i,a in enumerate(order[:10]) if a in positives)
    ideal=math.fsum(1/math.log2(i+2) for i in range(min(10,len(positives))))
    result['ndcg@10']=dcg/ideal
    result['candidate_pool_recall']=len(set(order)&positives)/len(positives)
    return result


def exposure_bin(value):
    return 'unseen' if value==0 else 'tail_1_99' if value<100 else 'middle_100_999' if value<1000 else 'head_1000_plus'


def scores_for(model,data,rows,candidates):
    context=model.encode(np.asarray(data.arrays['long'][rows]),np.asarray(data.arrays['events'][rows]),np.asarray(data.arrays['event_mask'][rows]))
    lengths=np.asarray([len(ids) for ids in candidates])
    owners=np.repeat(np.arange(len(rows)),lengths)
    flat=np.concatenate(candidates)
    hours=np.maximum(0,(data.arrays['times'][rows][owners]-data.arrays['first_seen'][flat])/3600)
    ages=np.minimum(1,np.log1p(hours)/np.log1p(168)).astype(np.float32)
    scores=model.score(context,owners,flat,ages).numpy()
    if not np.isfinite(scores).all():raise RuntimeError('Nonfinite ranking scores')
    return np.split(scores,np.cumsum(lengths)[:-1])


def policy_order(data,ids,scores,at,preset):
    names=data.arrays['article_ids'][ids].tolist()
    ages=np.maximum(0,(at-data.arrays['first_seen'][ids])/3600)
    order=rerank(names,scores,data.arrays['vectors'][ids],data.arrays['categories'][ids],ages,preset)
    chosen=set(order)
    tail=sorted((i for i in range(len(ids)) if i not in chosen),key=lambda i:(-scores[i],tie_key(names[i])))
    return order+tail,slate_metrics(order,data.arrays['vectors'][ids],data.arrays['categories'][ids],ages)


def evaluate(data_directory,model_directory,retrieval_report,output_directory,batch_size=128):
    output_directory=Path(output_directory);output_directory.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();begin=time.perf_counter()
    data=load_session(data_directory)
    training=json.loads((Path(model_directory)/'training.json').read_text());config=training['config']
    assert config['data_manifest_sha256']==file_sha256(Path(data_directory)/'manifest.json')
    retrieval_report=Path(retrieval_report)
    previous=json.loads(retrieval_report.read_text())
    pred=previous['arms']['learned_content']['private_prediction_artifact']
    source=retrieval_report.parent/pred['file']
    assert file_sha256(source)==pred['sha256']
    models={}
    reload_count=min(256,len(data.arrays['fit']))
    reload_rows=np.arange(reload_count)
    reload_items=[data.items(row)[0] for row in reload_rows]
    for arm,enabled in [('long_only',False),('session',True)]:
        model=SessionRanker(data.arrays['vectors'],enabled)
        model(inputs_for(data,np.asarray([0]),np.asarray([[1]],np.int32)))
        weights=Path(model_directory)/f'{arm}-epoch-{config["epochs"]}.weights.h5'
        assert file_sha256(weights)==training['arms'][arm]['weights_sha256']
        model.load_weights(str(weights));models[arm]=model
        other=SessionRanker(data.arrays['vectors'],enabled)
        other(inputs_for(data,np.asarray([0]),np.asarray([[1]],np.int32)));other.load_weights(str(weights))
        expected=scores_for(model,data,reload_rows,reload_items)
        for left,right in zip(expected,scores_for(other,data,reload_rows,reload_items)):
            np.testing.assert_array_equal(left,right)
    logged_names=('long_only','session','session_fresh','session_balanced')
    logged={arm:{s:MetricMeans() for s in SEGMENTS} for arm in logged_names}
    catalog={preset:{s:MetricMeans() for s in SEGMENTS} for preset in ('relevance','fresh','balanced')}
    daily=defaultdict(lambda:{arm:MetricMeans() for arm in logged_names})
    exposure={arm:defaultdict(lambda:[0,0]) for arm in logged_names}
    article_attempts=Counter();article_hits={arm:Counter() for arm in logged_names}
    coverage={preset:set() for preset in catalog}
    counts=Counter();digest=hashlib.sha256();batch_times=[]
    ids=data.arrays['article_ids'];mapping={a:i for i,a in enumerate(ids)}
    popular=np.asarray(sorted(range(1,len(ids)),key=lambda i:(-data.arrays['fit_clicks'][i],tie_key(ids[i]))),np.int32)
    dev=np.flatnonzero(~data.arrays['fit'])
    path=output_directory/'predictions.jsonl.gz'
    with gzip.open(source,'rt',encoding='utf-8') as prior,gzip.open(path,'xt',encoding='utf-8') as predictions:
        for offset in range(0,len(dev),batch_size):
            rows=dev[offset:offset+batch_size]
            candidates=[data.items(row)[0] for row in rows]
            pools=[]
            for row in rows:
                record=json.loads(next(prior))
                assert record[0]==str(data.arrays['impression_ids'][row])
                at=data.arrays['times'][row]
                popular_candidates=popular[:100]
                if not np.all(data.arrays['first_seen'][popular_candidates]<at):
                    popular_candidates=popular[data.arrays['first_seen'][popular]<at][:100]
                pool=list(dict.fromkeys([mapping[a] for a in record[2]]+popular_candidates.tolist()))
                assert len(set(pool))==len(pool) and all(data.arrays['first_seen'][a]<at for a in pool)
                pools.append(np.asarray(pool,np.int32))
            start=time.perf_counter()
            model_scores={arm:scores_for(model,data,rows,candidates) for arm,model in models.items()}
            pool_scores=scores_for(models['session'],data,rows,pools)
            batch_times.append(time.perf_counter()-start)
            for j,row in enumerate(rows):
                at=int(data.arrays['times'][row]);candidate_ids,labels=data.items(row);positives=set(candidate_ids[labels==1].tolist())
                present=bool(data.arrays['event_mask'][row].any())
                segments=['all','with_session' if present else 'without_session']
                if data.arrays['cold_user'][row]:segments.append('cold_user')
                if not np.any(data.arrays['long'][row]):segments.append('empty_long_history')
                if any(data.arrays['fit_exposures'][a]==0 for a in positives):segments.append('cold_positive_article')
                for segment in segments:counts[segment+'_impressions']+=1
                counts['candidate_entries']+=len(candidate_ids);counts['positive_events']+=len(positives)
                counts['zero_positive_impressions']+=not positives
                counts['single_class_impressions']+=len(positives) in (0,len(candidate_ids))
                counts['unavailable_positive_events']+=sum(int(data.arrays['first_seen'][a]>=at) for a in positives)
                article_attempts.update(positives)
                record={'iid':str(data.arrays['impression_ids'][row]),'logged':{},'catalog':{}}
                day=datetime.fromtimestamp(at,timezone.utc).date().isoformat()
                for arm in logged_names:
                    base='long_only' if arm=='long_only' else 'session'
                    preset=arm.split('_',1)[1] if arm.startswith('session_') else 'relevance'
                    scores=model_scores[base][j]
                    order,slate=policy_order(data,candidate_ids,scores,at,preset)
                    values=ranking_metrics(labels[order].tolist());values.pop('positive_count')
                    if arm in ('long_only','session'):values['auc']=auc(labels.tolist(),scores.tolist())
                    values.update(slate)
                    for segment in segments:logged[arm][segment].add(values)
                    daily[day][arm].add(values)
                    selected=set(candidate_ids[order[:10]].tolist())
                    article_hits[arm].update(positives&selected)
                    for article in positives:
                        bucket=exposure_bin(int(data.arrays['fit_exposures'][article]))
                        exposure[arm][bucket][0]+=int(article in selected);exposure[arm][bucket][1]+=1
                    record['logged'][arm]={'ids':ids[candidate_ids[order]].tolist()}
                    if arm in ('long_only','session'):record['logged'][arm]['scores']=[float(scores[i]) for i in order]
                for preset in catalog:
                    pool=pools[j];order,slate=policy_order(data,pool,pool_scores[j],at,preset)
                    ranked=pool[order].tolist();values=catalog_values(ranked,positives);values.update(slate)
                    for segment in segments:catalog[preset][segment].add(values)
                    coverage[preset].update(ranked[:10])
                    record['catalog'][preset]=ids[ranked].tolist()
                    if preset=='relevance':record['catalog_relevance_scores']=[float(pool_scores[j][i]) for i in order]
                encoded=json.dumps(record,separators=(',',':'))+'\n'
                digest.update(encoded.encode());predictions.write(encoded)
        assert prior.readline()==''
    result={'format_version':1,'data':{k:v for k,v in data.metadata.items() if k!='sha256'},'training_config':config,
        'logged_ranking':{arm:{s:m.report() for s,m in groups.items()} for arm,groups in logged.items()},
        'catalog_ranking':{arm:{s:m.report() for s,m in groups.items()} for arm,groups in catalog.items()},
        'daily':{day:{arm:m.report() for arm,m in groups.items()} for day,groups in daily.items()},
        'exposure_analysis':{arm:{'positive_recall@10':{bucket:{'hits':v[0],'events':v[1],'recall':v[0]/v[1]} for bucket,v in groups.items()},
                                  'macro_clicked_article_recall@10':math.fsum(article_hits[arm][a]/n for a,n in article_attempts.items())/len(article_attempts) if article_attempts else None,
                                  'distinct_clicked_articles':len(article_attempts)} for arm,groups in exposure.items()},
        'catalog_coverage':{arm:{'distinct_top10_articles':len(values),'observed_catalog_articles':len(ids)-1,'fraction':len(values)/(len(ids)-1)} for arm,values in coverage.items()},
        'counts':dict(counts),'private_prediction_artifact':{'file':path.name,'sha256':file_sha256(path),'uncompressed_sha256':digest.hexdigest()},
        'source_retrieval_report_sha256':file_sha256(retrieval_report),'source_retrieval_predictions_sha256':pred['sha256'],
        'audit':{'bitwise_model_reload_impressions_per_arm':reload_count,'all_catalog_candidates_unique_and_strictly_earlier':True},
        'protocol':{'candidate_pool':'Union of frozen learned retrieval100 and frozen popularity100, deduplicated, strictly earlier observations. No positive-injection oracle.',
                    'feedback':'Frozen weights; earlier observed clicks can enter delayed session context after5min. Equal-time and current outcomes excluded;30min inactivity reset.',
                    'freshness':'First-observation age proxy; publication/edit timestamps unavailable. Fixed exp(-age_hours/24) decay: 24-hour scale, approximately16.64-hour half-life.',
                    'controls':'Relevance rank scale; fresh weight.15; balanced freshness.10 and redundancy.15, half max content similarity/half category share. Top10 modified, remaining model order retained.',
                    'exposure':'Only logged nonclicks used as training negatives. Missing catalog labels are unknown; catalog metrics count known logged positives only. No propensities or randomized exposure: no unbiased IPS/causal claim.'},
        'limitations':['Five-minute feedback availability and30-minute sessions are simulation assumptions because click timestamps are unavailable.',
                       'Only a minority of impressions have prior same-session context; all other impressions remain included.',
                       'One fixed seed and three epochs per arm; no significance or development-selected checkpoint.',
                       'Article-macro and exposure-frequency strata are descriptive sensitivity analyses, not causal correction.',
                       'Uncalibrated sampled-softmax logits are ranking scores, not click probabilities. Unknown full-catalog relevance prevents full-catalog AUC.',
                       'Frozen candidate coverage limits all downstream ranking; no live engagement claim.'],
        'runtime':{'wall_seconds':time.perf_counter()-begin,'scoring_seconds':sum(batch_times),'batch_size':batch_size,'scoring_batch_p95_seconds':float(np.percentile(batch_times,95))},
        'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'final_holdout_opened':False,
        'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ('evaluate_session.py','session_model.py','session_data.py','rank_policy.py')}}
    write_json(output_directory/'report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Evaluate fixed session ranking and freshness/diversity policies without opening the final holdout.')
    for name in ('data','model','retrieval-report','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    result=evaluate(args.data,args.model,args.retrieval_report,args.output)
    print(json.dumps({'completed_at_utc':result['completed_at_utc'],'counts':result['counts'],'runtime':result['runtime']}))
