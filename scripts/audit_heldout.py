"""Audit the final causal replay and aggregate scores independently of newsrec."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from audit_baseline import key, score, sha256
from audit_session import load_arrays, mean_vector, policy_check, slate, truth


def audit(training_behaviors, run, output):
    began=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    manifest,a=load_arrays(run/'prepared'); report=json.loads((run/'report.json').read_text())
    training,first,exposures,clicks,users=truth(training_behaviors,datetime(2019,11,13))
    raw,final_first,_,_,_=truth(run/'input/behaviors.tsv',datetime(2019,11,13))
    rows=[(at,'final:'+iid,user,history,items) for at,iid,user,history,items in raw]
    assert rows[0][0]>training[-1][0]
    assert sha256(run/'input/behaviors.tsv')==manifest['input_sha256']['behaviors.tsv']
    assert sha256(run/'input/news.tsv')==manifest['input_sha256']['news.tsv']
    for article,at in final_first.items():first[article]=min(first.get(article,at),at)
    fit_content={article for at,_,_,_,items in training if at<datetime(2019,11,13) for article,_ in items}
    for at,_,_,history,items in training:
        if at<datetime(2019,11,13) and any(y for _,y in items):fit_content.update(history[-20:])
    mapping={article:i for i,article in enumerate(a['article_ids'])}
    assert set(first)==set(mapping)-{''}
    for article,at in first.items():
        i=mapping[article]
        assert a['first_seen'][i]==int((at-datetime(1970,1,1)).total_seconds())
        assert a['fit_clicks'][i]==clicks[article] and a['fit_exposures'][i]==exposures[article]
        assert bool(a['fit_content'][i])==(article in fit_content)
    by_user=defaultdict(list);position={row[1]:i for i,row in enumerate(rows)};feature_error=0.0
    for row in training+rows:by_user[row[2]].append(row)
    for i,(at,iid,user,history,items) in enumerate(rows):
        assert str(a['impression_ids'][i])==iid and str(a['users'][i])==user
        assert a['times'][i]==int((at-datetime(1970,1,1)).total_seconds())
        assert bool(a['cold_user'][i])==(user not in users)
        start,end=a['offsets'][i:i+2]
        assert a['candidate_ids'][start:end].tolist()==[mapping[x] for x,_ in items]
        assert a['labels'][start:end].tolist()==[y for _,y in items]
        h=[mapping[x] for x in history[-20:]]
        assert a['histories'][i].tolist()==h+[0]*(20-len(h))
        error=float(np.max(np.abs(a['long'][i]-mean_vector(a['vectors'][h]))));feature_error=max(feature_error,error);assert error<=1e-6
    context_count=0
    for user,user_rows in by_user.items():
        earlier=[];last=None
        for at,group in itertools.groupby(user_rows,key=lambda row:row[0]):
            current=list(group)
            if last is None or (at-last).total_seconds()>1800:earlier=[]
            eligible=[(when,vec) for when,vec in earlier if (at-when).total_seconds()>300][-10:]
            expected=np.zeros((10,65));mask=np.zeros(10,bool)
            for j,(when,vec) in enumerate(eligible):
                expected[j,:64]=vec;expected[j,64]=min(1,math.log1p((at-when).total_seconds())/math.log1p(1800));mask[j]=True
            for _,iid,_,_,_ in current:
                if iid not in position:continue
                i=position[iid];np.testing.assert_array_equal(a['event_mask'][i],mask)
                error=float(np.max(np.abs(a['events'][i]-expected)));feature_error=max(feature_error,error);assert error<=1e-6
                context_count+=bool(eligible)
            clicked=[mapping[article] for row in current for article,y in row[4] if y]
            if clicked:earlier.append((at,mean_vector(a['vectors'][clicked])))
            last=at
    pm=json.loads((run/'predictions/manifest.json').read_text())
    path=run/'predictions/predictions.jsonl.gz';assert sha256(path)==pm['predictions_sha256']
    assert sha256(run/'prepared/manifest.json')==pm['prepared_manifest_sha256']
    popular=sorted(first,key=lambda article:(-clicks[article],key(article)))
    arrival_times=sorted(first.values())
    sample=set(sorted([row[1] for row in rows],key=lambda iid:hashlib.sha256(('newsrec-final-audit-v1:'+iid).encode()).hexdigest())[:128])
    values=defaultdict(lambda:defaultdict(list));counts=Counter();attempts=Counter();hits=defaultdict(Counter)
    exposure=defaultdict(lambda:defaultdict(lambda:[0,0]));coverage=defaultdict(set)
    digest=hashlib.sha256();policy_maximum=0.0;policy_queries=0
    def add(group,result):
        for name,value in result.items():values[group][name].append(value)
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for at,iid,user,history,items in rows:
            line=next(stream);digest.update(line.encode());prediction=json.loads(line);assert prediction['iid']==iid
            positives={article for article,y in items if y};labels=dict(items);r=position[iid]
            assert prediction['session_groups']==int(a['event_mask'][r].sum())
            segments=['all','with_session' if a['event_mask'][r].any() else 'without_session','cold_user' if user not in users else 'warm_user']
            if not history:segments.append('empty_long_history')
            if any(exposures[x]==0 for x in positives):segments.append('cold_positive_article')
            if any(x not in fit_content for x in positives):segments.append('unseen_content_positive')
            for segment in segments:counts[segment+'_impressions']+=1
            counts['impressions']+=1;counts['candidate_entries']+=len(items);counts['positive_events']+=len(positives)
            counts['zero_positive_impressions']+=not positives;counts['single_class_impressions']+=len(positives) in (0,len(items))
            counts['unavailable_positive_events']+=sum(first[x]>=at for x in positives);attempts.update(positives)
            for arm,ranked in prediction['logged'].items():
                order=ranked['ids'];assert len(order)==len(set(order))==len(items) and set(order)==set(labels)
                scores=ranked.get('scores',list(range(len(order),0,-1)))
                if 'scores' in ranked:assert order==[x for _,_,x in sorted((-s,key(x),x) for x,s in zip(order,scores))]
                if arm=='popularity':assert scores==[float(clicks[x]) for x in order]
                metrics=score([labels[x] for x in order],scores)
                if 'scores' not in ranked:metrics.pop('auc',None)
                metrics.update(slate(order,at,first,a['vectors'],a['categories'],mapping))
                for segment in segments:add(('logged_ranking',arm,segment),metrics)
                add(('daily',str(at.date()),arm),metrics)
                selected=set(order[:10]);hits[arm].update(positives&selected)
                for article in positives:
                    n=exposures[article];bucket='unseen' if n==0 else 'tail_1_99' if n<100 else 'middle_100_999' if n<1000 else 'head_1000_plus'
                    exposure[arm][bucket][0]+=article in selected;exposure[arm][bucket][1]+=1
            expected_pop=list(itertools.islice((x for x in popular if first[x]<at),100))
            assert prediction['catalog']['popularity']==expected_pop
            pool=set(expected_pop)|set(prediction['catalog']['content'])
            for arm,order in prediction['catalog'].items():
                assert len(order)==len(set(order)) and all(first[x]<at for x in order)
                if arm.startswith('session'):assert set(order)==pool
                else:assert len(order)==min(100,bisect_left(arrival_times,at))
                metrics={}
                if positives:
                    ranks=[i+1 for i,x in enumerate(order) if x in positives]
                    metrics={f'recall@{k}':sum(r<=k for r in ranks)/len(positives) for k in (10,100)}
                    metrics['candidate_pool_recall']=len(ranks)/len(positives)
                    metrics['ndcg@10']=math.fsum(1/math.log2(r+1) for r in ranks if r<=10)/math.fsum(1/math.log2(r+1) for r in range(1,min(10,len(positives))+1))
                metrics.update(slate(order,at,first,a['vectors'],a['categories'],mapping))
                for segment in segments:add(('catalog_ranking',arm,segment),metrics)
                coverage[arm].update(order[:10])
            base=prediction['catalog']['session'];scores=prediction['catalog_session_scores']
            assert base==[x for _,_,x in sorted((-s,key(x),x) for x,s in zip(base,scores))]
            if iid in sample:
                for preset in ('fresh','balanced'):
                    for base,actual in [(prediction['logged']['session']['ids'],prediction['logged']['session_'+preset]['ids']),
                                        (prediction['catalog']['session'],prediction['catalog']['session_'+preset])]:
                        policy_maximum=max(policy_maximum,policy_check(base,actual,preset,at,first,a['vectors'],a['categories'],mapping))
                policy_queries+=1
        assert stream.readline()==''
    assert digest.hexdigest()==pm['uncompressed_sha256'] and dict(counts)==report['counts']
    maximum=0.0;checked=0
    for group,metrics in values.items():
        expected=report
        for part in group:expected=expected[part]
        assert set(expected)==set(metrics)
        for name,samples in metrics.items():
            assert len(samples)==expected[name]['impressions']
            error=abs(math.fsum(samples)/len(samples)-expected[name]['mean'])
            tolerance=1e-7 if name=='intra_list_cosine_distance@10' else 1e-8 if name=='mean_first_seen_age_hours@10' else 1e-10
            assert error<=tolerance,(group,name,error);maximum=max(maximum,error);checked+=1
    expected_groups={(root,arm,s) for root in ('logged_ranking','catalog_ranking') for arm,groups in report[root].items() for s,v in groups.items() if v}
    expected_groups|={('daily',day,arm) for day,groups in report['daily'].items() for arm,v in groups.items() if v}
    assert set(values)==expected_groups
    for arm,buckets in exposure.items():
        expected=report['exposure_analysis'][arm];assert set(buckets)==set(expected['positive_recall@10'])
        for bucket,(hit,n) in buckets.items():assert expected['positive_recall@10'][bucket]=={'hits':hit,'events':n,'recall':hit/n}
        assert expected['distinct_clicked_articles']==len(attempts)
        assert abs(expected['macro_clicked_article_recall@10']-math.fsum(hits[arm][x]/n for x,n in attempts.items())/len(attempts))<1e-10
    for arm,articles in coverage.items():assert len(articles)==report['catalog_coverage'][arm]['distinct_top10_articles']
    result={'status':'passed','impressions_checked':len(rows),'context_impressions_checked':len(rows),'with_session_context':context_count,
            'maximum_feature_difference':feature_error,'metric_values_checked':checked,'metric_groups_checked':len(values),
            'maximum_aggregate_difference':maximum,'policy_queries_checked':policy_queries,'policy_decisions_checked':policy_queries*4,
            'maximum_policy_utility_difference':policy_maximum,
            'tolerances':{'context':1e-6,'ranking':1e-10,'age':1e-8,'cosine_diversity':1e-7,'policy_utility':1e-6},
            'prediction_sha256':sha256(path),'uncompressed_prediction_sha256':digest.hexdigest(),'report_sha256':sha256(run/'report.json'),
            'method':'Independent raw training/final TSV parsing; per-user float64 delayed contexts; all labels, candidates, histories, fit counts, availability, metric groups, exposure and coverage; sampled brute-force controls. No newsrec imports.',
            'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-began,
            'source_sha256':{p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('audit_session.py'),Path(__file__).with_name('audit_baseline.py'))}}
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training-behaviors','run','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.training_behaviors,args.run,args.output),indent=2))
