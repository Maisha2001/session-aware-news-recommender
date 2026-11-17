"""Independently verify delayed contexts, frozen ranking metrics and slate controls."""

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from audit_baseline import key,score,sha256


def truth(path,cutoff):
    rows=[];first={};exposures=Counter();clicks=Counter();users=set()
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            iid,user,stamp,history,raw=line.rstrip('\r\n').split('\t')
            at=datetime.strptime(stamp,'%m/%d/%Y %I:%M:%S %p')
            items=[(a,int(y)) for a,y in (item.rsplit('-',1) for item in raw.split())]
            history=history.split()
            for article in set(history)|{a for a,_ in items}:first[article]=min(first.get(article,at),at)
            if at<cutoff:
                users.add(user);exposures.update(a for a,_ in items)
                for a,y in items:clicks[a]+=y
            rows.append((at,iid,user,history,items))
    return sorted(rows),first,exposures,clicks,users


def mean_vector(vectors):
    if not len(vectors):return np.zeros(64)
    result=np.asarray(vectors,dtype=np.float64).sum(axis=0)/len(vectors)
    return result/max(float(np.sqrt(np.square(result).sum())),1e-12)


def load_arrays(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():assert sha256(directory/name)==expected
    return manifest,{Path(name).stem:np.load(directory/name,mmap_mode='r',allow_pickle=False) for name in manifest['sha256']}


def context_audit(behaviors,directory,index_directory,output):
    began=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    manifest,a=load_arrays(directory)
    assert sha256(behaviors)==manifest['input_sha256']['behaviors.tsv']
    rows,first,exposures,clicks,fit_users=truth(behaviors,datetime.fromisoformat(manifest['cutoff']))
    import faiss
    im=json.loads((index_directory/'manifest.json').read_text())
    assert sha256(index_directory/'manifest.json')==manifest['index_manifest_sha256']
    for name,expected in im['sha256'].items():assert sha256(index_directory/name)==expected
    frozen=faiss.read_index(str(index_directory/'vectors.faiss'))
    np.testing.assert_array_equal(a['vectors'][1:],frozen.reconstruct_n(0,frozen.ntotal))
    article_ids=a['article_ids'].tolist();mapping={article:i for i,article in enumerate(article_ids)}
    assert set(first)==set(article_ids[1:])
    for article in first:
        i=mapping[article]
        assert a['first_seen'][i]==int((first[article]-datetime(1970,1,1)).total_seconds())
        assert a['fit_exposures'][i]==exposures[article] and a['fit_clicks'][i]==clicks[article]
    by_user=defaultdict(list);row_map={}
    maximum=0.0
    for i,row in enumerate(rows):
        at,iid,user,history,items=row;row_map[iid]=i;by_user[user].append(row)
        assert str(a['impression_ids'][i])==iid and str(a['users'][i])==user
        assert bool(a['fit'][i])==(at<datetime.fromisoformat(manifest['cutoff']))
        assert bool(a['cold_user'][i])==(user not in fit_users)
        start,end=a['offsets'][i:i+2]
        assert a['candidate_ids'][start:end].tolist()==[mapping[x] for x,_ in items]
        assert a['labels'][start:end].tolist()==[y for _,y in items]
        expected=mean_vector(a['vectors'][[mapping[x] for x in history[-20:]]])
        error=float(np.max(np.abs(a['long'][i]-expected)));maximum=max(maximum,error);assert error<=1e-6
    config=manifest['configuration'];gap=config['gap_seconds'];delay=config['feedback_delay_seconds'];length=config['length']
    active=0
    for user,user_rows in by_user.items():
        earlier=[];last=None
        for at,group in itertools.groupby(user_rows,key=lambda row:row[0]):
            current=list(group)
            if last is None or (at-last).total_seconds()>gap:earlier=[]
            eligible=[(when,vec) for when,vec in earlier if (at-when).total_seconds()>delay][-length:]
            expected=np.zeros((length,65));mask=np.zeros(length,bool)
            for j,(when,vec) in enumerate(eligible):
                expected[j,:64]=vec;expected[j,64]=min(1,math.log1p((at-when).total_seconds())/math.log1p(gap));mask[j]=True
            for _,iid,_,_,_ in current:
                i=row_map[iid]
                np.testing.assert_array_equal(a['event_mask'][i],mask)
                error=float(np.max(np.abs(a['events'][i]-expected)));maximum=max(maximum,error);assert error<=1e-6
                active+=bool(eligible)
            clicked=[mapping[article] for row in current for article,y in row[4] if y]
            if clicked:earlier.append((at,mean_vector(a['vectors'][clicked])))
            last=at
    result={'status':'passed','impressions_checked':len(rows),'with_session_context':active,'maximum_feature_difference':maximum,'absolute_tolerance':1e-6,
            'data_manifest_sha256':sha256(directory/'manifest.json'),'frozen_index_manifest_sha256':sha256(index_directory/'manifest.json'),
            'method':'Independent raw TSV parsing; per-user session partitioning; float64 history and time-group centroids; delayed feedback masks; full candidate/label/first-observation checks; frozen vector equality. No newsrec imports.',
            'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-began,
            'source_sha256':sha256(Path(__file__)),'final_holdout_opened':False}
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    return result


def slate(order,at,first,vectors,categories,mapping):
    chosen=order[:10]
    ages=[max(0,(at-first[x]).total_seconds()/3600) for x in chosen]
    distances=[1-max(-1,min(1,float(np.asarray(vectors[mapping[x]],dtype=np.float64)@np.asarray(vectors[mapping[y]],dtype=np.float64)))) for j,x in enumerate(chosen) for y in chosen[j+1:]]
    return {'categories@10':len({categories[mapping[x]] for x in chosen}),
            'mean_first_seen_age_hours@10':math.fsum(ages)/len(ages),
            'intra_list_cosine_distance@10':math.fsum(distances)/len(distances) if distances else 0.0}


def policy_check(base,actual,preset,at,first,vectors,categories,mapping):
    fresh,diversity={'fresh':(.15,0),'balanced':(.10,.15)}[preset]
    relevance={a:1-i/max(1,len(base)-1) for i,a in enumerate(base)}
    selected=[];remaining=set(base);maximum=0.0
    for chosen in actual[:10]:
        utility={}
        for article in remaining:
            v=np.asarray(vectors[mapping[article]],dtype=np.float64)
            redundancy=max((max(0,float(v@vectors[mapping[prev]])) for prev in selected),default=0)
            same=sum(categories[mapping[article]]==categories[mapping[p]] for p in selected)/max(1,len(selected))
            age=max(0,(at-first[article]).total_seconds()/3600)
            utility[article]=relevance[article]+fresh*math.exp(-age/24)-diversity*(.5*redundancy+.5*same)
        error=max(utility.values())-utility[chosen]
        assert error<=1e-6,(preset,error)
        maximum=max(maximum,error);selected.append(chosen);remaining.remove(chosen)
    assert actual==selected+[a for a in base if a not in selected]
    return maximum


def metric_audit(behaviors,directory,report_path,retrieval_report,output):
    began=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    manifest,a=load_arrays(directory);report=json.loads(report_path.read_text())
    assert sha256(directory/'manifest.json')==report['training_config']['data_manifest_sha256']
    assert sha256(behaviors)==manifest['input_sha256']['behaviors.tsv']
    cutoff=datetime.fromisoformat(manifest['cutoff'])
    all_rows,first,exposures,clicks,users=truth(behaviors,cutoff)
    rows=[row for row in all_rows if row[0]>=cutoff]
    position={str(iid):i for i,iid in enumerate(a['impression_ids'])}
    mapping={article:i for i,article in enumerate(a['article_ids'])}
    previous=json.loads(retrieval_report.read_text());original=previous['arms']['learned_content']['private_prediction_artifact']
    assert sha256(retrieval_report)==report['source_retrieval_report_sha256']
    original_path=retrieval_report.parent/original['file'];assert sha256(original_path)==original['sha256']
    pred=report['private_prediction_artifact'];path=report_path.parent/pred['file'];assert sha256(path)==pred['sha256']
    popular=sorted(first,key=lambda article:(-clicks[article],key(article)))
    sample=set(sorted([row[1] for row in rows],key=lambda iid:hashlib.sha256(('session-policy-audit:'+iid).encode()).hexdigest())[:128])
    values=defaultdict(lambda:defaultdict(list));counts=Counter();attempts=Counter();hits=defaultdict(Counter);exposure=defaultdict(lambda:defaultdict(lambda:[0,0]));coverage=defaultdict(set)
    digest=hashlib.sha256();policy_maximum=0.0;policy_queries=0
    def add(group,result):
        for name,value in result.items():values[group][name].append(value)
    with gzip.open(path,'rt',encoding='utf-8') as stream,gzip.open(original_path,'rt',encoding='utf-8') as previous_stream:
        for at,iid,user,history,items in rows:
            line=next(stream);digest.update(line.encode());prediction=json.loads(line);assert prediction['iid']==iid
            prior=json.loads(next(previous_stream));assert prior[0]==iid
            positives={article for article,y in items if y};labels=dict(items);row=position[iid]
            segments=['all','with_session' if a['event_mask'][row].any() else 'without_session']
            if user not in users:segments.append('cold_user')
            if not history:segments.append('empty_long_history')
            if any(exposures[article]==0 for article in positives):segments.append('cold_positive_article')
            for segment in segments:counts[segment+'_impressions']+=1
            counts['candidate_entries']+=len(items);counts['positive_events']+=len(positives)
            counts['zero_positive_impressions']+=not positives;counts['single_class_impressions']+=len(positives) in (0,len(items))
            counts['unavailable_positive_events']+=sum(first[x]>=at for x in positives);attempts.update(positives)
            for arm,ranked in prediction['logged'].items():
                order=ranked['ids'];assert len(order)==len(set(order))==len(items) and set(order)==set(labels)
                scores=ranked.get('scores',list(range(len(order),0,-1)))
                if 'scores' in ranked:assert order==[x for _,_,x in sorted((-s,key(x),x) for x,s in zip(order,scores))]
                metrics=score([labels[x] for x in order],scores)
                if 'scores' not in ranked:metrics.pop('auc',None)
                metrics.update(slate(order,at,first,a['vectors'],a['categories'],mapping))
                for segment in segments:add(('logged_ranking',arm,segment),metrics)
                add(('daily',str(at.date()),arm),metrics)
                selected=set(order[:10]);hits[arm].update(positives&selected)
                for article in positives:
                    n=exposures[article];bucket='unseen' if n==0 else 'tail_1_99' if n<100 else 'middle_100_999' if n<1000 else 'head_1000_plus'
                    exposure[arm][bucket][0]+=article in selected;exposure[arm][bucket][1]+=1
            pool=set(prior[2])|set(list(itertools.islice((x for x in popular if first[x]<at),100)))
            for preset,order in prediction['catalog'].items():
                assert len(order)==len(set(order)) and set(order)==pool and all(first[x]<at for x in order)
                metrics={}
                if positives:
                    ranks=[i+1 for i,x in enumerate(order) if x in positives]
                    metrics={f'recall@{k}':sum(r<=k for r in ranks)/len(positives) for k in (10,100)}
                    metrics['candidate_pool_recall']=len(ranks)/len(positives)
                    metrics['ndcg@10']=math.fsum(1/math.log2(r+1) for r in ranks if r<=10)/math.fsum(1/math.log2(r+1) for r in range(1,min(10,len(positives))+1))
                metrics.update(slate(order,at,first,a['vectors'],a['categories'],mapping))
                for segment in segments:add(('catalog_ranking',preset,segment),metrics)
                coverage[preset].update(order[:10])
            base=prediction['catalog']['relevance'];scores=prediction['catalog_relevance_scores']
            assert base==[x for _,_,x in sorted((-s,key(x),x) for x,s in zip(base,scores))]
            if iid in sample:
                for preset in ('fresh','balanced'):
                    for base,actual in [(prediction['logged']['session']['ids'],prediction['logged']['session_'+preset]['ids']),
                                        (prediction['catalog']['relevance'],prediction['catalog'][preset])]:
                        policy_maximum=max(policy_maximum,policy_check(base,actual,preset,at,first,a['vectors'],a['categories'],mapping))
                policy_queries+=1
        assert stream.readline()==previous_stream.readline()==''
    assert digest.hexdigest()==pred['uncompressed_sha256'] and dict(counts)==report['counts']
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
    expected_groups={(root,arm,segment) for root in ('logged_ranking','catalog_ranking') for arm,groups in report[root].items() for segment,v in groups.items() if v}
    expected_groups|={('daily',day,arm) for day,groups in report['daily'].items() for arm,v in groups.items() if v}
    assert set(values)==expected_groups
    for arm,buckets in exposure.items():
        expected=report['exposure_analysis'][arm]
        assert set(buckets)==set(expected['positive_recall@10'])
        for bucket,(hit,n) in buckets.items():assert expected['positive_recall@10'][bucket]=={'hits':hit,'events':n,'recall':hit/n}
        assert expected['distinct_clicked_articles']==len(attempts)
        assert abs(expected['macro_clicked_article_recall@10']-math.fsum(hits[arm][x]/n for x,n in attempts.items())/len(attempts))<1e-10
    for arm,articles in coverage.items():assert len(articles)==report['catalog_coverage'][arm]['distinct_top10_articles']
    result={'status':'passed','impressions_checked':len(rows),'metric_values_checked':checked,'metric_groups_checked':len(values),
            'maximum_aggregate_difference':maximum,'tolerances':{'ranking':1e-10,'age':1e-8,'cosine_diversity':1e-7,'policy_utility':1e-6},
            'policy_queries_checked':policy_queries,'policy_decisions_checked':policy_queries*4,'maximum_policy_utility_difference':policy_maximum,
            'prediction_sha256':digest.hexdigest(),'report_sha256':sha256(report_path),'source_sha256':sha256(Path(__file__)),
            'method':'Independent raw TSV labels and exposure counts, full frozen prediction digest/pool/time checks, rank-sum AUC/positive-rank metrics, float64 slate metrics, article-macro recall, sampled brute-force controls. No newsrec imports.',
            'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-began,'final_holdout_opened':False}
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['context','metrics'])
    for name in ('behaviors','data','output'):parser.add_argument('--'+name,required=True,type=Path)
    for name in ('index','report','retrieval-report'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    result=context_audit(args.behaviors,args.data,args.index,args.output) if args.stage=='context' else metric_audit(args.behaviors,args.data,args.report,args.retrieval_report,args.output)
    print(json.dumps(result,indent=2))
