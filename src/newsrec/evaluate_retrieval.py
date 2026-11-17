import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import time

import faiss
import numpy as np

from .baseline import tie_key
from .data import file_sha256
from .metrics import MetricMeans, auc, ranking_metrics
from .retrieval_data import prepare_content, development_impressions
from .retrieval_model import ContentRetriever, article_embeddings, tf
from .retrieval_index import TemporalIndex
from .train_retrieval import write_json


SEGMENTS = ('all', 'cold_user', 'warm_user', 'empty_history', 'cold_positive_article', 'unseen_content_positive')


def catalog_metrics(ranking, positives):
    if not positives:
        return {}
    result = {f'recall@{k}':len(set(ranking[:k]) & positives) / len(positives) for k in (20,100)}
    gain = math.fsum(1/math.log2(i+2) for i,a in enumerate(ranking[:100]) if a in positives)
    ideal = math.fsum(1/math.log2(i+2) for i in range(min(100,len(positives))))
    result['ndcg@100'] = gain / ideal
    return result


def batches(iterator, size):
    batch = []
    for row in iterator:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def exhaustive_check(vectors, query, at, index, result):
    scores = vectors.astype(np.float64) @ query.astype(np.float64)
    eligible = np.flatnonzero(index.first_seen < at)
    ranked = sorted(eligible, key=lambda i:(-scores[i],index.ties[i]))[:100]
    actual = [i-1 for i,_ in result]
    assert len(actual) == len(ranked) and len(set(actual)) == len(actual)
    errors = [abs(scores[i-1]-score) for i,score in result]
    maximum = max(errors, default=0.0)
    assert maximum < 1e-5
    if ranked:
        assert min(scores[i] for i in actual) >= scores[ranked[-1]] - 1e-6
    return actual == ranked, maximum


def evaluate(database, model_directory, output_directory, batch_size=128, audit_queries=128):
    if output_directory.exists():
        raise FileExistsError('Refusing to overwrite retrieval evidence')
    output_directory.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    training = json.loads((model_directory/'training.json').read_text())
    config = training['config']
    data = prepare_content(database, config['buckets'], config['token_length'], config['history_length'])
    if data.metadata != config['data']:
        raise ValueError('Training data identity/configuration changed')
    ids = [row[0] for row in development_impressions(database,data.mapping,config['history_length'])]
    audit_ids = set(sorted(ids, key=lambda iid:hashlib.sha256(('newsrec-audit-v1:'+iid).encode()).hexdigest())[:audit_queries])
    faiss.omp_set_num_threads(4)
    arms = {'initial_content':'initial.weights.h5','learned_content':f"epoch-{config['epochs']}.weights.h5"}
    reports = {}
    for arm, weights_name in arms.items():
        arm_start = time.perf_counter()
        weights = model_directory/weights_name
        assert file_sha256(weights) == training['weights_sha256'][weights_name]
        model = ContentRetriever(data.tokens, config['buckets'], config['dimension'], config['temperature'])
        model(tf.constant(data.fit_histories[:1]))
        model.load_weights(str(weights))
        vector_start = time.perf_counter()
        vectors = article_embeddings(model,len(data.article_ids))
        index = TemporalIndex(vectors,data.first_seen[1:],data.article_ids[1:])
        index.save(output_directory/arm)
        loaded_index = TemporalIndex.load(output_directory/arm)
        vector_seconds = time.perf_counter()-vector_start
        query_reload = model(tf.constant(data.fit_histories[:256])).numpy()
        reload_model = ContentRetriever(data.tokens,config['buckets'],config['dimension'],config['temperature'])
        reload_model(tf.constant(data.fit_histories[:1]))
        reload_model.load_weights(str(weights))
        np.testing.assert_array_equal(query_reload,reload_model(tf.constant(data.fit_histories[:256])).numpy())
        np.testing.assert_array_equal(model.articles(tf.range(1,min(257,len(data.article_ids)))).numpy(),
                                      reload_model.articles(tf.range(1,min(257,len(data.article_ids)))).numpy())
        del reload_model
        ranking = {s:MetricMeans() for s in SEGMENTS}
        retrieval = {s:MetricMeans() for s in SEGMENTS}
        daily = {}
        counts = Counter()
        digest = hashlib.sha256()
        batch_times, encoding_seconds = [], 0.0
        audit_count, exact_order, maximum_error = 0, 0, 0.0
        prediction_path = output_directory / (arm + '-predictions.jsonl.gz')
        prediction_stream = gzip.open(prediction_path, 'xt', encoding='utf-8')
        for batch in batches(development_impressions(database,data.mapping,config['history_length']),batch_size):
            before = time.perf_counter()
            queries = model(np.asarray([row[3] for row in batch],dtype=np.int32)).numpy()
            encoding_seconds += time.perf_counter()-before
            before = time.perf_counter()
            found = loaded_index.search(queries,[row[2] for row in batch],100)
            batch_times.append(time.perf_counter()-before)
            for q,(iid,user,at,history,items) in enumerate(batch):
                positives = {i for i,y in items if y}
                segments = ['all','warm_user' if user in data.fit_users else 'cold_user']
                if not any(history):segments.append('empty_history')
                if any(i not in data.fit_candidates for i in positives):segments.append('cold_positive_article')
                if any(i not in data.fit_content for i in positives):segments.append('unseen_content_positive')
                for segment in segments:counts[segment+'_impressions'] += 1
                counts['candidate_entries'] += len(items)
                counts['positive_events'] += len(positives)
                counts['unavailable_positive_events'] += sum(data.first_seen[i]>=at for i in positives)
                counts['zero_positive_impressions'] += not positives
                counts['single_class_impressions'] += len(positives) in (0,len(items))
                selected = [i for i,_ in found[q]]
                assert len(selected) == min(100,int((data.first_seen[1:]<at).sum()))
                assert len(set(selected))==len(selected) and all(data.first_seen[i]<at for i in selected)
                raw_scores = np.asarray([vectors[i-1] @ queries[q] for i,_ in items])
                ordered = sorted(range(len(items)),key=lambda i:(-raw_scores[i],tie_key(data.article_ids[items[i][0]])))
                metrics = ranking_metrics([items[i][1] for i in ordered])
                metrics.pop('positive_count')
                metrics['auc'] = auc([y for _,y in items],raw_scores.tolist())
                recall = catalog_metrics(selected,positives)
                if positives:recall['available_positive_fraction'] = sum(data.first_seen[i]<at for i in positives)/len(positives)
                for segment in segments:
                    ranking[segment].add(metrics)
                    retrieval[segment].add(recall)
                day = datetime.fromtimestamp(at,timezone.utc).date().isoformat()
                daily.setdefault(day,{'ranking':MetricMeans(),'catalog':MetricMeans()})
                daily[day]['ranking'].add(metrics)
                daily[day]['catalog'].add(recall)
                digest.update(json.dumps([iid,[data.article_ids[items[i][0]] for i in ordered],[data.article_ids[i] for i in selected]],separators=(',',':')).encode()+b'\n')
                prediction_stream.write(json.dumps([iid, [[data.article_ids[items[i][0]], float(raw_scores[i])] for i in ordered],
                                                     [data.article_ids[i] for i in selected]], separators=(',', ':')) + '\n')
                if iid in audit_ids:
                    exact,error = exhaustive_check(vectors,queries[q],at,index,found[q])
                    audit_count += 1
                    exact_order += exact
                    maximum_error = max(maximum_error,error)
        prediction_stream.close()
        assert audit_count == len(audit_ids)
        reports[arm] = {'counts':{name:int(value) for name,value in counts.items()},'logged_ranking':{s:m.report() for s,m in ranking.items()},
                        'catalog_retrieval':{s:m.report() for s,m in retrieval.items()},
                        'daily':{day:{k:v.report() for k,v in groups.items()} for day,groups in daily.items()},
                        'prediction_sha256':digest.hexdigest(),'weights_sha256':file_sha256(weights),
                        'private_prediction_artifact':{'file':prediction_path.name,'sha256':file_sha256(prediction_path)},
                        'audit':{'exhaustive_float64_queries':audit_count,'exact_order_matches':exact_order,'maximum_score_difference':maximum_error,
                                 'score_tolerance':1e-5,'boundary_tie_tolerance':1e-6,'policy_checks':'Every returned ID strictly earlier than its query; unique; complete requested cardinality.',
                                 'reload_query_predictions':len(query_reload),'reload_predictions_bitwise_equal':True,'persisted_index_used_for_all_queries':True},
                        'runtime':{'wall_seconds':time.perf_counter()-arm_start,'encoding_and_index_build_seconds':vector_seconds,
                                   'query_encoding_seconds':encoding_seconds,'catalog_search_seconds':sum(batch_times),
                                   'batch_size':batch_size,'catalog_batch_p50_seconds':float(np.median(batch_times)),
                                   'catalog_batch_p95_seconds':float(np.percentile(batch_times,95)),
                                   'index_bytes':(output_directory/arm/'vectors.faiss').stat().st_size}}
        write_json(output_directory/(arm+'-report.json'),reports[arm])
        print(json.dumps({'arm':arm,'recall@100':reports[arm]['catalog_retrieval']['all'].get('recall@100'),'seconds':reports[arm]['runtime']['wall_seconds']}),flush=True)
    assert reports['initial_content']['counts'] == reports['learned_content']['counts']
    report = {'format_version':1,'data':data.metadata,'training_config':config,'arms':reports,
              'protocol':{'availability':'All logged candidates for ranking. For catalog search, strict first_seen < query time; earlier label-free exposures allowed. All positives remain in denominators.',
                          'unseen_content_positive':'At least one positive absent from fitting candidate exposures and the last20 histories used for fitting. This is stricter than cold_positive_article.',
                          'cold_user':'Absent from fitting impressions. Content history still available; no user-ID embedding.',
                          'empty_history':'Trainable generic query vector; no subsequent user feedback.',
                          'initial_content':'Same saved initial word/article parameters with identity query projection; untuned content-overlap comparator.',
                          'metrics':'Macro per impression. nDCG@100 ideal denominator includes unavailable positives. Invalid zero-positive and one-class ratios excluded with counts.'},
              'limitations':['Only one fixed seed and three epochs; no significance claim or development-tuned checkpoint.',
                             'Fixed token hashing can collide; unseen words may remain untrained. Titles and category snapshots have no version history.',
                             'Mean of last20 provided earlier clicks has no per-click timestamps; this is a retrieval representation, not a timed session ranker.',
                             'In-batch negatives are other fitting clicks, with duplicate target IDs removed; other false negatives and exposure bias remain.',
                             'Offline results on the sampled observed catalog do not establish live engagement lift.',
                             'Exact FAISS search at this catalog size; timings are batches, not serving latency. Artifacts contain licensed-data-derived IDs and remain private.'],
              'software':{name:importlib.metadata.version(name) for name in ('tensorflow','tf-keras','tensorflow-recommenders','faiss-cpu','numpy')},
              'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-begin,
              'final_holdout_opened':False,'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ('evaluate_retrieval.py','retrieval_index.py','retrieval_model.py','retrieval_data.py')}}
    write_json(output_directory/'report.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate saved content retrievers against the chronological training-archive development split.')
    for name in ('database','model','output'):parser.add_argument('--'+name,required=True,type=Path)
    args = parser.parse_args()
    report = evaluate(args.database,args.model,args.output)
    print(json.dumps({'completed_at_utc':report['completed_at_utc'],'wall_seconds':report['wall_seconds']}))
