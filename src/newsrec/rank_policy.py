import hashlib
import math

import numpy as np


PRESETS={'relevance':(0.0,0.0),'fresh':(0.15,0.0),'balanced':(0.10,0.15)}


def rerank(ids, scores, vectors, categories, ages_hours, preset='relevance', k=10):
    if preset not in PRESETS or k<1 or len(ids)!=len(scores) or len(set(ids))!=len(ids):
        raise ValueError('Invalid ranking policy input')
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(ages_hours).all() or np.any(np.asarray(ages_hours)<0):
        raise ValueError('Scores and ages must be finite; ages nonnegative')
    ties=[hashlib.sha256(('newsrec-tie-v1:'+str(a)).encode()).hexdigest() for a in ids]
    order=sorted(range(len(ids)),key=lambda i:(-scores[i],ties[i]))
    if preset=='relevance':return order[:k]
    # Rank scale makes controls comparable across uncalibrated scoring models.
    relevance=np.zeros(len(ids),dtype=float)
    for rank,i in enumerate(order):relevance[i]=1-rank/max(1,len(ids)-1)
    freshness=np.exp(-np.asarray(ages_hours,dtype=float)/24)
    fresh_weight,diversity_weight=PRESETS[preset]
    selected=[]
    active=np.ones(len(ids),dtype=bool)
    redundancy=np.zeros(len(ids),dtype=float)
    category_count=np.zeros(len(ids),dtype=float)
    categories=np.asarray(categories)
    vectors=np.asarray(vectors)
    while active.any() and len(selected)<k:
        utilities=relevance+fresh_weight*freshness-diversity_weight*(0.5*redundancy+0.5*category_count/max(1,len(selected)))
        utilities[~active]=-np.inf
        best=np.flatnonzero(utilities==utilities.max())
        chosen=min(best,key=lambda i:ties[i])
        selected.append(int(chosen));active[chosen]=False
        redundancy=np.maximum(redundancy,np.maximum(0,vectors@vectors[chosen]))
        category_count+=categories==categories[chosen]
    return selected


def slate_metrics(order, vectors, categories, ages_hours):
    chosen=list(order[:10])
    if not chosen:return {}
    distances=[1-float(np.clip(np.dot(vectors[i],vectors[j]),-1,1)) for offset,i in enumerate(chosen) for j in chosen[offset+1:]]
    return {'categories@10':len({categories[i] for i in chosen}),'mean_first_seen_age_hours@10':float(np.mean(np.asarray(ages_hours)[chosen])),
            'intra_list_cosine_distance@10':math.fsum(distances)/len(distances) if distances else 0.0}
