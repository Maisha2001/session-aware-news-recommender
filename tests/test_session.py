from datetime import datetime
from pathlib import Path
import sqlite3

import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('tensorflow')
pytest.importorskip('faiss')

from newsrec.data import ingest
from newsrec.retrieval_index import TemporalIndex
from newsrec.session_data import SessionHistory, prepare_session, load_session
from newsrec.session_model import SessionRanker, training_examples, tf
from newsrec.rank_policy import rerank, slate_metrics


@pytest.fixture
def session_fixture(tmp_path):
    news=tmp_path/'news.tsv'
    news.write_text(''.join(f'N{i}\tcategory{i%2}\tsub\tAuthored title {i}\ttext\t\t[]\t[]\n' for i in range(1,7)),encoding='utf-8')
    behaviors=tmp_path/'behaviors.tsv'
    rows=['1\tU1\t11/12/2019 08:00:00 AM\tN1\tN2-1 N3-0',
          '2\tU1\t11/12/2019 08:05:00 AM\tN1\tN3-1 N2-0',
          '3\tU1\t11/12/2019 08:06:00 AM\tN1\tN4-1 N3-0',
          '4\tU1\t11/12/2019 08:06:00 AM\tN1\tN5-1 N2-0',
          '5\tU1\t11/12/2019 08:12:00 AM\tN1\tN2-1 N4-0',
          '6\tU1\t11/12/2019 08:43:00 AM\tN1\tN2-0 N4-1',
          '7\tU2\t11/13/2019 08:00:00 AM\t\tN6-1 N4-0',
          '8\tU2\t11/13/2019 08:06:00 AM\tN6\tN5-1 N2-0',
          '9\tU2\t11/13/2019 08:12:00 AM\tN6\tN4-0 N3-0']
    behaviors.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    database=tmp_path/'fixture.sqlite'
    ingest(news,behaviors,database,datetime(2019,11,13))
    with sqlite3.connect(database) as con:
        observed=con.execute('SELECT news_id,first_seen FROM observations ORDER BY news_id').fetchall()
    from newsrec.retrieval_data import seconds
    rng=np.random.default_rng(81)
    vectors=rng.normal(size=(len(observed),64)).astype(np.float32)
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    index=tmp_path/'index'
    TemporalIndex(vectors,[seconds(at) for _,at in observed],[a for a,_ in observed]).save(index)
    return database,index,behaviors


def test_session_delay_is_strict_and_inactivity_resets():
    history=SessionHistory()
    vector=np.ones(64,dtype=np.float32)
    assert history.before('user',0)==[]
    history.after('user',0,vector)
    assert history.before('user',300)==[]
    assert len(history.before('user',301))==1
    assert history.before('user',2102)==[]
    with pytest.raises(ValueError):history.before('user',2101)


def test_context_groups_ties_and_never_reads_current_or_future_labels(session_fixture,tmp_path):
    database,index,_=session_fixture
    data=prepare_session(database,index,tmp_path/'prepared')
    np.testing.assert_array_equal(data.arrays['event_mask'].sum(axis=1),[0,0,1,1,3,0,0,1,2])
    np.testing.assert_array_equal(data.arrays['events'][2],data.arrays['events'][3])
    with sqlite3.connect(database) as con:
        con.execute("UPDATE candidates SET clicked=1-clicked WHERE impression_id IN ('3','4','8','9')")
    changed=prepare_session(database,index,tmp_path/'changed')
    np.testing.assert_array_equal(data.arrays['events'][:4],changed.arrays['events'][:4])
    np.testing.assert_array_equal(data.arrays['events'][7],changed.arrays['events'][7])
    assert not np.array_equal(data.arrays['events'][4],changed.arrays['events'][4])
    with (tmp_path/'prepared/times.npy').open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='hash'):load_session(tmp_path/'prepared')


def test_training_negatives_are_logged_nonclicks_and_exclude_development(session_fixture,tmp_path):
    database,index,_=session_fixture
    data=prepare_session(database,index,tmp_path/'prepared')
    rows,candidates,skipped=training_examples(data)
    assert skipped==0 and len(rows)==6
    for row,selected in zip(rows,candidates):
        assert data.arrays['fit'][row]
        ids,labels=data.items(row)
        assert selected[0] in ids[labels==1]
        assert all(i in ids[labels==0] for i in selected[1:])


def test_session_order_changes_prediction_and_masked_padding_does_not():
    rng=np.random.default_rng(22)
    vectors=rng.normal(size=(5,64)).astype(np.float32)
    inputs={'long':vectors[1:2],'events':rng.normal(size=(1,3,65)).astype(np.float32),'event_mask':np.asarray([[True,True,False]]),
            'candidate_ids':np.asarray([[1,2,3]]),'ages':np.zeros((1,3),np.float32)}
    tf.keras.utils.set_random_seed(3)
    model=SessionRanker(vectors)
    first=model(inputs).numpy()
    modified={**inputs,'events':inputs['events'].copy()};modified['events'][:,2]=100
    np.testing.assert_array_equal(first,model(modified).numpy())
    modified['events'][:,[0,1]]=modified['events'][:,[1,0]]
    assert not np.allclose(first,model(modified).numpy())
    no_session=SessionRanker(vectors,False)
    np.testing.assert_array_equal(no_session(inputs).numpy(),no_session(modified).numpy())


def test_freshness_and_diversity_policies_preserve_candidates():
    ids=[f'N{i}' for i in range(20)]
    vectors=np.vstack([np.tile([1.,0.],(10,1)),np.tile([0.,1.],(10,1))])
    categories=['a']*10+['b']*10
    scores=np.arange(20,0,-1)
    ages=np.asarray([240.]*10+[0.]*10)
    relevance=rerank(ids,scores,vectors,categories,ages,'relevance')
    fresh=rerank(ids,scores,vectors,categories,ages,'fresh')
    balanced=rerank(ids,scores,vectors,categories,ages,'balanced')
    assert relevance==list(range(10))
    assert any(i>=10 for i in fresh) and any(i>=10 for i in balanced)
    assert len(set(balanced))==10 and set(balanced)<=set(range(20))
    before=slate_metrics(relevance,vectors,categories,ages)
    after=slate_metrics(balanced,vectors,categories,ages)
    assert after['categories@10']>before['categories@10']
    assert after['mean_first_seen_age_hours@10']<before['mean_first_seen_age_hours@10']


def test_fixed_session_training_and_reload(session_fixture,tmp_path):
    from newsrec.train_session import train,inputs_for
    from newsrec.evaluate_session import evaluate
    import gzip,hashlib,json,sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from audit_session import context_audit,metric_audit
    database,index,behaviors=session_fixture
    data=prepare_session(database,index,tmp_path/'prepared')
    checked=context_audit(behaviors,tmp_path/'prepared',index,tmp_path/'context-audit.json')
    assert checked['impressions_checked']==9 and checked['with_session_context']==5
    result=train(tmp_path/'prepared',tmp_path/'model',epochs=1,batch_size=2)
    rows,candidates,_=training_examples(data)
    assert result['config']['fit_examples']==6
    for arm,enabled in [('long_only',False),('session',True)]:
        model=SessionRanker(data.arrays['vectors'],enabled)
        inputs=inputs_for(data,rows[:2],candidates[:2]);model(inputs)
        model.load_weights(str(tmp_path/f'model/{arm}-epoch-1.weights.h5'))
        assert np.isfinite(model(inputs).numpy()).all()
    prior=tmp_path/'prior.jsonl.gz'
    with gzip.open(prior,'wt',encoding='utf-8') as stream:
        for row in np.flatnonzero(~data.arrays['fit']):
            candidates,_=data.items(row)
            eligible=np.flatnonzero(data.arrays['first_seen']<data.arrays['times'][row])
            stream.write(json.dumps([str(data.arrays['impression_ids'][row]),[[str(data.arrays['article_ids'][i]),0.] for i in candidates],data.arrays['article_ids'][eligible].tolist()])+'\n')
    old={'arms':{'learned_content':{'private_prediction_artifact':{'file':prior.name,'sha256':hashlib.sha256(prior.read_bytes()).hexdigest()}}}}
    (tmp_path/'prior-report.json').write_text(json.dumps(old))
    evaluated=evaluate(tmp_path/'prepared',tmp_path/'model',tmp_path/'prior-report.json',tmp_path/'evaluation',batch_size=2)
    assert evaluated['counts']['all_impressions']==3 and evaluated['counts']['zero_positive_impressions']==1
    audited=metric_audit(behaviors,tmp_path/'prepared',tmp_path/'evaluation/report.json',tmp_path/'prior-report.json',tmp_path/'metric-audit.json')
    assert audited['status']=='passed' and audited['policy_queries_checked']==3
