import json
from pathlib import Path
import sys
import zipfile

import pytest

np=pytest.importorskip('numpy')
pytest.importorskip('tensorflow')
pytest.importorskip('faiss')
pytest.importorskip('tensorflow_recommenders')

from newsrec.data import connect_readonly, file_sha256
from newsrec.heldout_data import prepare_final, read_final
from newsrec.heldout import final_metrics, predict_final
from newsrec.retrieval_data import text_tokens
from newsrec.retrieval_model import ContentRetriever
from newsrec.session_model import SessionRanker


def archive_at(path, behavior=None, news=None):
    rows=behavior or [
        '1\tU1\t11/15/2019 12:00:00 PM\tN1\tN5-1 N2-0',
        '2\tU1\t11/15/2019 12:00:00 PM\tN1\tN2-1 N3-0',
        '3\tU1\t11/15/2019 12:05:00 PM\tN1\tN1-0 N2-0',
        '4\tU1\t11/15/2019 12:06:00 PM\tN1\tN5-1 N1-0',
        '5\tU8\t11/15/2019 12:06:00 PM\t\tN5-1 N3-0',
        '6\tU1\t11/15/2019 12:37:00 PM\tN5\tN5-1 N2-0']
    content=news or ''.join(f'N{i}\tcategory\tsubcategory\tChanged snapshot {i}\tAuthored abstract\t\t[]\t[]\n' for i in (1,2,3,4,5))
    with zipfile.ZipFile(path,'x') as stream:
        stream.writestr('news.tsv',content);stream.writestr('behaviors.tsv','\n'.join(rows)+'\n')
    return path


def test_final_ingestion_keeps_original_text_and_all_outcomes(database,tmp_path):
    archive=archive_at(tmp_path/'final.zip')
    articles,seen,users,content,rows,meta=read_final(database,archive,tmp_path/'input')
    assert articles['N1'][2]=='Authored title 1'
    assert articles['N5'][2]=='Authored title 5'
    assert len(rows)==6 and sum(sum(r.labels) for r in rows)==5
    assert meta['metadata_counts']['changed_shared_metadata_ids']==5
    assert rows[0].impression_id=='1'
    assert meta['input_sha256']['behaviors.tsv']==file_sha256(tmp_path/'input/behaviors.tsv')
    with pytest.raises(FileExistsError):read_final(database,archive,tmp_path/'input')


def test_final_archive_directory_wrapper_and_ambiguous_members(database,tmp_path):
    source=archive_at(tmp_path/'source.zip')
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(tmp_path/'wrapped.zip','x') as wrapped:
        for name in original.namelist():wrapped.writestr('MINDsmall_dev/'+name,original.read(name))
    result=read_final(database,tmp_path/'wrapped.zip',tmp_path/'wrapped-input')
    assert len(result[4])==6
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(tmp_path/'ambiguous.zip','x') as ambiguous:
        for name in original.namelist():
            ambiguous.writestr(name,original.read(name));ambiguous.writestr('MINDsmall_dev/'+name,original.read(name))
    with pytest.raises(ValueError,match='Expected one'):read_final(database,tmp_path/'ambiguous.zip',tmp_path/'ambiguous-input')


@pytest.mark.parametrize('rows,message',[
    (['1\tU1\t11/14/2019 08:00:00 AM\t\tN1-1 N2-0'],'strictly later'),
    (['1\tU1\t11/15/2019 12:00:00 PM\t\tN999-1 N2-0'],'no metadata'),
    (['1\tU1\t11/15/2019 12:00:00 PM\t\tN1-1 N2-0']*2,'Duplicate final')])
def test_final_ingestion_fails_without_excluding_bad_events(database,tmp_path,rows,message):
    with pytest.raises(ValueError,match=message):read_final(database,archive_at(tmp_path/'final.zip',rows),tmp_path/'input')


@pytest.fixture
def final_fixture(database,source,tmp_path):
    with connect_readonly(database) as con:
        records=con.execute('SELECT o.news_id,a.title,a.category,a.subcategory FROM observations o JOIN articles a USING(news_id) ORDER BY o.news_id').fetchall()
        fit=dict(con.execute("SELECT news_id,sum(clicked) FROM candidates JOIN impressions USING(impression_id) WHERE split='fit' GROUP BY news_id"))
        exposures=dict(con.execute("SELECT news_id,count(*) FROM candidates JOIN impressions USING(impression_id) WHERE split='fit' GROUP BY news_id"))
    ids=['']+[r[0] for r in records]
    tokens=np.asarray([[0]*32]+[text_tokens(r[1],r[2],r[3]) for r in records],np.int32)
    content=ContentRetriever(tokens);content(np.zeros((1,20),np.int32))
    paths={'content':tmp_path/'content.weights.h5'};content.save_weights(str(paths['content']))
    vectors=content.articles(np.arange(len(ids))).numpy()
    old=tmp_path/'original';old.mkdir()
    for name,array in {'article_ids':np.asarray(ids),'vectors':vectors,'fit_clicks':np.asarray([fit.get(a,0) for a in ids]),
                       'fit_exposures':np.asarray([exposures.get(a,0) for a in ids])}.items():np.save(old/(name+'.npy'),array)
    (old/'manifest.json').write_text(json.dumps({'sha256':{p.name:file_sha256(p) for p in old.glob('*.npy')}}))
    run=tmp_path/'final'
    data=prepare_final(database,archive_at(tmp_path/'final.zip'),old,paths['content'],run)
    for name,enabled in [('long_only',False),('session',True)]:
        ranker=SessionRanker(data.arrays['vectors'],enabled)
        ranker({'long':np.zeros((1,64),np.float32),'events':np.zeros((1,10,65),np.float32),'event_mask':np.zeros((1,10),bool),
                'candidate_ids':np.ones((1,1),np.int32),'ages':np.zeros((1,1),np.float32)})
        paths[name]=tmp_path/(name+'.weights.h5');ranker.save_weights(str(paths[name]))
    return data,paths,run,source[1]


def test_final_context_excludes_equal_time_exact_delay_and_inactivity(final_fixture):
    data,_,_,_=final_fixture
    np.testing.assert_array_equal(data.arrays['event_mask'].sum(axis=1),[0,0,0,1,0,0])
    assert data.arrays['impression_ids'][0]=='final:1'
    mapping={a:i for i,a in enumerate(data.arrays['article_ids'])}
    expected=data.arrays['vectors'][[mapping['N5'],mapping['N2']]].mean(axis=0)
    expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(data.arrays['events'][3,0,:64],expected,atol=1e-7)


def test_final_prediction_freeze_and_independent_audit(final_fixture):
    data,paths,run,training=final_fixture
    labels=data.arrays['labels']
    class SealedLabels:
        def __getitem__(self,key):raise AssertionError('Current/future labels accessed while predicting')
    data.arrays['labels']=SealedLabels()
    manifest=predict_final(data,paths,run/'predictions',batch_size=2,audit_queries=6)
    data.arrays['labels']=labels
    report=final_metrics(data,run/'predictions',run/'report.json')
    assert report['counts']['impressions']==6 and report['counts']['positive_events']==5
    assert report['counts']['unavailable_positive_events']==1
    assert isinstance(report['comparison_rows'][0]['logged_ndcg10'],float)
    assert manifest['audit']['exhaustive_float64_queries']==6
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_heldout import audit
    result=audit(training,run,run/'audit.json')
    assert result['status']=='passed' and result['context_impressions_checked']==6
    with pytest.raises(FileExistsError):predict_final(data,paths,run/'predictions')
    with pytest.raises(FileExistsError):final_metrics(data,run/'predictions',run/'report.json')
