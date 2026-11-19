import json

import pytest

pytest.importorskip('mlflow')
from newsrec.data import file_sha256
from newsrec.tracking import log_release


def test_local_tracking_roundtrip_and_model_guard(tmp_path):
    models={name:tmp_path/(name+'.h5') for name in ('content','long_only','session')}
    for name,path in models.items():path.write_bytes(('authored fixture '+name).encode())
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'counts':{'impressions':3},'prediction_manifest':{'model_sha256':{name:file_sha256(path) for name,path in models.items()}},
        'logged_ranking':{'session':{'all':{'ndcg@10':{'mean':0.25,'impressions':3}}}},'catalog_ranking':{}}))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'status':'passed','report_sha256':file_sha256(report)}))
    protocol=tmp_path/'protocol.json';protocol.write_text('{"authored":true}')
    artifact=tmp_path/'fixture.txt';artifact.write_text('Authored artifact; not model evidence.')
    receipt=tmp_path/'receipt.json'
    result=log_release(tmp_path/'local',report,audit,protocol,models,{'package/fixture.txt':artifact},receipt)
    assert result['status']=='FINISHED' and result['metrics_verified']==2
    assert len(result['artifact_sha256'])==7
    assert result['artifact_sha256']['package/fixture.txt']==file_sha256(artifact)
    with pytest.raises(FileExistsError):log_release(tmp_path/'local',report,audit,protocol,models,{},receipt)
    models['content'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='Model hash mismatch'):log_release(tmp_path/'other',report,audit,protocol,models,{},tmp_path/'other.json')
    assert not (tmp_path/'other').exists()


def test_tracking_rejects_remote_store_and_unverified_report(tmp_path):
    for value in ('https://example.com/store','//server/share'):
        with pytest.raises(ValueError,match='local filesystem'):log_release(value,'missing','missing','missing',{}, {},tmp_path/'receipt.json')
