"""Verify measured session-ranking source, preserved protocols and audited results."""

import json
from pathlib import Path

from verify_evidence import digest


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'results/session-v1'
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():assert digest(folder/name)==expected,name
    protocol=json.loads((folder/'protocol.json').read_text())
    clarification=json.loads((folder/'protocol-documentation-clarification.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    training=json.loads((folder/'training.json').read_text())
    contexts=json.loads((folder/'context-audit.json').read_text())
    metrics=json.loads((folder/'metric-audit.json').read_text())
    sources=dict(protocol['source_sha256'])
    assert sources[clarification['path']]==clarification['original_sha256']
    corrected=(root/clarification['path']).read_bytes()
    assert digest(root/clarification['path'])==clarification['corrected_sha256']
    original=corrected.replace(clarification['corrected_text'].encode(),clarification['old_text'].encode())
    import hashlib
    assert hashlib.sha256(original).hexdigest()==clarification['original_sha256']
    sources[clarification['path']]=clarification['corrected_sha256']
    for path,expected in sources.items():assert digest(root/path)==expected,path
    assert protocol['frozen_at_utc']<training['config']['started_at_utc']<training['completed_at_utc']<report['started_at_utc']
    assert clarification['recorded_at_utc']<report['started_at_utc'] and not clarification['real_ranking_evaluation_started']
    assert contexts['completed_at_utc']<training['config']['started_at_utc']
    assert report['training_config']==training['config']
    assert contexts['data_manifest_sha256']==training['config']['data_manifest_sha256']
    assert contexts['data_manifest_sha256']==digest(folder/'data-manifest.json')
    assert contexts['source_sha256']==metrics['source_sha256']==digest(root/'scripts/audit_session.py')
    assert contexts['status']==metrics['status']=='passed'
    assert metrics['report_sha256']==digest(folder/'report.json')
    assert metrics['prediction_sha256']==report['private_prediction_artifact']['uncompressed_sha256']
    assert report['source_retrieval_report_sha256']==digest(root/'results/retrieval-v1/report.json')
    assert report['counts']['all_impressions']==metrics['impressions_checked']==61894
    assert metrics['policy_queries_checked']==128
    assert all(not obj['final_holdout_opened'] for obj in (report,training,contexts,metrics))
    assert not training['development_scores_used_for_selection']
    print(f"Verified {len(manifest['sha256'])} session evidence files, fixed training, source, context and metric audits.")


if __name__=='__main__':
    main()
