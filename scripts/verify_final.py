"""Check public final evidence, source identity and release interfaces without private data."""
import json
from pathlib import Path

from verify_evidence import digest


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'results/heldout-v1'
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,sha in manifest['sha256'].items():assert digest(folder/name)==sha,name
    protocol=json.loads((folder/'protocol.json').read_text())
    for name,sha in protocol['source_sha256'].items():assert digest(root/name)==sha,name
    report=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'audit.json').read_text())
    assert audit['status']=='passed' and audit['report_sha256']==digest(folder/'report.json')
    assert report['counts']['impressions']==audit['impressions_checked']==73152
    assert report['counts']['candidate_entries']==2740998 and report['counts']['positive_events']==111383
    prediction=report['prediction_manifest']
    assert prediction['predictions_sha256']==audit['prediction_sha256']
    assert prediction['uncompressed_sha256']==audit['uncompressed_prediction_sha256']
    assert prediction['model_sha256']==protocol['model_sha256']
    assert prediction['audit']['exhaustive_float64_queries']==128
    assert prediction['audit']['exact_order_matches']==128
    assert prediction['audit']['reload_impressions_per_arm']==256 and prediction['audit']['reload_predictions_bitwise_equal']
    assert audit['context_impressions_checked']==73152 and audit['policy_decisions_checked']==512
    assert protocol['frozen_at_utc']<report['data']['archive_content_opened_at_utc']<=report['data']['labels_first_parsed_at_utc']
    assert report['data']['prepared_at_utc']<prediction['frozen_at_utc']<report['started_at_utc']<report['completed_at_utc']<audit['completed_at_utc']
    assert set(report['logged_ranking'])=={'popularity','content','long_only','session','session_fresh','session_balanced'}
    assert set(report['catalog_ranking'])=={'popularity','content','session','session_fresh','session_balanced'}
    assert len(report['comparison_rows'])==3
    release=root/'results/release-v1'
    rm=json.loads((release/'manifest.json').read_text())
    for name,sha in rm['sha256'].items():assert digest(release/name)==sha,name
    for name,sha in rm['source_sha256'].items():assert digest(root/name)==sha,name
    demo=json.loads((release/'demo.json').read_text());tracking=json.loads((release/'tracking.json').read_text())
    assert demo['status']=='passed' and demo['javascript_errors']==[] and demo['final_cases_in_demo']==0
    assert tracking['status']=='FINISHED' and not tracking['training_performed']
    assert tracking['report_sha256']==digest(folder/'report.json')
    assert tracking['artifact_sha256']['evidence/report.json']==digest(folder/'report.json')
    assert len(tracking['artifact_sha256'])>=6
    print('Verified frozen final predictions, all73,152 impressions, independent metrics, models, local demo and tracking evidence.')


if __name__=='__main__':main()
