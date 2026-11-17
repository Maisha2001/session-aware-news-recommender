"""Verify aggregate learned-retrieval evidence and the measured implementation."""

import json
from pathlib import Path

from verify_evidence import digest


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/'results/retrieval-v1'
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert digest(folder/name) == expected, name
    report = json.loads((folder/'report.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    training = json.loads((folder/'training.json').read_text())
    audit = json.loads((folder/'audit.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        assert digest(root/name) == expected, name
    assert training['config'] == report['training_config']
    assert protocol['frozen_at_utc'] < training['config']['started_at_utc'] < training['completed_at_utc'] < report['started_at_utc']
    assert not training['development_scores_used_for_selection']
    assert not training['final_holdout_opened'] and not report['final_holdout_opened'] and not audit['final_archive_opened']
    assert audit['status'] == 'passed' and audit['report_sha256'] == digest(folder/'report.json')
    assert audit['source_sha256'] == digest(root/'scripts/audit_retrieval.py')
    assert audit['independent_metric_source_sha256'] == digest(root/'scripts/audit_baseline.py')
    assert report['data']['database_sha256'] == protocol['database_sha256']
    for arm, values in report['arms'].items():
        assert values['prediction_sha256'] == audit['prediction_order_sha256'][arm]
        assert values['counts']['all_impressions'] == audit['development_impressions_per_arm']
        assert values['audit']['exhaustive_float64_queries'] == 128
        assert values['audit']['reload_predictions_bitwise_equal']
        assert values['audit']['persisted_index_used_for_all_queries']
    assert report['arms']['initial_content']['counts'] == report['arms']['learned_content']['counts']
    print(f"Verified {len(manifest['sha256'])} retrieval evidence files, measured source, fixed training and independent audit.")


if __name__ == '__main__':
    main()
