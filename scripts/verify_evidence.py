"""Verify published aggregate evidence against the measured source files."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'results/baseline-v1'
    report = json.loads((evidence / 'report.json').read_text())
    protocol = json.loads((evidence / 'protocol.json').read_text())
    audit = json.loads((evidence / 'audit.json').read_text())
    data = json.loads((evidence / 'data-manifest.json').read_text())
    manifest = json.loads((evidence / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert digest(evidence / name) == expected, name
    for name, expected in protocol['source_sha256'].items():
        assert digest(root / name) == expected, name
    assert audit['source_sha256'] == digest(root / 'scripts/audit_baseline.py')
    assert audit['report_sha256'] == digest(evidence / 'report.json')
    assert audit['prediction_order_sha256'] == report['prediction_order_sha256']
    assert audit['status'] == 'passed' and not audit['final_archive_opened']
    assert data['final_holdout_content_opened'] is False
    assert report['dataset']['cutoff'] == protocol['cutoff']
    assert report['dataset']['partition_counts'] == protocol['expected_partition_counts']
    for name, expected in data['training_inputs'].items():
        assert report['dataset']['input_sha256'][name] == expected['sha256']
        assert protocol['training_inputs'][name] == expected
    assert protocol['frozen_at_utc'] < report['runtime']['started_at_utc']
    print(f"Verified {len(manifest['sha256'])} evidence files, measured sources, input identities and independent audit.")


if __name__ == '__main__':
    main()
