"""Verify source-bound aggregate evidence without accessing private data."""
import json
from pathlib import Path

from verify_evidence import digest


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'results/service-v1'
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert digest(folder / name) == expected, name
    protocol = json.loads((folder / 'protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        assert digest(root / name) == expected, name
    report = json.loads((folder / 'report.json').read_text())
    assert report['status'] == 'passed' and not report['final_holdout_opened']
    assert not protocol['labels_opened_for_this_check'] and not protocol['final_holdout_opened']
    for phase in ('parity', 'consistency', 'restart', 'redis_down', 'postgres_down', 'recovered'):
        record = report['checks'][phase]
        assert record['status'] == 'passed' and not record['final_holdout_opened']
        assert protocol['frozen_at_utc'] < record['started_at_utc'] < record['completed_at_utc']
    parity = report['checks']['parity']
    assert parity['queries'] == protocol['queries'] == 128
    assert parity['exact_top20_by_policy'] == dict.fromkeys(protocol['policies'], 128)
    assert parity['scores_compared'] == 128 * 3 * 20
    assert parity['maximum_absolute_score_difference'] <= protocol['score_absolute_tolerance']
    assert parity['serial_uncached_http']['count'] == 383
    assert parity['serial_cached_http']['count'] == 128
    assert parity['concurrent_uncached_http']['count'] == 64
    assert report['checks']['consistency']['index_articles_each_worker'] == 51283
    assert report['checks']['postgres_down']['http_status'] == 503
    assert report['checks']['redis_down']['cache_failure_fallback']
    assert report['checks']['recovered']['feedback_revision_preserved'] == 2
    assert report['preservation']['existing_containers_unchanged'] == 7
    assert report['package']['returncode'] == 0
    assert report['source_reference_report_sha256'] == digest(root / 'results/session-v1/report.json')
    assert report['source_prediction_compressed_sha256'] == protocol['reference_predictions_sha256']
    assert report['verification_source_sha256'] == digest(root / 'scripts/audit_service.py')
    print('Verified service protocol, source,384 top20 comparisons, consistency, recovery and package evidence.')


if __name__ == '__main__':
    main()
