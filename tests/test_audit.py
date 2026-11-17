import importlib.util
import json
from pathlib import Path

import pytest

from newsrec.baseline import evaluate
from newsrec.data import ingest


def auditor():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_baseline.py'
    spec = importlib.util.spec_from_file_location('independent_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit


def test_independent_raw_tsv_audit_and_tamper_detection(source, tmp_path):
    news, behaviors, cutoff = source
    database = tmp_path / 'source.sqlite'
    report_path = tmp_path / 'report.json'
    ingest(news, behaviors, database, cutoff)
    report = evaluate(database, report_path)
    result = auditor()(news, behaviors, database, report_path, tmp_path / 'audit.json')
    assert result['status'] == 'passed'
    assert result['development_impressions'] == 4
    assert result['metric_values_checked'] > 20
    report['impression_ranking']['popularity']['all']['auc']['mean'] += 0.1
    report_path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        auditor()(news, behaviors, database, report_path, tmp_path / 'tampered-audit.json')
    assert not (tmp_path / 'tampered-audit.json').exists()
