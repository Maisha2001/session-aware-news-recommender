from datetime import datetime
import json
import sqlite3
import subprocess
import sys
import pytest

from newsrec.baseline import eligible_topk, evaluate, fit_popularity, tie_key
from newsrec.data import connect_readonly, ingest


def test_fit_excludes_development_labels_and_history(database):
    with connect_readonly(database) as con:
        counts, users = fit_popularity(con)
    assert counts == {"N1": 2, "N2": 0}
    assert users == {"U1", "U2"}
    assert "N3" not in counts and "N4" not in counts


def test_same_time_and_future_articles_are_not_retrievable():
    seen = {"N1": "2019-11-12T23:59:59", "N2": "2019-11-13T00:00:00", "N3": "2019-11-14T00:00:00"}
    assert eligible_topk(["N3", "N2", "N1"], seen, "2019-11-13T00:00:00", 100) == ["N1"]
    assert eligible_topk(["N3", "N2", "N1"], seen, "2019-11-13T00:00:01", 100) == ["N2", "N1"]


def test_report_matches_hand_calculated_catalog_recall(database, tmp_path):
    report = evaluate(database, tmp_path / "report.json")
    assert report["counts"]["impressions"] == 4
    assert report["counts"]["zero_positive_impressions"] == 1
    assert report["counts"]["unavailable_positive_events"] == 2
    assert report["counts"]["cold_user_impressions"] == 2
    assert report["catalog_retrieval"]["all"]["recall@100"] == {"mean": 0.5, "impressions": 3}
    assert report["impression_ranking"]["popularity"]["all"]["auc"]["impressions"] == 3


def test_development_labels_never_change_prediction_order(database, tmp_path):
    first = evaluate(database, tmp_path / "first.json")
    with sqlite3.connect(database) as con:
        con.execute("UPDATE candidates SET clicked=1-clicked WHERE impression_id IN (SELECT impression_id FROM impressions WHERE split='development')")
    second = evaluate(database, tmp_path / "second.json")
    assert first["prediction_order_sha256"] == second["prediction_order_sha256"]
    assert first["counts"]["positives"] != second["counts"]["positives"]


def test_candidate_order_does_not_change_ranking(database, tmp_path):
    first = evaluate(database, tmp_path / "first.json")
    with sqlite3.connect(database) as con:
        con.execute("UPDATE candidates SET position=100-position")
    second = evaluate(database, tmp_path / "second.json")
    assert first["prediction_order_sha256"] == second["prediction_order_sha256"]
    assert first["impression_ranking"] == second["impression_ranking"]


def test_evidence_cannot_be_overwritten(database, tmp_path):
    path = tmp_path / "evidence.json"
    evaluate(database, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluate(database, path)
    assert path.read_bytes() == before


def test_row_order_does_not_change_first_seen_or_fit(source, tmp_path):
    news, behaviors, cutoff = source
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    ingest(news, behaviors, first, cutoff)
    behaviors.write_text("\n".join(reversed(behaviors.read_text().splitlines())) + "\n")
    ingest(news, behaviors, second, cutoff)
    one = evaluate(first, tmp_path / "one.json")
    two = evaluate(second, tmp_path / "two.json")
    assert one["prediction_order_sha256"] == two["prediction_order_sha256"]
    assert one["catalog_retrieval"] == two["catalog_retrieval"]


def test_command_line_runs_without_opening_adjacent_holdout(database, tmp_path):
    forbidden = tmp_path / "MINDsmall_dev.zip"
    forbidden.write_bytes(b"sealed final archive: intentionally not a valid zip")
    output = tmp_path / "cli.json"
    result = subprocess.run([sys.executable, "-m", "newsrec.cli", "evaluate", "--database", str(database), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["counts"]["impressions"] == 4
    assert forbidden.read_bytes().startswith(b"sealed final")
