from datetime import datetime, timezone
import json
import sqlite3
import pytest

from newsrec.data import connect_readonly, ingest, parse_behavior, timestamp


def test_empty_history_and_midnight():
    row = parse_behavior("23\tU7\t11/13/2019 12:00:00 AM\t\tN2-0 N1-1\n")
    assert row.time == datetime(2019, 11, 13)
    assert row.history == () and row.labels == (0, 1)


@pytest.mark.parametrize("line", [
    "1\tU1\t11/13/2019 12:00:00 AM\t\tN1-1 N1-0",
    "1\tU1\t11/13/2019 12:00:00 AM\tNbad\tN1-1",
    "1\tU1\t11/13/2019 12:00:00 AM\t\tN1",
    "1\tU1\t11/13/2019 12:00:00 AM\t\tN1-2",
    "1\tU1\t11/13/2019 12:00:00 AM\t\t",
    "x\tU1\t11/13/2019 12:00:00 AM\t\tN1-1",
    "1\tbad\t11/13/2019 12:00:00 AM\t\tN1-1",
    "wrong\tcolumns",
])
def test_malformed_rows_rejected(line):
    with pytest.raises(ValueError):
        parse_behavior(line)


def test_timezone_is_not_invented():
    with pytest.raises(ValueError, match="timezone"):
        timestamp(datetime(2019, 11, 13, tzinfo=timezone.utc))


def test_split_and_preserved_history(database):
    with connect_readonly(database) as con:
        assert dict(con.execute("SELECT split,count(*) FROM impressions GROUP BY split")) == {"fit": 2, "development": 4}
        assert json.loads(con.execute("SELECT history_json FROM impressions WHERE impression_id='3'").fetchone()[0]) == ["N1", "N3"]
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM impressions")


def test_cutoff_is_strict(source, tmp_path):
    news, behaviors, cutoff = source
    behaviors.write_text("1\tU1\t11/12/2019 11:59:59 PM\t\tN1-1\n2\tU1\t11/13/2019 12:00:00 AM\t\tN2-1\n")
    path = tmp_path / "boundary.sqlite"
    result = ingest(news, behaviors, path, cutoff)
    assert result["partition_counts"] == {"fit": 1, "development": 1}


def test_duplicate_ids_fail_atomically(source, tmp_path):
    news, behaviors, cutoff = source
    with behaviors.open("a") as out:
        out.write(behaviors.read_text().splitlines()[0] + "\n")
    path = tmp_path / "duplicates.sqlite"
    with pytest.raises(sqlite3.IntegrityError):
        ingest(news, behaviors, path, cutoff)
    assert not path.exists() and not path.with_name(path.name + ".building").exists()


def test_existing_database_is_preserved(source, database):
    before = database.read_bytes()
    with pytest.raises(FileExistsError):
        ingest(source[0], source[1], database, source[2])
    assert database.read_bytes() == before


def test_no_empty_partition(source, tmp_path):
    with pytest.raises(ValueError, match="Both"):
        ingest(source[0], source[1], tmp_path / "invalid.sqlite", datetime(2018, 1, 1))


def test_missing_metadata_counted_without_dropping_outcomes(source, tmp_path):
    news, behaviors, cutoff = source
    behaviors.write_text(behaviors.read_text().replace("N4", "N999"))
    result = ingest(news, behaviors, tmp_path / "missing.sqlite", cutoff)
    assert result["observed_ids_without_metadata"] == 1
    assert sum(result["partition_counts"].values()) == 6


def test_existing_incomplete_build_is_preserved(source, tmp_path):
    path = tmp_path / "busy.sqlite"
    partial = path.with_name(path.name + ".building")
    partial.write_bytes(b"another active build")
    with pytest.raises(FileExistsError):
        ingest(source[0], source[1], path, source[2])
    assert partial.read_bytes() == b"another active build"
