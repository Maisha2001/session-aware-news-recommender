from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import sqlite3


NEWS_ID = re.compile(r"N[0-9]+")
USER_ID = re.compile(r"U[0-9]+")


@dataclass(frozen=True)
class Impression:
    impression_id: str
    user_id: str
    time: datetime
    history: tuple[str, ...]
    candidates: tuple[str, ...]
    labels: tuple[int, ...]


def parse_behavior(line: str) -> Impression:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 5:
        raise ValueError("Expected five MIND behavior columns")
    iid, uid, raw_time, raw_history, raw_candidates = fields
    if not iid.isdigit() or not USER_ID.fullmatch(uid):
        raise ValueError("Invalid impression or user ID")
    time = datetime.strptime(raw_time, "%m/%d/%Y %I:%M:%S %p")
    history = tuple(raw_history.split())
    if any(not NEWS_ID.fullmatch(a) for a in history):
        raise ValueError("Invalid history article ID")
    candidates, labels = [], []
    for candidate in raw_candidates.split():
        article, separator, label = candidate.rpartition("-")
        if not separator or not NEWS_ID.fullmatch(article) or label not in ("0", "1"):
            raise ValueError("Expected labeled article candidates")
        candidates.append(article)
        labels.append(int(label))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Empty or duplicate impression candidates")
    return Impression(iid, uid, time, history, tuple(candidates), tuple(labels))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp(value: datetime) -> str:
    if value.tzinfo is not None:
        raise ValueError("MIND timestamps have no published timezone; use naive dataset time")
    return value.isoformat(timespec="seconds")


def connect_readonly(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def ingest(news_path: Path, behaviors_path: Path, output: Path, cutoff: datetime) -> dict:
    split_at = timestamp(cutoff)
    if output.exists():
        raise FileExistsError("Refusing to replace an existing database")
    if news_path.name != "news.tsv" or behaviors_path.name != "behaviors.tsv":
        raise ValueError("Pass the explicit training news.tsv and behaviors.tsv files")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError("Incomplete build exists; choose a new output path")
    staging.touch(exist_ok=False)
    con = sqlite3.connect(staging)
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE articles (
                news_id TEXT PRIMARY KEY, category TEXT, subcategory TEXT,
                title TEXT, abstract TEXT);
            CREATE TABLE impressions (
                impression_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL, split TEXT NOT NULL CHECK(split IN ('fit','development')),
                history_json TEXT NOT NULL);
            CREATE TABLE candidates (
                impression_id TEXT REFERENCES impressions(impression_id),
                position INTEGER NOT NULL, news_id TEXT NOT NULL,
                clicked INTEGER NOT NULL CHECK(clicked IN (0,1)),
                PRIMARY KEY(impression_id,news_id));
            CREATE TABLE observations (news_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL);
        """)
        with news_path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 8 or not NEWS_ID.fullmatch(fields[0]):
                    raise ValueError(f"Invalid article row {number}")
                con.execute("INSERT INTO articles VALUES (?,?,?,?,?)", fields[:5])
        counts = {"fit": 0, "development": 0}
        with behaviors_path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = parse_behavior(line)
                except ValueError as error:
                    raise ValueError(f"Invalid behavior row {number}: {error}") from error
                time = timestamp(row.time)
                split = "fit" if time < split_at else "development"
                counts[split] += 1
                con.execute("INSERT INTO impressions VALUES (?,?,?,?,?)",
                            (row.impression_id, row.user_id, time, split, json.dumps(row.history)))
                con.executemany("INSERT INTO candidates VALUES (?,?,?,?)",
                                [(row.impression_id, i, a, y) for i, (a, y) in enumerate(zip(row.candidates, row.labels))])
                con.executemany("""INSERT INTO observations VALUES (?,?)
                    ON CONFLICT(news_id) DO UPDATE SET first_seen=min(first_seen,excluded.first_seen)""",
                    [(a, time) for a in set(row.history + row.candidates)])
        if not all(counts.values()):
            raise ValueError("Both chronological partitions must contain impressions")
        con.executescript("""
            CREATE INDEX impression_time ON impressions(occurred_at, impression_id);
            CREATE INDEX impression_split ON impressions(split);
            CREATE INDEX candidate_article ON candidates(news_id);
        """)
        meta = {
            "format_version": 1, "dataset": "MIND-format files; archive origin requires input-hash verification",
            "cutoff": split_at, "timezone": "unspecified by dataset; naive timestamps retained",
            "partition_counts": counts,
            "input_sha256": {"news.tsv": file_sha256(news_path), "behaviors.tsv": file_sha256(behaviors_path)},
            "holdout_provenance": "Not inferred from filenames; verify input hashes against the training archive.",
            "history_policy": "Provided pre-impression history is preserved, not appended to popularity counts.",
            "catalog_policy": "Only IDs observed strictly before each query time are retrievable; same-time observations excluded.",
        }
        meta["articles"] = con.execute("SELECT count(*) FROM articles").fetchone()[0]
        meta["candidates"] = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
        meta["observed_ids_without_metadata"] = con.execute(
            "SELECT count(*) FROM observations LEFT JOIN articles USING(news_id) WHERE articles.news_id IS NULL").fetchone()[0]
        meta["date_range"] = list(con.execute("SELECT min(occurred_at),max(occurred_at) FROM impressions").fetchone())
        con.execute("INSERT INTO metadata VALUES ('provenance',?)", (json.dumps(meta, sort_keys=True),))
        con.commit()
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Database integrity check failed")
        con.close()
        staging.replace(output)
        return meta
    except BaseException:
        con.close()
        staging.unlink(missing_ok=True)
        raise
