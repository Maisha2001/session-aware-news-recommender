from datetime import datetime
import pytest

from newsrec.data import ingest


@pytest.fixture
def source(tmp_path):
    news = tmp_path / "news.tsv"
    behaviors = tmp_path / "behaviors.tsv"
    # Authored fixtures contain no benchmark text, users, or labels.
    news.write_text("".join(f"N{i}\tcategory\tsubcategory\tAuthored title {i}\tAuthored abstract\t\t[]\t[]\n" for i in range(1, 7)), encoding="utf-8")
    rows = [
        "1\tU1\t11/11/2019 08:00:00 AM\tN3\tN1-1 N2-0",
        "2\tU2\t11/12/2019 08:00:00 AM\t\tN1-1 N2-0",
        "3\tU1\t11/13/2019 08:00:00 AM\tN1 N3\tN2-0 N1-1 N4-1",
        "4\tU3\t11/13/2019 08:00:00 AM\t\tN4-1 N3-0",
        "5\tU3\t11/13/2019 09:00:00 AM\tN4\tN4-1 N1-0",
        "6\tU1\t11/14/2019 08:00:00 AM\tN1\tN1-0 N2-0",
    ]
    behaviors.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return news, behaviors, datetime(2019, 11, 13)


@pytest.fixture
def database(source, tmp_path):
    news, behaviors, cutoff = source
    output = tmp_path / "fixture.sqlite"
    ingest(news, behaviors, output, cutoff)
    return output
