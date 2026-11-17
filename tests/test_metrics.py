import math
import random
import pytest
from newsrec.metrics import MetricMeans, auc, ranking_metrics


def test_known_ranking():
    result = ranking_metrics([0, 1, 0, 1], (1, 2, 4))
    assert result["mean_clicked_rr"] == pytest.approx(0.375)
    assert result["first_hit_rr"] == 0.5
    assert result["recall@1"] == 0
    assert result["recall@2"] == 0.5
    assert result["recall@4"] == 1
    assert result["ndcg@2"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_degenerate_groups_have_explicit_undefined_metrics():
    assert ranking_metrics([0, 0])["ndcg@5"] is None
    assert auc([1, 1], [2, 3]) is None
    assert auc([0, 0], [2, 3]) is None
    assert auc([0, 1], [4, 4]) == 0.5
    assert ranking_metrics([1, 1])["ndcg@5"] == 1


def test_auc_matches_independent_pairwise_calculation():
    rng = random.Random(31)
    for size in range(2, 30):
        labels = [0, 1] + [rng.randrange(2) for _ in range(size - 2)]
        scores = [rng.randrange(4) for _ in labels]
        positives = [s for s, y in zip(scores, labels) if y]
        negatives = [s for s, y in zip(scores, labels) if not y]
        expected = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))
        assert auc(labels, scores) == pytest.approx(expected)


@pytest.mark.parametrize("labels,scores", [([], []), ([1], []), ([2], [1]), ([0, 1], [1, float("nan")])])
def test_auc_rejects_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        auc(labels, scores)


def test_means_record_different_denominators():
    means = MetricMeans()
    means.add({"auc": None, "recall": 1})
    means.add({"auc": 0.5, "recall": 0})
    assert means.report() == {"auc": {"mean": 0.5, "impressions": 1}, "recall": {"mean": 0.5, "impressions": 2}}


@pytest.mark.parametrize("labels", [[0, 0], [0, 1]])
@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_invalid_cutoffs_rejected_for_all_classes(labels, k):
    with pytest.raises(ValueError):
        ranking_metrics(labels, (k,))
