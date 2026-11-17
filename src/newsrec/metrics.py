import math


def ranking_metrics(labels: list[int], ks: tuple[int, ...] = (5, 10)) -> dict:
    if not labels or any(y not in (0, 1) for y in labels):
        raise ValueError("Expected nonempty binary relevance labels in ranked order")
    if any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("k must be a positive integer")
    positives = sum(labels)
    if positives == 0:
        return {"positive_count": 0, "mean_clicked_rr": None, "first_hit_rr": None,
                **{f"ndcg@{k}": None for k in ks}, **{f"recall@{k}": None for k in ks}}
    result = {"positive_count": positives,
              "mean_clicked_rr": sum(y / (i + 1) for i, y in enumerate(labels)) / positives,
              "first_hit_rr": 1 / (labels.index(1) + 1)}
    for k in ks:
        if k <= 0:
            raise ValueError("k must be positive")
        dcg = sum(y / math.log2(i + 2) for i, y in enumerate(labels[:k]))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, positives)))
        result[f"ndcg@{k}"] = dcg / ideal
        result[f"recall@{k}"] = sum(labels[:k]) / positives
    return result


def auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels or any(y not in (0, 1) for y in labels):
        raise ValueError("Expected equal nonempty binary labels and finite scores")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    below, wins, i = 0, 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        pos = sum(y for _, y in ordered[i:j])
        neg = j - i - pos
        wins += pos * (below + 0.5 * neg)
        below += neg
        i = j
    return wins / (positives * negatives)


class MetricMeans:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, metrics: dict) -> None:
        for key, value in metrics.items():
            if value is not None:
                self.sums[key] = self.sums.get(key, 0.0) + value
                self.counts[key] = self.counts.get(key, 0) + 1

    def report(self) -> dict:
        return {key: {"mean": value / self.counts[key], "impressions": self.counts[key]}
                for key, value in sorted(self.sums.items())}
