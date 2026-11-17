from datetime import datetime
import json
from pathlib import Path
import sqlite3

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('tensorflow')
pytest.importorskip('faiss')

from newsrec.data import ingest
from newsrec.retrieval_data import prepare_content, text_tokens
from newsrec.retrieval_index import TemporalIndex
from newsrec.retrieval_model import ContentRetriever, tf


def test_tokens_are_stable_bounded_and_ignore_other_articles():
    one = text_tokens('A NEW headline!', 'Tech', 'Devices', 128, 8)
    assert one == text_tokens('a new headline', 'tech', 'devices', 128, 8)
    assert len(one) == 8 and all(0 <= x < 128 for x in one)
    assert one[-1] == 0


def test_training_examples_do_not_read_development_labels(database):
    before = prepare_content(database)
    with sqlite3.connect(database) as con:
        con.execute("UPDATE candidates SET clicked=1-clicked WHERE impression_id IN (SELECT impression_id FROM impressions WHERE split='development')")
    after = prepare_content(database)
    np.testing.assert_array_equal(before.fit_histories, after.fit_histories)
    np.testing.assert_array_equal(before.fit_targets, after.fit_targets)
    np.testing.assert_array_equal(before.tokens, after.tokens)
    assert len(before.fit_targets) == 2
    assert before.mapping['N3'] in before.fit_content
    assert before.mapping['N4'] not in before.fit_content


def test_same_time_future_and_boundary_ties_are_filtered(tmp_path):
    vectors = np.asarray([[1, 0]] * 300, dtype=np.float32)
    index = TemporalIndex(vectors, np.asarray([0] * 298 + [1, 2]), [f'N{i}' for i in range(300)])
    actual = index.search(np.asarray([[1,0]], dtype=np.float32), [1], k=10)[0]
    expected = sorted(range(298), key=lambda i:index.ties[i])[:10]
    assert [i - 1 for i, _ in actual] == expected
    assert index.search(np.asarray([[1,0]], dtype=np.float32), [0], k=10) == [[]]
    index.save(tmp_path / 'index')
    loaded = TemporalIndex.load(tmp_path / 'index')
    assert loaded.search(np.asarray([[1,0]], dtype=np.float32), [1], k=10) == [actual]
    with (tmp_path / 'index/articles.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash'):
        TemporalIndex.load(tmp_path / 'index')


def test_index_matches_exhaustive_numpy():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(500, 16)).astype(np.float32)
    queries = rng.normal(size=(12,16)).astype(np.float32)
    seen = rng.integers(0,10,size=500)
    index = TemporalIndex(vectors, seen, [f'N{i}' for i in range(500)])
    actual = index.search(queries, np.arange(12), k=20)
    for q, result in enumerate(actual):
        scores = vectors @ queries[q]
        expected = sorted(np.flatnonzero(seen < q), key=lambda i:(-scores[i],index.ties[i]))[:20]
        assert [i-1 for i,_ in result] == expected
        np.testing.assert_allclose([s for _,s in result], scores[expected], atol=3e-5)


def test_new_article_encoding_padding_and_weight_reload(database, tmp_path):
    data = prepare_content(database)
    tf.keras.utils.set_random_seed(23)
    model = ContentRetriever(data.tokens, dimension=8)
    h = np.asarray([[data.mapping['N1'],0,0],[0,0,0]],dtype=np.int32)
    first = model(h).numpy()
    assert np.isfinite(first).all()
    np.testing.assert_allclose(model(np.asarray([[data.mapping['N1']]],dtype=np.int32)).numpy(), first[:1], atol=1e-6)
    new = np.asarray([text_tokens('Authored unseen news', 'science', 'space')],dtype=np.int32)
    assert model.articles_from_tokens(new).shape == (1,8)
    assert np.linalg.norm(model.articles_from_tokens(new).numpy()) > 0.99
    model.save_weights(str(tmp_path/'model.weights.h5'))
    other = ContentRetriever(data.tokens, dimension=8)
    other(h)
    other.load_weights(str(tmp_path/'model.weights.h5'))
    np.testing.assert_array_equal(first,other(h).numpy())


def test_accidental_duplicate_targets_do_not_become_negatives(database):
    data = prepare_content(database)
    model = ContentRetriever(data.tokens,dimension=8)
    histories = np.asarray([[data.mapping['N1']],[data.mapping['N1']]],dtype=np.int32)
    targets = np.asarray([data.mapping['N1']]*2,dtype=np.int32)
    assert float(model.retrieval_loss(histories,targets)) == pytest.approx(0,abs=1e-6)


def test_gradient_updates_learnable_content_parameters(database):
    data = prepare_content(database)
    tf.keras.utils.set_random_seed(1)
    model = ContentRetriever(data.tokens,dimension=8)
    histories = np.asarray([[data.mapping['N1']],[data.mapping['N2']]],dtype=np.int32)
    targets = np.asarray([data.mapping['N2'],data.mapping['N1']],dtype=np.int32)
    with tf.GradientTape() as tape:
        loss = model.retrieval_loss(histories,targets)
    gradients = tape.gradient(loss,model.trainable_variables)
    assert np.isfinite(float(loss))
    before = model.words.get_weights()[0].copy()
    tf.keras.optimizers.Adam(0.01).apply_gradients(zip(gradients,model.trainable_variables))
    assert np.max(np.abs(before-model.words.get_weights()[0])) > 0


def test_catalog_denominator_retains_unavailable_positives():
    from newsrec.evaluate_retrieval import catalog_metrics
    result = catalog_metrics([1, 2], {1, 3})
    assert result['recall@100'] == 0.5
    assert result['ndcg@100'] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert catalog_metrics([], set()) == {}


def test_train_evaluate_and_independent_audit(database, source, tmp_path):
    from newsrec.train_retrieval import train
    from newsrec.evaluate_retrieval import evaluate
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
    from audit_retrieval import audit
    model_dir, result_dir = tmp_path/'model', tmp_path/'result'
    trained = train(database, model_dir, epochs=1, batch_size=2, dimension=8)
    assert trained['epochs'][0]['examples'] == 2
    report = evaluate(database, model_dir, result_dir, batch_size=2, audit_queries=2)
    for arm in report['arms'].values():
        assert arm['counts']['all_impressions'] == 4
        assert arm['counts']['zero_positive_impressions'] == 1
        assert arm['audit']['reload_predictions_bitwise_equal']
        assert arm['audit']['exhaustive_float64_queries'] == 2
    checked = audit(source[1], result_dir/'report.json', tmp_path/'audit.json')
    assert checked['status'] == 'passed'
    with pytest.raises(FileExistsError):
        train(database, model_dir)
