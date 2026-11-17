import os

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


class ContentRetriever(tf.keras.Model):
    def __init__(self, token_table, buckets=32768, dimension=64, temperature=0.1):
        super().__init__()
        self.token_table = tf.constant(token_table, dtype=tf.int32)
        self.words = tf.keras.layers.Embedding(buckets, dimension, name='word_embedding')
        self.article_projection = tf.keras.layers.Dense(dimension, activation='tanh', use_bias=False, name='article_projection')
        self.query_projection = tf.keras.layers.Dense(dimension, use_bias=False, kernel_initializer='identity', name='query_projection')
        self.empty_query = self.add_weight(name='empty_query', shape=(dimension,), initializer='random_normal')
        self.task = tfrs.tasks.Retrieval(temperature=temperature, remove_accidental_hits=True)

    def articles_from_tokens(self, tokens):
        mask = tf.cast(tokens != 0, tf.float32)[..., None]
        embedded = self.words(tokens)
        pooled = tf.math.divide_no_nan(tf.reduce_sum(embedded * mask, axis=-2), tf.reduce_sum(mask, axis=-2))
        return tf.math.l2_normalize(self.article_projection(pooled), axis=-1)

    def articles(self, indices):
        values = self.articles_from_tokens(tf.gather(self.token_table, indices))
        return values * tf.cast(indices != 0, tf.float32)[..., None]

    def call(self, histories):
        mask = tf.cast(histories != 0, tf.float32)[..., None]
        values = self.articles(histories)
        mean = tf.math.divide_no_nan(tf.reduce_sum(values * mask, axis=1), tf.reduce_sum(mask, axis=1))
        mean = tf.where(tf.reduce_sum(mask, axis=1) > 0, mean, self.empty_query[None, :])
        return tf.math.l2_normalize(self.query_projection(mean), axis=-1)

    def retrieval_loss(self, histories, targets):
        queries = self(histories)
        candidates = self.articles(targets)
        return self.task(queries, candidates, candidate_ids=targets, compute_metrics=False,
                         compute_batch_metrics=False) / tf.cast(tf.shape(targets)[0], tf.float32)


def article_embeddings(model, count, batch_size=512):
    return np.concatenate([model.articles(tf.range(start, min(count, start + batch_size))).numpy()
                           for start in range(1, count, batch_size)]).astype(np.float32)
