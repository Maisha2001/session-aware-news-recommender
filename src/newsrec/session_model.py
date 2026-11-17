import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')

import numpy as np
import tensorflow as tf


class SessionRanker(tf.keras.Model):
    def __init__(self, vectors, use_session=True):
        super().__init__()
        self.vectors=tf.constant(vectors,dtype=tf.float32)
        self.use_session=use_session
        self.gru=tf.keras.layers.GRU(32,name='session_gru')
        self.session_projection=tf.keras.layers.Dense(64,activation='tanh',use_bias=False)
        self.hidden=tf.keras.layers.Dense(64,activation='relu')
        self.combination=tf.keras.layers.Dense(16,activation='relu')
        self.output_score=tf.keras.layers.Dense(1)

    def encode(self, long, events, event_mask):
        context=self.session_projection(self.gru(events,mask=event_mask))
        present=tf.cast(tf.reduce_any(event_mask,axis=1),tf.float32)[:,None]
        if not self.use_session:
            context=tf.zeros_like(context)
            present=tf.zeros_like(present)
        return long,context,present

    def score(self, context, owners, candidates, ages):
        long,session,present=[tf.gather(value,owners) for value in context]
        article=tf.gather(self.vectors,candidates)
        features=tf.concat([article,long,session,article*long,article*session,present,ages[:,None]],axis=1)
        return tf.squeeze(self.output_score(self.combination(self.hidden(features))),axis=1)

    def call(self, inputs):
        context=self.encode(inputs['long'],inputs['events'],inputs['event_mask'])
        candidates=inputs['candidate_ids']
        owners=tf.repeat(tf.range(tf.shape(candidates)[0]),tf.shape(candidates)[1])
        scores=self.score(context,owners,tf.reshape(candidates,[-1]),tf.reshape(inputs['ages'],[-1]))
        return tf.reshape(scores,tf.shape(candidates))


def training_examples(data, seed=20261117, negatives=8):
    rows, candidates=[],[]
    skipped=0
    for row in np.flatnonzero(data.arrays['fit']):
        ids,labels=data.items(row)
        negative=ids[labels==0]
        if not len(negative):
            skipped+=1
            continue
        rng=np.random.default_rng(seed+int(row))
        for target in ids[labels==1]:
            sampled=rng.choice(negative,size=negatives,replace=len(negative)<negatives)
            rows.append(row);candidates.append([target]+sampled.tolist())
    return np.asarray(rows,dtype=np.int32),np.asarray(candidates,dtype=np.int32),skipped
