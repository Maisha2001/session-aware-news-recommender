import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .data import file_sha256
from .retrieval_data import prepare_content
from .retrieval_model import ContentRetriever, tf


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def train(database, directory, epochs=3, batch_size=256, seed=20261116, dimension=64):
    if directory.exists():
        raise FileExistsError('Use a new output directory; training evidence is not overwritten')
    if epochs < 1 or batch_size < 2 or dimension < 2:
        raise ValueError('Invalid training configuration')
    directory.mkdir(parents=True)
    start = time.perf_counter()
    data = prepare_content(database)
    config = {'epochs':epochs, 'batch_size':batch_size, 'seed':seed, 'dimension':dimension,
              'buckets':32768, 'token_length':32, 'history_length':20, 'temperature':0.1,
              'optimizer':'Adam', 'learning_rate':0.001, 'clipnorm':1.0, 'data':data.metadata,
              'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'selection':'Fixed configuration and final epoch; no development score or final labels used for fitting/stopping.',
              'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ('train_retrieval.py','retrieval_model.py','retrieval_data.py')}}
    write_json(directory / 'config.json', config)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = ContentRetriever(data.tokens, dimension=dimension)
    model(tf.constant(data.fit_histories[:1]))
    model.save_weights(str(directory / 'initial.weights.h5'))
    optimizer = tf.keras.optimizers.Adam(0.001, clipnorm=1.0)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    @tf.function(reduce_retracing=True)
    def step(histories, targets):
        with tf.GradientTape() as tape:
            loss = model.retrieval_loss(histories, targets)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    history = []
    for epoch in range(epochs):
        began = time.perf_counter()
        order = np.random.default_rng(seed + epoch).permutation(len(data.fit_targets))
        total = 0.0
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            loss = float(step(data.fit_histories[batch], data.fit_targets[batch]))
            if not np.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            total += loss * len(batch)
        row = {'epoch':epoch + 1, 'mean_loss':total / len(order), 'examples':len(order),
               'wall_seconds':time.perf_counter() - began, 'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        model.save_weights(str(directory / f'epoch-{epoch + 1}.weights.h5'))
        checkpoint.write(str(directory / f'epoch-{epoch + 1}.checkpoint'))
        write_json(directory / f'epoch-{epoch + 1}.json', row)
        history.append(row)
        print(json.dumps(row), flush=True)
    report = {'config':config, 'epochs':history, 'parameters':model.count_params(),
              'wall_seconds':time.perf_counter() - start, 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
              'weights_sha256':{p.name:file_sha256(p) for p in sorted(directory.glob('*.weights.h5'))},
              'final_holdout_opened':False, 'development_scores_used_for_selection':False}
    write_json(directory / 'training.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a fixed content retrieval model on chronological fitting positives.')
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    print(json.dumps(train(args.database, args.output, args.epochs, args.batch_size, args.seed, args.dimension), indent=2))
