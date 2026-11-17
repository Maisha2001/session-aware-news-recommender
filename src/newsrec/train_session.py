import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .data import file_sha256
from .session_data import load_session, candidate_ages
from .session_model import SessionRanker, training_examples, tf
from .train_retrieval import write_json


def inputs_for(data, rows, candidates):
    return {'long':np.asarray(data.arrays['long'][rows]),'events':np.asarray(data.arrays['events'][rows]),
            'event_mask':np.asarray(data.arrays['event_mask'][rows]),'candidate_ids':candidates,'ages':candidate_ages(data,rows,candidates)}


def train(data_directory, directory, epochs=3, batch_size=128, seed=20261117):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter()
    data=load_session(data_directory)
    rows,candidates,skipped=training_examples(data,seed)
    if not len(rows):raise ValueError('No fitting examples with logged negatives')
    config={'epochs':epochs,'batch_size':batch_size,'seed':seed,'sampled_logged_negatives':8,
            'objective':'Sampled within-impression softmax; all fitting positives, eight logged nonclicks each; replacement only when necessary. Logits are uncalibrated.',
            'fit_examples':len(rows),'fitting_impressions_without_negatives':skipped,'gru_units':32,'hidden_units':[64,16],
            'learning_rate':0.001,'clipnorm':1.0,'content_encoder_frozen':True,'selection':'Fixed final epoch; no development stopping/tuning.',
            'data_manifest_sha256':file_sha256(Path(data_directory)/'manifest.json'),
            'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ('train_session.py','session_model.py','session_data.py')}}
    write_json(directory/'config.json',config)
    reports={}
    for arm,use_session in [('long_only',False),('session',True)]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        model=SessionRanker(data.arrays['vectors'],use_session)
        model(inputs_for(data,rows[:1],candidates[:1]))
        optimizer=tf.keras.optimizers.Adam(0.001,clipnorm=1.0)
        checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)

        @tf.function(reduce_retracing=True)
        def step(inputs):
            with tf.GradientTape() as tape:
                logits=model(inputs)
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros(tf.shape(logits)[0],tf.int32),logits=logits))
            gradients=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients([(g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None])
            return loss

        history=[]
        arm_start=time.perf_counter()
        for epoch in range(epochs):
            start=time.perf_counter()
            order=np.random.default_rng(seed+epoch).permutation(len(rows))
            total=0.0
            for offset in range(0,len(order),batch_size):
                selected=order[offset:offset+batch_size]
                loss=float(step(inputs_for(data,rows[selected],candidates[selected])))
                if not np.isfinite(loss):raise RuntimeError('Nonfinite ranking loss')
                total+=loss*len(selected)
            record={'epoch':epoch+1,'mean_loss':total/len(rows),'wall_seconds':time.perf_counter()-start,
                    'completed_at_utc':datetime.now(timezone.utc).isoformat()}
            model.save_weights(str(directory/f'{arm}-epoch-{epoch+1}.weights.h5'))
            checkpoint.write(str(directory/f'{arm}-epoch-{epoch+1}.checkpoint'))
            write_json(directory/f'{arm}-epoch-{epoch+1}.json',record)
            history.append(record)
            print(json.dumps({'arm':arm,**record}),flush=True)
        reports[arm]={'epochs':history,'parameters':model.count_params(),'wall_seconds':time.perf_counter()-arm_start,
                      'weights_sha256':file_sha256(directory/f'{arm}-epoch-{epochs}.weights.h5')}
    result={'config':config,'arms':reports,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'wall_seconds':time.perf_counter()-begin,'final_holdout_opened':False,'development_scores_used_for_selection':False}
    write_json(directory/'training.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Fit long-term and delayed-session ranking models on logged fitting impressions.')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(train(args.data,args.output),indent=2))
