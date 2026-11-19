"""Record already frozen evidence in a local-only MLflow store."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .data import file_sha256


def log_release(store, report, audit, protocol, models, artifacts, receipt):
    if str(store).startswith(('\\\\','//')) or '://' in str(store):
        raise ValueError('Tracking requires a local filesystem directory')
    store, report, audit, protocol, receipt = map(Path,(store,report,audit,protocol,receipt))
    if receipt.exists():
        raise FileExistsError('A tracking receipt exists; inspect its run rather than logging a duplicate')
    result=json.loads(report.read_text()); verification=json.loads(audit.read_text())
    if verification['status']!='passed' or verification['report_sha256']!=file_sha256(report):
        raise ValueError('Independent audit does not match this report')
    if set(models)!={'content','long_only','session'}:
        raise ValueError('All three frozen learned models are required')
    model_paths={name:Path(path) for name,path in models.items()}
    for name,path in model_paths.items():
        if file_sha256(path)!=result['prediction_manifest']['model_sha256'][name]:
            raise ValueError('Model hash mismatch: '+name)
    store=store.resolve();store.mkdir(parents=True,exist_ok=True)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    uri='sqlite:///'+(store/'tracking.sqlite').as_posix()
    os.environ['MLFLOW_DISABLE_TELEMETRY']='true'
    os.environ['DO_NOT_TRACK']='true'
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param
    client=MlflowClient(tracking_uri=uri,registry_uri=uri)
    name='Session-Aware News Recommendation'
    experiment=client.get_experiment_by_name(name)
    experiment_id=experiment.experiment_id if experiment else client.create_experiment(name,artifact_location=(store/'artifacts').as_uri())
    run=client.create_run(experiment_id,tags={'mlflow.runName':'Frozen chronological release','evaluation':'heldout-v1',
        'scope':'Local research; no live engagement claim','training_performed':'false','report_sha256':file_sha256(report)})
    record={'run_id':run.info.run_id,'status':'RUNNING','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'tracking_mode':'local SQLite and filesystem artifacts; no cloud API','training_performed':False}
    with receipt.open('x') as stream:json.dump(record,stream,indent=2)
    try:
        metrics={}
        for root in ('logged_ranking','catalog_ranking'):
            for arm,groups in result[root].items():
                for segment,values in groups.items():
                    for key,value in values.items():metrics[f'{root}.{arm}.{segment}.{key}'.replace('@','_at_')]=float(value['mean'])
        metrics.update({'count.'+key:float(value) for key,value in result['counts'].items()})
        params={'report_sha256':file_sha256(report),'audit_sha256':file_sha256(audit),'protocol_sha256':file_sha256(protocol),
                'feedback_delay_seconds':'300','session_gap_seconds':'1800','model_epochs':'3','selection':'frozen before final outcomes'}
        params.update({name+'_weights_sha256':file_sha256(path) for name,path in model_paths.items()})
        stamp=int(time.time()*1000);entries=[Metric(key,value,stamp,0) for key,value in metrics.items()]
        for offset in range(0,len(entries),500):client.log_batch(run.info.run_id,metrics=entries[offset:offset+500])
        client.log_batch(run.info.run_id,params=[Param(key,value) for key,value in params.items()])
        files={f'models/{name}.weights.h5':path for name,path in model_paths.items()}
        files.update({'evidence/report.json':report,'evidence/audit.json':audit,'evidence/protocol.json':protocol})
        for name,path in artifacts.items():
            if not name or '..' in Path(name).parts or Path(name).is_absolute() or name in files:
                raise ValueError('Invalid or duplicate artifact name')
            files[name]=Path(path)
        staging=store/('upload-'+run.info.run_id);staging.mkdir()
        for name,path in files.items():
            target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
            client.log_artifact(run.info.run_id,str(target),artifact_path=Path(name).parent.as_posix())
        verified=client.get_run(run.info.run_id)
        if verified.data.metrics!=metrics or verified.data.params!=params:
            raise RuntimeError('Tracking metric/parameter verification failed')
        downloads=store/('verify-'+run.info.run_id);downloads.mkdir()
        hashes={}
        for name,path in files.items():
            copied=Path(client.download_artifacts(run.info.run_id,name,str(downloads)))
            hashes[name]=file_sha256(path)
            if file_sha256(copied)!=hashes[name]:raise RuntimeError('Tracking artifact hash mismatch: '+name)
        client.set_terminated(run.info.run_id,'FINISHED')
        record.update(status=client.get_run(run.info.run_id).info.status,metrics_verified=len(metrics),parameters_verified=len(params),
                      artifact_sha256=hashes,report_sha256=file_sha256(report),completed_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException:
        client.set_terminated(run.info.run_id,'FAILED')
        record.update(status='FAILED',completed_at_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(record,indent=2)+'\n');raise
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('store','report','audit','protocol','models','receipt'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--artifacts',type=Path,help='JSON mapping safe artifact names to local paths')
    args=parser.parse_args()
    result=log_release(args.store,args.report,args.audit,args.protocol,json.loads(args.models.read_text()),
                       json.loads(args.artifacts.read_text()) if args.artifacts else {},args.receipt)
    print(json.dumps({'run_id':result['run_id'],'status':result['status'],'metrics_verified':result['metrics_verified'],'artifacts_verified':len(result['artifact_sha256'])}))
