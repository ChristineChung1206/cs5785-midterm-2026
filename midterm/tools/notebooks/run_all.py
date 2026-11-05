"""Execute consolidated notebooks in saved-evidence mode without network sockets.

Usage: python tools/notebooks/run_all.py
Optional: python tools/notebooks/run_all.py 03_part2.ipynb 04_error_analysis.ipynb
Default notebooks set both RUN_TRAINING and RUN_HISTORICAL_TRAINING to False.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
NAMES=('01_data_protocol.ipynb','02_part1.ipynb','03_part2.ipynb','04_error_analysis.ipynb','05_historical_experiments.ipynb')


def execute_one(name):
    from ipykernel.inprocess.manager import InProcessKernelManager
    path=ROOT/'tools/notebooks'/name
    document=json.loads(path.read_text())
    os.chdir(path.parent)
    kernel=InProcessKernelManager();kernel.start_kernel()
    client=kernel.client();client.start_channels()

    def execute(source):
        mid=client.execute(source,store_history=True)
        outputs=[];count=None;failed=False
        while True:
            message=client.get_iopub_msg(timeout=120)
            if message['parent_header'].get('msg_id')!=mid:continue
            kind=message['msg_type'];value=message['content']
            if kind=='execute_input':count=value['execution_count']
            elif kind in ('display_data','execute_result'):
                output={'output_type':kind,'data':value['data'],'metadata':value.get('metadata',{})}
                if kind=='execute_result':output['execution_count']=value['execution_count']
                outputs.append(output)
            elif kind=='stream':outputs.append({'output_type':'stream','name':value['name'],'text':value['text']})
            elif kind=='error':
                outputs.append({'output_type':'error','ename':value['ename'],'evalue':value['evalue'],'traceback':value['traceback']});failed=True
            elif kind=='status' and value['execution_state']=='idle':break
        return outputs,count,failed

    started=time.monotonic();errors=[];executed=0
    try:
        initial,_,failed=execute('%matplotlib inline')
        if failed:raise RuntimeError(initial)
        for index,cell in enumerate(document['cells']):
            if cell['cell_type']!='code':continue
            source=cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
            outputs,count,failed=execute(source)
            cell['outputs']=outputs;cell['execution_count']=count;executed+=1
            if failed:
                errors.append({'cell':index,'outputs':outputs});break
        verification={'runner':'ipykernel InProcessKernelManager; real Jupyter messages, no network sockets',
            'code_cells_executed':executed,'errors':len(errors),'training_enabled':bool(kernel.kernel.shell.user_ns.get('RUN_TRAINING',False)),
            'historical_training_enabled':bool(kernel.kernel.shell.user_ns.get('RUN_HISTORICAL_TRAINING',False)),
            'elapsed_seconds':round(time.monotonic()-started,3)}
        document['metadata']['execution_verification']=verification
        path.write_text(json.dumps(document,indent=1)+'\n')
    finally:
        client.stop_channels();kernel.shutdown_kernel()
    sys.__stdout__.write(json.dumps({'notebook':name,**verification})+'\n')
    if errors:
        sys.__stdout__.write(json.dumps(errors,indent=2)+'\n')
        return 1
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',nargs='*',choices=NAMES)
    parser.add_argument('--one',choices=NAMES)
    args=parser.parse_args()
    if not (ROOT/'models/selection.json').is_file():
        raise FileNotFoundError('Run inside the project with models/selection.json')
    os.environ.setdefault('PYDEVD_DISABLE_FILE_VALIDATION','1')
    os.environ.setdefault('IPYTHONDIR',str(Path(tempfile.gettempdir())/'midterm-ipython'))
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'midterm-report-mpl'))
    if args.one:return execute_one(args.one)
    failures=[]
    for name in args.names or NAMES:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--one',name],check=False)
        if result.returncode:failures.append(name)
    records=[]
    for name in NAMES:
        path=ROOT/'tools/notebooks'/name
        notebook=json.loads(path.read_text())
        record={'notebook':str(path.relative_to(ROOT)),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                **notebook['metadata'].get('execution_verification',{})}
        record['success']=record.get('errors')==0 and record.get('code_cells_executed')==sum(c['cell_type']=='code' for c in notebook['cells'])
        records.append(record)
    out=ROOT/'tools/verification/current';out.mkdir(parents=True,exist_ok=True)
    (out/'notebook_execution.json').write_text(json.dumps({'python':platform.python_version(),'no_training':all(not r.get('training_enabled') and not r.get('historical_training_enabled') for r in records),
        'all_successful':all(r['success'] for r in records),'notebooks':records},indent=2)+'\n')
    return int(bool(failures))


if __name__=='__main__':
    raise SystemExit(main())
