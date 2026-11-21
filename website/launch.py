"""Start/stop the local website outside a temporary terminal session."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
import uuid

ROOT=Path(__file__).resolve().parents[1]
SERVICES=ROOT/'website/.services'


def health(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=2) as response:
            value=json.loads(response.read(1_000_001))
        return value if isinstance(value,dict) and value.get('application')=='pramaan' else None
    except (OSError,ValueError):return None


def start(port):
    address=f'http://127.0.0.1:{port}'
    if health(port):print(address);return
    SERVICES.mkdir(parents=True,exist_ok=True)
    identity=uuid.uuid4().hex
    log_path=SERVICES/f'website-{port}.log'
    with log_path.open('ab') as stream:
        process=subprocess.Popen([sys.executable,str(ROOT/'website/server.py'),'--port',str(port),'--instance-id',identity],
                                 cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+35
    while time.monotonic()<deadline:
        ready=health(port)
        if ready and ready.get('instance_id')==identity:
            (SERVICES/f'website-{port}.json').write_text(json.dumps({'pid':process.pid,'instance_id':identity,'port':port})+'\n')
            print(address);return
        if process.poll() is not None:raise SystemExit('Website failed to start. See '+str(log_path))
        time.sleep(.2)
    raise SystemExit('Website startup is still pending. See '+str(log_path))


def stop(port):
    record=SERVICES/f'website-{port}.json'
    try:owner=json.loads(record.read_text())
    except (OSError,ValueError):raise SystemExit('No managed website instance was found. Stop a foreground server with Ctrl+C.')
    ready=health(port)
    if not ready:print('Website is already stopped.');return
    if ready.get('instance_id')!=owner.get('instance_id'):
        raise SystemExit('This port belongs to another instance; it was not stopped.')
    os.kill(owner['pid'],signal.SIGTERM)
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        if not health(port):print('Website stopped.');return
        time.sleep(.2)
    raise SystemExit('Website shutdown is still finishing. Check its local log.')


def main():
    parser=argparse.ArgumentParser(description='Manage the Pramaan local website and its local LLM')
    parser.add_argument('action',choices=['start','stop','status'],nargs='?',default='start')
    parser.add_argument('--port',type=int,default=8790)
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('Invalid port')
    if args.action=='start':start(args.port)
    elif args.action=='stop':stop(args.port)
    else:
        ready=health(args.port)
        print(json.dumps({'running':bool(ready),'local_model_ready':bool(ready and ready.get('ollama')),
                          'default_provider':ready.get('default_provider') if ready else None},indent=2))


if __name__=='__main__':main()
