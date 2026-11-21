"""Local Ollama lifecycle and readiness. No model download happens on a search."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
MODEL='qwen3:4b-instruct-2507-q4_K_M'
MODEL_DIRECTORY=ROOT/'website/.models/ollama'


def settings():
    try:
        value=json.loads((ROOT/'website/configs/local_llm.json').read_text())
    except (OSError,ValueError):
        return {'enabled':False,'model':MODEL,'base_url':'http://127.0.0.1:11434','num_ctx':8192,'num_predict':1024,'keep_alive':'5m'}
    base=value.get('base_url','http://127.0.0.1:11434').rstrip('/')
    parsed=urlsplit(base)
    if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ValueError('The local model must use a plain loopback Ollama endpoint')
    model=value.get('model',MODEL)
    if not isinstance(model,str) or not model or len(model)>160 or ':cloud' in model:
        raise ValueError('Configure a local Ollama model')
    for key,minimum,maximum in (('num_ctx',2048,16384),('num_predict',256,2048)):
        if type(value.get(key)) is not int or not minimum<=value[key]<=maximum:
            raise ValueError('Invalid local model setting: '+key)
    if value.get('keep_alive') not in {'1m','2m','5m','10m'}:
        raise ValueError('Invalid local model keep-alive setting')
    return {**value,'base_url':base,'model':model,'enabled':value.get('enabled') is True}


def environment(config=None):
    config=config or settings()
    result=dict(os.environ)
    result.update(OLLAMA_HOST=config['base_url'],OLLAMA_MODELS=str(MODEL_DIRECTORY),
                  OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1',
                  OLLAMA_CONTEXT_LENGTH=str(config['num_ctx']),OLLAMA_FLASH_ATTENTION='1',
                  OLLAMA_KV_CACHE_TYPE='q8_0',OLLAMA_NO_CLOUD='1')
    return result


def status(config=None):
    config=config or settings()
    result={'enabled':config['enabled'],'running':False,'ready':False,'model':config['model'],'base_url':config['base_url']}
    if not config['enabled']:
        return {**result,'message':'Local answer generation is disabled in the project settings.'}
    try:
        with urlopen(config['base_url']+'/api/tags',timeout=2) as response:
            data=json.loads(response.read(1_000_001))
        if not isinstance(data,dict) or not isinstance(data.get('models'),list):raise ValueError('Invalid model readiness response')
        result['running']=True
        result['ready']=any(model.get('name')==config['model'] or model.get('model')==config['model'] for model in data.get('models',[]))
    except (OSError,URLError,ValueError):
        return {**result,'message':'Local model service is offline. Restart the website or run python3 website/setup_llm.py.'}
    return {**result,'message':None if result['ready'] else 'Local model is not downloaded. Run python3 website/setup_llm.py.'}


class LocalRuntime:
    def __init__(self):
        self.process=None
        self.log=None

    def start(self,timeout=20):
        config=settings()
        if not config['enabled'] or status(config)['running']:
            return status(config)
        executable=shutil.which('ollama')
        if not executable:
            return {**status(config),'message':'Install Ollama, then run python3 website/setup_llm.py.'}
        MODEL_DIRECTORY.mkdir(parents=True,exist_ok=True)
        self.log=(MODEL_DIRECTORY.parent/'runtime.log').open('a',encoding='utf-8')
        self.process=subprocess.Popen([executable,'serve'],env=environment(config),stdout=self.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            ready=status(config)
            if ready['running']:return ready
            if self.process.poll() is not None:break
            time.sleep(.2)
        self.close()
        return status(config)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        if self.log is not None:self.log.close()
        self.process=self.log=None
