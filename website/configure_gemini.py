"""Configure both Task 5 entry points using a hidden prompt, never a key argument."""
import argparse
from getpass import getpass
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from task5.config import DEFAULT_GEMINI_MODEL
from website.gemini import probe


def update_environment(path,values):
    lines=path.read_text().splitlines() if path.exists() else []
    replaced=set();output=[]
    for line in lines:
        key=line.partition('=')[0].strip()
        if key in values:output.append(key+'='+values[key]);replaced.add(key)
        else:output.append(line)
    output.extend(key+'='+value for key,value in values.items() if key not in replaced)
    temporary=path.with_name(path.name+'.pending')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    try:
        with os.fdopen(descriptor,'w') as stream:stream.write('\n'.join(output)+'\n')
        os.chmod(temporary,0o600);temporary.replace(path);os.chmod(path,0o600)
    finally:temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',default=DEFAULT_GEMINI_MODEL)
    args=parser.parse_args();key=getpass('Gemini API key (hidden): ').strip()
    if not key or len(key)>500 or '\n' in key or '\r' in key:raise SystemExit('Invalid key format')
    result=probe(args.model,key)
    if not result['ready']:raise SystemExit(result['message'])
    for folder in ('integrated_pipeline','task5_standalone'):
        update_environment(ROOT/folder/'.env',{'GEMINI_API_KEY':key,'GEMINI_MODEL':result['model'],'TASK5_PROVIDER':'auto','TASK5_VERIFICATION':'standard'})
    print('Gemini model access validated: '+result['model'])
    print('Saved protected, ignored backend configuration. No key was logged.')
    return 0


if __name__=='__main__':raise SystemExit(main())
