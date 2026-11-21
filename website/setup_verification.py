"""Explicit, pinned download of the multilingual NLI verifier. Never runs on a search."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPO='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
REVISION='8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c'
DIRECTORY=ROOT/'website/.models/nli'
FILES={
    'onnx/model_quantized.onnx':('model.onnx',338679133,'27c39e884c14b03cf46cfc5485971b6db70ff330220d93dfe729c63fde43af0e'),
    'tokenizer.json':('tokenizer.json',16331396,'3aca3ce69a0a35aeb144a52c4f1d41c4246b8785f8f398315cc8fb6b24057810'),
    'config.json':('config.json',1066,None),
}


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    manifest={'model':REPO,'revision':REVISION,'files':{}}
    for remote,(name,size,expected) in FILES.items():
        target=DIRECTORY/name
        checksum=hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        if not (target.is_file() and target.stat().st_size==size and (expected is None or checksum==expected)):
            temporary=target.with_suffix('.part')
            digest=hashlib.sha256();written=0
            print('Downloading '+name+' ('+str(round(size/1_000_000,1))+' MB)',flush=True)
            try:
                with urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}',timeout=60) as response,temporary.open('wb') as stream:
                    while chunk:=response.read(1_048_576):
                        written+=len(chunk)
                        if written>size:raise RuntimeError('Model file exceeded its pinned size')
                        digest.update(chunk);stream.write(chunk)
                checksum=digest.hexdigest()
                if written!=size or expected and checksum!=expected:raise RuntimeError('Pinned model checksum/size mismatch')
                if name=='config.json':
                    raw=temporary.read_bytes()
                    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!='25b875998a15441ca06ee6d96b4cae362514549d':
                        raise RuntimeError('Pinned config checksum mismatch')
                temporary.replace(target)
            finally:temporary.unlink(missing_ok=True)
        manifest['files'][name]={'bytes':size,'sha256':checksum}
    (DIRECTORY/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Multilingual NLI files ready at '+str(DIRECTORY),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
