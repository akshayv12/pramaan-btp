"""Install checksum-pinned English/Hindi data for an existing Tesseract runtime."""
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

MODELS={
    'eng':'7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2',
    'hin':'4c73ffc59d497c186b19d1e90f5d721d678ea6b2e277b719bee4e2af12271825',
}


def main():
    if not shutil.which('tesseract'):
        raise SystemExit('Install the Tesseract executable first (macOS: brew install tesseract).')
    folder=Path(__file__).parent/'.ocr/tessdata'
    folder.mkdir(parents=True,exist_ok=True)
    for language,checksum in MODELS.items():
        path=folder/(language+'.traineddata')
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==checksum:
            print(language+': verified')
            continue
        url=f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/{language}.traineddata'
        with urlopen(url,timeout=45) as response:
            data=response.read(12_000_001)
        if len(data)>12_000_000 or hashlib.sha256(data).hexdigest()!=checksum:
            raise SystemExit('Unexpected OCR data for '+language+'; existing model was preserved.')
        pending=path.with_suffix('.pending')
        pending.write_bytes(data);pending.replace(path)
        print(language+': installed and verified')
    print('Restart the local website to refresh OCR readiness.')


if __name__=='__main__':main()
