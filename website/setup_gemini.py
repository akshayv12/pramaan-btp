"""Check Gemini configuration without printing or accepting a key on the command line."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from task5.config import load_env_file
from website.gemini import probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',help='Available Gemini model ID; otherwise GEMINI_MODEL or the current documented Flash default')
    args=parser.parse_args()
    load_env_file(ROOT/'integrated_pipeline/.env');load_env_file(ROOT/'task5_standalone/.env')
    result=probe(model=args.model)
    print(json.dumps(result,indent=2))
    return 0 if result['ready'] else 2


if __name__=='__main__':raise SystemExit(main())
