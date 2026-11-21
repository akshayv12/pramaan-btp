"""Explicitly download the configured local model; searches never trigger downloads."""
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from website.local_llm import LocalRuntime,environment,settings,status


def main():
    if not shutil.which('ollama'):
        raise SystemExit('Install Ollama first (macOS: brew install ollama).')
    runtime=LocalRuntime()
    try:
        runtime.start()
        if not status()['running']:raise SystemExit('Could not start the local model service. Check website/.models/runtime.log.')
        config=settings()
        if not status(config)['ready']:
            subprocess.run(['ollama','pull',config['model']],env=environment(config),check=True)
        print('Local model ready: '+config['model'])
        print('Start the website with: python3 website/launch.py start')
    finally:runtime.close()


if __name__=='__main__':main()
