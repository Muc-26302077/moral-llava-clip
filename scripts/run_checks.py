"""Run each test module in its own process, including on memory-limited CPUs."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    # CLI tests launch child interpreters, so run those before importing torch.
    for path in sorted((ROOT/'tests').glob('test_*.py')):
        print(path.name,flush=True)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',path.name,'-v'],
                       cwd=ROOT,env=environment,check=True)
    print('All test modules passed.',flush=True)


if __name__=='__main__':
    main()
