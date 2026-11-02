"""Rebuild the release inventory after reviewing an intentional packaging change."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


def main():
    listed = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                                     cwd=REPOSITORY).decode().split('\0')
    files=[]
    for name in sorted(set(listed)):
        path=REPOSITORY/name
        if not name.startswith('3.1/') or not path.is_file() or path==ROOT/'release_manifest.json':
            continue
        # A tracked file may become ignored after restructuring; do not absorb generated outputs.
        if '__pycache__' in path.parts or path.is_relative_to(ROOT/'results/runs'):
            continue
        with path.open('rb') as stream:
            checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        files.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':checksum})
    (ROOT/'release_manifest.json').write_text(json.dumps({'format_version':1,'files':files},
                                                      ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Inventory: {len(files)} files')


if __name__=='__main__':
    main()
