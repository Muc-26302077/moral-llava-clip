"""Verify published data/model/result fingerprints and all local document links."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'results/manifest.json'


def files():
    for directory in ('data','models','results'):
        for path in sorted((ROOT/directory).rglob('*')):
            relative=path.relative_to(ROOT)
            if (not path.is_file() or path==MANIFEST or '__pycache__' in path.parts
                    or relative.parts[:2] in (('data','polarity-v1'),('data','moralise'),('models','base'),('results','runs'))
                    or relative.as_posix() in ('data/moralise-native.json','data/moralise-audit.json','data/moralise-image-audit.json')):
                continue
            yield path


def fingerprints():
    result={}
    for path in files():
        with path.open('rb') as stream:
            if path.suffix in ('.safetensors','.bin') and stream.read(80).startswith(b'version https://git-lfs'):
                raise ValueError('Unresolved LFS pointer: '+str(path.relative_to(ROOT)))
            stream.seek(0)
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        result[path.relative_to(ROOT).as_posix()]={'bytes':path.stat().st_size,'sha256':digest}
    return result


def check_links():
    total=0
    for path in ROOT.rglob('*.md'):
        if any(x in path.relative_to(ROOT).parts for x in ('base','runs','raw','.cache','moralise')):
            continue
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('https://','http://','mailto:')):
                continue
            filename=target.split('#')[0]
            if filename and not (path.parent/filename).exists():
                raise ValueError(f'Broken link in {path.relative_to(ROOT)}: {target}')
            if '#' in target:
                anchor = target.split('#',1)[1]
                destination = path.parent/filename if filename else path
                if anchor and f'id="{anchor}"' not in destination.read_text():
                    raise ValueError(f'Broken jump in {path.relative_to(ROOT)}: {target}')
            total+=1
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true',help='Maintainer: create a new final artifact index')
    args=parser.parse_args()
    actual=fingerprints()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({'version':1,'files':actual},ensure_ascii=False,indent=2)+'\n')
    expected=json.loads(MANIFEST.read_text())['files']
    if actual!=expected:
        changed=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
        raise ValueError('Missing or modified final artifacts: '+', '.join(changed[:10]))
    print(json.dumps({'status':'PASS','verified_files':len(actual),'local_document_links':check_links()},indent=2))


if __name__=='__main__':
    main()
