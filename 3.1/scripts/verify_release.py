"""Verify the released files and independently rescore frozen predictions on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moral_llava.crossmodal import REVISION, score
from moral_llava.schema import labels_to_target, parse_prediction


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = read(ROOT / 'release_manifest.json')
    require(manifest['format_version'] == 1, 'Unsupported release manifest')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        require(path.resolve().is_relative_to(ROOT), f'Unsafe path: {entry["path"]}')
        require(path.is_file(), f'Missing file: {entry["path"]}')
        require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                f'File mismatch: {entry["path"]}; for model files, run git lfs pull')

    image_root = ROOT / 'data/raw/SMID_moralclip'
    original = read(image_root / 'smid_manifest.json') + read(image_root / 'test/smid_manifest.json')
    require(len(original) == len({r['id'] for r in original}) == 2400, 'Duplicate/missing dataset IDs')
    require(dict(Counter(r['split'] for r in original)) == {'train':2140, 'validation':134, 'test':126},
            'Unexpected split counts')
    with (ROOT / 'data/image_attribution.csv').open(encoding='utf-8', newline='') as stream:
        attribution = list(csv.DictReader(stream))
    require(len(attribution) == len({r['id'] for r in attribution}) == 2400,
            'Attribution inventory must cover every unique image ID')
    attributed = {r['id']:r for r in attribution}
    exclusions = read(ROOT / 'data/excluded_images.json')['images']
    excluded = {r['image_path']:r for r in exclusions}
    require(set(excluded) == {'train/b15_p445_5.jpg','train/b15_p465_20.jpg'}, 'Unexpected image exclusions')
    hashes_by_split = {split:set() for split in ['train','validation','test']}
    for split in hashes_by_split:
        processed = read(ROOT / f'data/processed/{split}.json')
        source = {r['id']:r for r in original if r['split'] == split}
        require(len(processed) == len(source) and {r['id'] for r in processed} == set(source),
                f'Processed coverage mismatch: {split}')
        for row in processed:
            source_row = source[row['id']]
            require(row['image'] == source_row['image_path'], 'Processed image path changed')
            require(parse_prediction(row['conversations'][1]['value']) == labels_to_target(source_row['moral_labels']),
                    'Processed reference labels changed')
            if row['image'] in excluded:
                require(not (image_root / row['image']).exists(), 'Restricted image was included in the public package')
                image_hash = excluded[row['image']]['sha256']
            else:
                image_hash = sha256(image_root / row['image'])
            hashes_by_split[split].add(image_hash)
            record = attributed[row['id']]
            require(record['sha256'] == image_hash and record['split'] == split and record['image_path'] == row['image'],
                    'Attribution inventory refers to a different image')
    for left, right in [('train','validation'),('train','test'),('validation','test')]:
        require(not (hashes_by_split[left] & hashes_by_split[right]), 'Image bytes overlap across splits')

    reference = ROOT / 'results/reference'
    training = reference / 'runs/crossmodal-joint-v1'
    selection = read(training / 'selection.json')
    config = read(training / 'run_config.json')
    require(selection['selected_epoch'] == 2, 'Unexpected selected epoch')
    selected = ROOT / 'models/moralllava'
    require(selection['selected_adapter'] == 'runs/crossmodal-joint-v1/epoch-002', 'Unexpected historical selection')
    require(read(training / 'status.json')['status'] == 'complete', 'Training incomplete')
    require(sha256(ROOT/'moral_llava/train_crossmodal.py') == config['source_sha256'], 'Training code changed')
    require(sha256(image_root/'smid_manifest.json') == config['manifest_sha256'], 'Training source manifest changed')
    initial_hash = sha256(ROOT/'models/image-lora/adapter_model.safetensors')
    require(initial_hash == config['initial_adapter_sha256'], 'Initial adapter mismatch')
    selected_hash = sha256(selected/'adapter_model.safetensors')
    require(selected_hash != initial_hash, 'Selected adapter equals initialization')
    require(read(selected/'training_metadata.json')['resolved_base_commit'] == REVISION, 'Base revision mismatch')
    initial = read(training/'validation_initial.json')['metrics']
    best_epoch, best_value = 0, initial['strict_agreement']
    for epoch in range(1,4):
        m = read(training/f'validation_epoch_{epoch:03d}.json')['metrics']
        tolerance = config['quality_tolerance']
        eligible = ((m['image_exact_match']+m['text_exact_match'])/2 >=
                    (initial['image_exact_match']+initial['text_exact_match'])/2-tolerance and
                    m['joint_correct_rate'] >= initial['joint_correct_rate']-tolerance)
        if eligible and m['strict_agreement'] > best_value:
            best_epoch, best_value = epoch, m['strict_agreement']
    require(best_epoch == selection['selected_epoch'], 'Validation selection cannot be reproduced')

    protocol_path = reference/'runs/crossmodal-v1/protocol.json'
    protocol, protocol_hash = read(protocol_path), sha256(protocol_path)
    require({s['id'] for s in protocol['scenes']} == {r['id'] for r in original if r['split']=='test'},
            'Protocol does not cover the test set')
    test_source = {r['id']:r for r in original if r['split']=='test'}
    for scene in protocol['scenes']:
        row = test_source[scene['id']]
        require(scene['image_sha256'] == sha256(image_root/row['image_path']), 'Protocol image mismatch')
        require(scene['caption'] == row['captions'][0] and scene['reference'] == labels_to_target(row['moral_labels']),
                'Protocol caption/reference mismatch')
    old = reference/'runs/crossmodal-v1/full'
    new = reference/'runs/crossmodal-joint-v1-test'
    for directory in [old,new]:
        status = read(directory/'status.json')
        require(status['status']=='complete' and status['protocol_sha256']==protocol_hash, 'Inference provenance mismatch')
    require(read(new/'status.json')['adapter_sha256']==selected_hash, 'Inference used another adapter')

    def predictions(path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(len(rows)==252, f'Unexpected prediction count: {path}')
        for row in rows:
            require(row['protocol_sha256']==protocol_hash, 'Prediction protocol mismatch')
            scene = next(s for s in protocol['scenes'] if s['id']==row['id'])
            prompt = (protocol['image_prompt'] if row['modality']=='image' else
                      protocol['text_prompt_template'].replace('{caption}',scene['caption']))
            require(hashlib.sha256(prompt.encode()).hexdigest()==row['prompt_sha256'], 'Prediction prompt mismatch')
        return rows

    results = {}
    for decoding in ['constrained','free']:
        base = predictions(old/f'base_{decoding}.jsonl')
        for version,directory in [('old',old),('new',new)]:
            lora = predictions(directory/f'lora_{decoding}.jsonl')
            actual = score(protocol, {'base':base,'lora':lora})
            saved_path = (directory if version=='old' else reference/'evaluation')/f'metrics_{decoding}.json'
            saved = read(saved_path)
            require(actual == {k:saved[k] for k in actual}, f'Recomputed metrics differ: {version}/{decoding}')
            results[f'{version}_{decoding}'] = {
                g:{k:actual['groups'][g]['models']['lora'][k]
                   for k in ['consistent_pairs','strict_agreement','both_correct_pairs','joint_correct_rate']}
                for g in ['A','B']}
    return {'status':'PASS','verified_files':len(manifest['files']), 'dataset_records':2400,
            'packaged_images':2398,'complete_test_images':126,'excluded_images':list(excluded),
            'split_counts':{'train':2140,'validation':134,'test':126},
            'selected_adapter_sha256':selected_hash,'protocol_sha256':protocol_hash,'scores':results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional new JSON report; existing files are never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f'Report already exists: {args.output}')
    result = verify()
    serialized = json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(serialized)
    print(serialized,end='')


if __name__=='__main__':
    main()
