"""Export and validate image-in/text-out SFT deliveries (Feishu v1).

Format checks are independent of receiver acceptance. Full validation requires
the receiver's processor/template and measures expanded tokens and loss masks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMATS = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}
ROLES = {'system', 'developer', 'user', 'assistant', 'tool'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def media_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name or '~' in name or '..' in name:
        raise ValueError('Invalid POSIX image path')
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(x in ('.', '..') or x.startswith('.') for x in p.parts):
        raise ValueError('Image path must be canonical, relative and non-hidden')
    if any(not re.fullmatch(r'[A-Za-z0-9._-]+', x) for x in p.parts):
        raise ValueError('Image path contains unsupported characters')
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError('Image escapes source_root or is a symlink')
    return result


def image_info(root, name):
    from PIL import Image
    p = media_path(root, name)
    if p.suffix not in FORMATS or not p.is_file():
        raise ValueError(f'Unsupported or missing image: {name}')
    if p.stat().st_size > 20_000_000:
        raise ValueError(f'Image exceeds 20 MB: {name}')
    with Image.open(p) as im:
        if im.format != FORMATS[p.suffix]:
            raise ValueError(f'Encoding and extension differ: {name}')
        width, height = im.size
        im.load()
    return {'path': name, 'sha256': sha(p), 'bytes': p.stat().st_size,
            'width': width, 'height': height}


def record(obj):
    if not isinstance(obj, dict):
        raise ValueError('Each line must be an object')
    obj = obj.get('trace', obj)
    if not isinstance(obj, dict) or not isinstance(obj.get('messages'), list) or not obj['messages']:
        raise ValueError('messages must be a nonempty list in the effective record')
    if any('mask' in k.lower() for k in obj):
        raise ValueError('Handwritten loss masks are not supported')
    if 'tools' in obj and not isinstance(obj['tools'], list):
        raise ValueError('tools must be a list')
    for tool in obj.get('tools', []):
        if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
            raise ValueError('Invalid tool definition')
        fn = tool['function']
        if not isinstance(fn.get('name'), str) or not fn['name'] or not isinstance(fn.get('parameters'), dict):
            raise ValueError('Tool needs a name and parameter schema')
    users, supervised, images = [], [], []
    for i, m in enumerate(obj['messages']):
        if not isinstance(m, dict) or m.get('role') not in ROLES:
            raise ValueError('Unknown message role')
        role, content = m['role'], m.get('content')
        if role == 'system' and i != 0:
            raise ValueError('system must be first')
        if any('mask' in k.lower() for k in m):
            raise ValueError('Handwritten message loss masks are not supported')
        for flag in ('use_loss', 'use_thinking'):
            if flag in m and (role != 'assistant' or type(m[flag]) is not bool):
                raise ValueError(f'{flag} must be an assistant JSON boolean')
        if 'reasoning_content' in m and (role != 'assistant' or not isinstance(m['reasoning_content'], str)):
            raise ValueError('reasoning_content must be an assistant string')
        message_has_image = False
        if isinstance(content, list):
            if not content:
                raise ValueError('Empty content blocks')
            text_parts = []
            for b in content:
                if not isinstance(b, dict) or b.get('type') not in ('image', 'text'):
                    raise ValueError('Only text/image blocks supported; video is rejected')
                if b['type'] == 'image':
                    if role != 'user' or not isinstance(b.get('image'), str):
                        raise ValueError('Images allowed only in user messages')
                    images.append(b['image'])
                    message_has_image = True
                else:
                    if not isinstance(b.get('text'), str) or not b['text'].strip():
                        raise ValueError('Text block must be nonempty')
                    text_parts.append(b['text'])
            text = ''.join(text_parts)
        elif isinstance(content, str):
            text = content
        else:
            raise ValueError('content must be a string or block list')
        if any(t in text for t in ('<image>', '<|image_pad|>', '<|vision_start|>')):
            raise ValueError('Replace handwritten vision placeholders with image blocks')
        calls = m.get('tool_calls', [])
        if not isinstance(calls, list) or (calls and role != 'assistant'):
            raise ValueError('tool_calls must be an assistant list')
        for call in calls:
            if not isinstance(call, dict) or call.get('type') != 'function' or not isinstance(call.get('function'), dict):
                raise ValueError('Invalid tool call')
            fn = call['function']
            if not isinstance(fn.get('name'), str) or not fn['name']:
                raise ValueError('Tool call needs a name')
            arguments = fn.get('arguments')
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be a JSON object')
        if role == 'user' and (message_has_image or not (text.strip().startswith('<tool_response>') and text.strip().endswith('</tool_response>'))):
            users.append(i)
        if role == 'assistant' and m.get('use_loss', True) and (
                text.strip() or calls or (m.get('use_thinking', True) and m.get('reasoning_content', '').strip())):
            supervised.append(i)
    if not users or not any(i > users[-1] for i in supervised):
        raise ValueError('No supervised assistant after the last real user')
    return obj, images


def validate(root, processor=None, max_length=262144):
    root = Path(root).resolve()
    if not root.is_dir() or max_length <= 0:
        raise ValueError('source_root must exist; max length must be positive')
    paths = sorted(p for p in root.rglob('*.jsonl') if not any(x.startswith('.') for x in p.relative_to(root).parts))
    if not paths:
        raise ValueError('No visible JSONL files')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('source_root must contain real files/directories, not symlinks')
        if path.is_file() and not any(x.startswith('.') for x in path.relative_to(root).parts):
            if path.name != 'manifest.json' and path.suffix not in {'.jsonl', *FORMATS}:
                raise ValueError(f'Unrelated or unsupported file in source_root: {path.relative_to(root)}')
    rows, referenced, seen = [], set(), set()
    for p in paths:
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError('JSONL must be inside source_root')
        for line, text in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
            if not text.strip():
                continue
            try:
                obj, images = record(json.loads(text))
                if 'id' in obj:
                    if not isinstance(obj['id'], str) or obj['id'] in seen:
                        raise ValueError('Invalid or duplicate record id')
                    seen.add(obj['id'])
                for name in images:
                    media_path(root, name)
                referenced.update(images)
                rows.append((obj, images))
            except (ValueError, TypeError) as error:
                raise ValueError(f'{p.relative_to(root)}:{line}: {error}') from error
    if not rows:
        raise ValueError('No records')
    all_media = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in FORMATS}
    if referenced or all_media:
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest.get('version') != '1.0' or not isinstance(manifest.get('dataset'), dict) or not isinstance(manifest.get('files'), list):
            raise ValueError('Invalid manifest version/dataset/files')
        listed = {}
        for entry in manifest['files']:
            if not isinstance(entry, dict) or not isinstance(entry.get('path'), str) or entry['path'] in listed:
                raise ValueError('Invalid or duplicate manifest entry')
            actual = image_info(root, entry['path'])
            if any(entry.get(k) != v or type(entry.get(k)) is not type(v) for k, v in actual.items()):
                raise ValueError(f'Manifest metadata mismatch: {entry["path"]}')
            listed[entry['path']] = entry
        if set(listed) != all_media or not referenced <= set(listed):
            raise ValueError('Manifest must cover all images and all referenced paths')
    counts = Counter('image' if images else 'text' for _, images in rows)
    categories = Counter(obj.get('task', 'unspecified') for obj, _ in rows)
    result = {'scope': 'format-and-media', 'records': len(rows), 'modalities': dict(counts),
              'unique_images': len(all_media), 'task_counts': dict(categories),
              'receiver_acceptance': 'pending_processor_validation', 'max_seq_length': max_length,
              'total_tokens': None, 'trainable_tokens': None}
    if processor:
        result.update(processor_stats(root, rows, processor, max_length))
    return result


def processor_stats(root, rows, processor_path, max_length):
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=False)
    template = processor.chat_template
    if not isinstance(template, str) or not re.search(r'{%-?\s*generation\s*-?%}', template):
        raise ValueError('Receiver template must include generation supervision blocks')
    lengths, loss_lengths = [], []
    for obj, images in rows:
        # Build a processor-only copy with local absolute paths; delivered JSONL
        # retains source_root-relative paths. No answers are used at inference.
        messages = json.loads(json.dumps(obj['messages']))
        for m in messages:
            if isinstance(m['content'], list):
                for b in m['content']:
                    if b['type'] == 'image':
                        b['image'] = str(media_path(root, b['image']))
        batch = processor.apply_chat_template(messages, tools=obj.get('tools'), tokenize=True,
                    return_dict=True, return_assistant_tokens_mask=True, add_generation_prompt=False)
        ids, mask = batch['input_ids'], batch.get('assistant_masks')
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if hasattr(mask, 'tolist'):
            mask = mask.tolist()
        if ids and isinstance(ids[0], list):
            ids, mask = ids[0], mask[0]
        if not isinstance(mask, list) or len(mask) != len(ids):
            raise ValueError('Processor failed to return an aligned assistant mask')
        if images and 'pixel_values' not in batch:
            raise ValueError('Processor did not load image tensors')
        if len(ids) > max_length or not any(mask[1:max_length+1]):
            raise ValueError('Expanded sequence exceeds budget or shifted supervision is empty')
        lengths.append(len(ids))
        loss_lengths.append(sum(mask[1:max_length+1]))
    ordered = sorted(lengths)
    return {'scope': 'format-media-and-receiver-processor', 'receiver_acceptance': 'passed',
            'processor': str(processor_path), 'template_sha256': hashlib.sha256(template.encode()).hexdigest(),
            'total_tokens': sum(lengths), 'trainable_tokens': sum(loss_lengths),
            'length_distribution': {'min': min(lengths), 'p50': ordered[len(ordered)//2],
                'p95': ordered[min(len(ordered)-1, int(len(ordered)*.95))], 'max': max(lengths)}}


def export(data, image_root, output, run_id):
    from PIL import Image
    output, image_root = Path(output), Path(image_root).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    source = json.loads(Path(data).read_text())
    if not isinstance(source, list) or not source:
        raise ValueError('Source must be a nonempty legacy conversation array')
    # Do not emit a partial public delivery when any referenced image is absent.
    for row in source:
        if row.get('image'):
            p = media_path(image_root, row['image'])
            if not p.is_file():
                raise FileNotFoundError(f'Missing source image: {row["image"]}; apply declared exclusions before export')
    output.mkdir(parents=True)
    files, exported = {}, []
    for row in source:
        image = None
        if row.get('image'):
            src = media_path(image_root, row['image'])
            with Image.open(src) as im:
                ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(im.format)
                if not ext:
                    raise ValueError(f'Unsupported source encoding: {src}')
            image = 'assets/' + sha(src) + ext
            dest = output/image
            if image not in files:
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(src, dest)
                files[image] = image_info(output, image)
        messages, used = [], False
        for turn in row['conversations']:
            role = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}.get(turn['from'])
            if role is None:
                raise ValueError('Unknown source role')
            text = turn['value']
            if '<image>' in text:
                if role != 'user' or not image or used or text.count('<image>') != 1:
                    raise ValueError('Image placeholder is ambiguous')
                before, after = text.split('<image>')
                content = ([{'type': 'text', 'text': before}] if before.strip() else [])
                content += [{'type': 'image', 'image': image}]
                if after.strip():
                    content.append({'type': 'text', 'text': after.lstrip('\n')})
                used = True
            else:
                content = text
            messages.append({'role': role, 'content': content})
        if image and not used:
            raise ValueError('Source image was not referenced in conversation')
        metadata = {k: v for k, v in row.items() if k not in ('conversations', 'image')}
        obj = {'trace': {**metadata, 'messages': messages}}
        record(obj)
        exported.append(obj)
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for obj in exported:
            stream.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n')
    if files:
        save(output/'manifest.json', {'version': '1.0', 'dataset': {'name': 'smid-moral-sft',
            'run_id': run_id, 'created_at': datetime.now(timezone.utc).date().isoformat()}, 'files': [files[k] for k in sorted(files)]})
    return validate(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--data', required=True)
    e.add_argument('--image-root', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--run-id', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--source-root', required=True)
    v.add_argument('--processor', help='Receiver Qwen3.5-VL processor with generation-aware template')
    v.add_argument('--max-seq-length', type=int, default=262144)
    v.add_argument('--report', help='Write OUTSIDE source_root')
    args = p.parse_args()
    if args.command == 'export':
        result = export(args.data, args.image_root, args.output, args.run_id)
    else:
        if args.report and Path(args.report).resolve().is_relative_to(Path(args.source_root).resolve()):
            p.error('Validation report must be outside source_root')
        result = validate(args.source_root, args.processor, args.max_seq_length)
        if args.report:
            save(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
