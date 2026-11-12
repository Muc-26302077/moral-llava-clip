"""Real SMID/MoralCLIP CPU feasibility check, NOT a 7B generative TICR run.

Cache frozen block-11 outputs. Train the actual new scene adapter through the
frozen final CLIP block using a small zero-shot polarity diagnostic head.
Train/validation scenes stay separate; test.json is never opened.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/polarity-v1')
    p.add_argument('--image-root', type=Path, default=ROOT/'data/raw/SMID_moralclip')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--per-class', type=int, default=2)
    p.add_argument('--validation-per-class', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.per_class, args.validation_per_class, args.batch_size) <= 0:
        p.error('Counts must be positive')
    if not 0 < args.learning_rate < 1:
        p.error('Learning rate must be in (0,1)')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.nn import functional as F
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image, ImageOps
    from moral_llava import polarity
    from moral_llava.scene_guided_vision import SceneConfig, SceneResidual, SceneContext, caption_index
    from moral_llava.moralclip_integration import WEIGHT_SHA256
    from run_text_guided_vision import balanced_subset, save, sha
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    def status(phase, **kw):
        payload = dict(status='running', phase=phase, seconds=time.monotonic()-started, **kw)
        save(args.output/'status.json', payload)
        print(json.dumps(payload), flush=True)
    status('load_real_weights')
    if sha(args.model/'model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Not the frozen MoralCLIP weights')
    model, info = CLIPModel.from_pretrained(args.model, local_files_only=True,
        low_cpu_mem_usage=True, output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Actual pretrained CLIP weights are incomplete')
    model.requires_grad_(False).eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    raw, rows, captions, scenes = {}, {}, {}, {}
    audit = json.loads((args.data/'audit.json').read_text())
    for split, count in [('train', args.per_class), ('validation', args.validation_per_class)]:
        if sha(args.data/f'{split}.json') != audit['split_record_sha256'][split]:
            raise ValueError('Frozen data changed')
        raw[split] = json.loads((args.data/f'{split}.json').read_text())
        normalized = polarity.load_records(args.data/f'{split}.json', args.image_root)
        rows[split] = balanced_subset(normalized, count)
        all_captions = caption_index(raw[split], split=split)
        scenes[split] = {r['scene_id'] for r in rows[split]}
        captions.update({k: all_captions[k] for k in scenes[split]})
    if scenes['train'] & scenes['validation']:
        raise ValueError('Cross-split scene leakage')
    config = SceneConfig()
    adapter = SceneResidual(config)
    # Two English class prompts per known task; no learned classifier or LLaVA.
    classes = {
        'care': ['A scene of harm, cruelty and suffering.', 'A scene of care, compassion and helping.'],
        'fairness': ['A scene of cheating, injustice and unfairness.', 'A scene of fairness, justice and equality.'],
        'loyalty': ['A scene of betrayal and disloyalty.', 'A scene of loyalty, solidarity and commitment.'],
        'authority': ['A scene of subversion and disrespect for legitimate authority.', 'A scene of respect for legitimate authority and social order.'],
        'sanctity': ['A scene of degradation, impurity and disgust.', 'A scene of purity, sanctity and dignity.']}
    def text_features(texts):
        inputs = processor.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt')
        return F.normalize(model.get_text_features(**inputs).float(), dim=-1)
    with torch.no_grad():
        prototypes = {task: text_features(texts) for task, texts in classes.items()}
        targets = {}
        for scene, caption in captions.items():
            targets[scene] = text_features([caption])[0]
    images = {}
    for r in rows['train']+rows['validation']:
        images[r['scene_id']] = r['image_path']
    hidden = {}
    for i, (scene, path) in enumerate(sorted(images.items())):
        with Image.open(path) as im:
            pixel = processor.image_processor(images=ImageOps.exif_transpose(im).convert('RGB'), return_tensors='pt')['pixel_values']
        with torch.no_grad():
            output = model.vision_model(pixel_values=pixel, output_hidden_states=True)
            hidden[scene] = output.hidden_states[-2].detach().clone()
        status('cache_real_visual_features', completed=i+1, total=len(images))
    scale = model.logit_scale.exp().detach().clone()
    last_block = model.vision_model.encoder.layers[-1]
    post_norm = model.vision_model.post_layernorm
    projection = model.visual_projection
    frozen_modules = torch.nn.ModuleList([last_block, post_norm, projection])
    # The prefix and text tower are frozen and their outputs are now cached.
    # Release them before differentiating the small retained visual suffix.
    del output, model, processor, pixel
    gc.collect()
    # Linux libc may retain freed CPU tensor arenas after the prefix is dropped.
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
    if trim:
        trim(0)
    status('release_frozen_prefix_before_training')
    def visual_projection(h):
        output = last_block(h, attention_mask=None, causal_attention_mask=None)[0]
        return F.normalize(projection(post_norm(output[:, 0])).float(), dim=-1)
    def batch(rows, mode, enabled):
        h = torch.cat([hidden[r['scene_id']] for r in rows])
        target = torch.stack([targets[r['scene_id']] for r in rows])
        alignment = torch.tensor(0.)
        penalty = torch.tensor(0.)
        if enabled:
            spatial, alignment = adapter(h[:, 1:], SceneContext(mode, target))
            penalty = (spatial-h[:, 1:]).square().mean()/(h[:, 1:].square().mean()+1e-6)
            h = torch.cat((h[:, :1], spatial), dim=1)
        features = visual_projection(h)
        label_features = torch.stack([prototypes[r['task']] for r in rows])
        logits = torch.einsum('bd,bcd->bc', features, label_features)*scale
        teacher = torch.einsum('bd,bcd->bc', target, label_features)*scale
        labels = torch.tensor([int(polarity.parse_answer(r['target_json'], r['task'])) for r in rows])
        ce = F.cross_entropy(logits, labels)
        teacher_nll = F.cross_entropy(teacher, labels, reduction='none')
        image_nll = F.cross_entropy(logits, labels, reduction='none')
        gate = (teacher_nll <= image_nll.detach()).float()
        kl = (F.kl_div(F.log_softmax(logits/2., -1), F.softmax(teacher/2., -1), reduction='none').sum(-1)*4*gate).mean()
        loss = ce+.2*kl+.1*alignment+.05*penalty
        return loss, logits, teacher, labels, features
    def evaluate(split, mode, enabled):
        records = rows[split]
        summaries = []
        with torch.no_grad():
            for offset in range(0, len(records), args.batch_size):
                current = records[offset:offset+args.batch_size]
                loss, logits, teacher, labels, features = batch(current, mode, enabled)
                spatial = torch.cat([hidden[r['scene_id']][:, 1:] for r in current])
                predicted = F.normalize(adapter.predictor(spatial.mean(1)), dim=-1)
                true_scene = torch.stack([targets[r['scene_id']] for r in current])
                summaries.append((len(current), {'diagnostic_loss': float(loss),
                    'image_polarity_accuracy': float((logits.argmax(-1) == labels).float().mean()),
                    'text_polarity_accuracy': float((teacher.argmax(-1) == labels).float().mean()),
                    'image_text_agreement': float((logits.argmax(-1) == teacher.argmax(-1)).float().mean()),
                    'scene_predictor_mean_cosine': float((predicted*true_scene).sum(-1).mean())}))
        return {k: sum(n*m[k] for n,m in summaries)/len(records) for k in summaries[0][1]}
    baseline = {s: evaluate(s, 'predicted', False) for s in rows}
    with torch.no_grad():
        selected = rows['validation'][:2]
        before = batch(selected, 'predicted', False)[1]
        torch.testing.assert_close(batch(selected, 'predicted', True)[1], before, atol=0, rtol=0)
        torch.testing.assert_close(batch(selected, 'paired', True)[1], before, atol=0, rtol=0)
    # Hash all frozen tensors to verify that only the new adapter changed.
    def frozen_hash():
        digest = hashlib.sha256()
        for name, parameter in frozen_modules.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().numpy().tobytes())
        return digest.hexdigest()
    initial_hash = frozen_hash()
    initial = {s: evaluate(s, 'predicted', True) for s in rows}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0)
    trace = []
    for epoch in range(1, args.epochs+1):
        shuffled = list(rows['train'])
        random.Random(args.seed+epoch).shuffle(shuffled)
        for i in range(0, len(shuffled), args.batch_size):
            current = shuffled[i:i+args.batch_size]
            optimizer.zero_grad()
            independent = batch(current, 'predicted', True)[0]
            independent.backward()
            paired = batch(current, 'paired', True)[0]
            (.5*paired).backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            optimizer.step()
        metrics = {s: evaluate(s, 'predicted', True) for s in rows}
        trace.append({'epoch': epoch, **metrics})
        status('train_actual_scene_adapter', epoch=epoch, metrics=metrics)
    final = {s: {mode: evaluate(s, mode, True) for mode in ('predicted', 'paired')} for s in rows}
    if initial_hash != frozen_hash() or any(p.grad is not None for p in frozen_modules.parameters()):
        raise ValueError('A frozen pretrained parameter changed or received a gradient')
    # Actual held-out image, different provided texts -> visual representation changes.
    sample = rows['validation'][0]
    h = hidden[sample['scene_id']]
    with torch.no_grad():
        first = targets[sample['scene_id']][None]
        other = targets[rows['validation'][-1]['scene_id']][None]
        a, _ = adapter(h[:, 1:], SceneContext('paired', first))
        b, _ = adapter(h[:, 1:], SceneContext('paired', other))
        paired_delta = float((a-b).abs().max())
        a, _ = adapter(h[:, 1:], SceneContext('predicted', first))
        b, _ = adapter(h[:, 1:], SceneContext('predicted', other))
        independent_delta = float((a-b).abs().max())
    if paired_delta <= 0 or independent_delta != 0:
        raise ValueError('Scene dependency/isolation check failed')
    report = {'status': 'PASS', 'diagnostic_only': True, 'formal_7b_ticr_available': False,
        'scope': 'real frozen MoralCLIP with new scene adapter; frozen final CLIP block + zero-shot class prompts; no LLaVA decoder',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'guidance_config': asdict(config), 'trainable_parameters': sum(p.numel() for p in adapter.parameters()),
        'sample_counts': {s: {'scene_task_questions': len(rows[s]), 'unique_scenes': len(scenes[s])} for s in rows},
        'selected_ids': {s: [r['id'] for r in rows[s]] for s in rows},
        'source_sha256': {'model': WEIGHT_SHA256, **{s: sha(args.data/f'{s}.json') for s in rows},
                          'script': sha(Path(__file__)), 'scene_module': sha(ROOT/'moral_llava/scene_guided_vision.py')},
        'baseline': baseline, 'initial_guided': initial, 'final_guided': final, 'training_trace': trace,
        'paired_text_changes_visual_features_max_abs': paired_delta,
        'independent_caption_swap_changes_features_max_abs': independent_delta,
        'retained_frozen_weights_unchanged': True, 'original_clip_parameters_frozen': True,
        'zero_init_matches_baseline': True,
        'test_split_used': False, 'validation_used_for_selection': False,
        'seconds': time.monotonic()-started,
        'interpretation': 'CPU structural/optimization feasibility only. Diagnostic agreement is not generative TICR and cannot establish 7B improvement.'}
    save(args.output/'report.json', report)
    save(args.output/'status.json', {'status': 'complete', 'formal_ticr_available': False, 'seconds': report['seconds']})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
