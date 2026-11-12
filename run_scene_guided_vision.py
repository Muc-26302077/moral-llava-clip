"""Scene-text guided visual encoder: inspect/preflight/smoke/experiment/infer.

Independent inference predicts a scene condition from the image, without reading
its caption. Paired inference uses the provided caption and has separate metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import run_text_guided_vision as common

ROOT = Path(__file__).resolve().parent


def scene_selection_key(metrics, baseline):
    """TICR is the validation objective, with accuracy/BA/joint/format guards."""
    m, b = [x['overall']['primary_macro'] for x in (metrics, baseline)]
    if any(m[key] + 1e-12 < b[key] for key in
           ('image_accuracy', 'image_balanced_accuracy', 'joint_accuracy')):
        return None
    invalid = lambda x: sum(t['image']['invalid_outputs'] for t in x['overall']['per_task'].values())
    if invalid(metrics) > invalid(baseline):
        return None
    return m['ticr'], m['joint_accuracy'], m['image_balanced_accuracy']


def inspect_data(args):
    from moral_llava.scene_guided_vision import caption_index, METHOD, INDEPENDENT, PAIRED
    raw, rows, audit = common.ensure_data(args)
    return {'method': METHOD, 'status': 'data_checked_not_trained',
        'paired_scene_counts': {s: len(caption_index(r, split=s)) for s, r in raw.items()},
        'question_counts': {s: len(r) for s, r in raw.items()},
        'training_image_questions': sum(r['modality'] == 'image' for r in rows['train']),
        'independent_protocol': INDEPENDENT, 'paired_protocol': PAIRED,
        'independent_image_caption_access': False,
        'training_condition': 'frozen MoralCLIP encoding of actual same-scene caption0',
        'deployment_condition': 'image-predicted scene semantics or supplied scene text',
        'test_policy': 'validation selects independent mode; test is evaluated only after selection',
        'formal_ticr_available': False}


def load_runtime(args, *, need_text):
    from moral_llava.scene_guided_vision import install_scene_guidance, SceneTextEncoder
    runtime = common.load_runtime(args, guidance_installer=install_scene_guidance)
    encoder = SceneTextEncoder(args.moralclip_model) if need_text else None
    if encoder and encoder.model.config.projection_dim != runtime[2].adapter.config.text_dim:
        raise ValueError('Actual MoralCLIP text projection dimension differs from scene adapter')
    return (*runtime, encoder)


def scene_features(encoder, records, captions):
    if encoder is None:
        raise ValueError('Paired mode requires the frozen scene text encoder')
    return encoder.encode([captions[r['scene_id']] for r in records])


def predictions(args, model, processor, block, dtype, records, *, enabled,
                split, smoke, guidance_hash=None, mode='predicted', encoder=None, captions=None):
    import torch
    from moral_llava import polarity
    from moral_llava.polarity_runtime import inputs_for_records
    from moral_llava.train import unpad_input_ids
    from moral_llava.scene_guided_vision import SceneContext, METHOD, INDEPENDENT, PAIRED
    from moral_llava.moralclip_integration import METHOD as BASE_METHOD, WEIGHT_SHA256
    if mode not in ('predicted', 'paired'):
        raise ValueError('Invalid inference mode')
    protocol = PAIRED if enabled and mode == 'paired' else INDEPENDENT
    if protocol == INDEPENDENT and captions is not None:
        raise ValueError('Independent inference must not receive external scene captions')
    processor.tokenizer.padding_side = 'left'
    model.eval()
    outputs = []
    for offset in range(0, len(records), args.eval_batch_size):
        current = records[offset:offset+args.eval_batch_size]
        images = [r for r in current if r['modality'] == 'image']
        features = scene_features(encoder, images, captions) if images and enabled and mode == 'paired' else None
        context = SceneContext(mode, features) if images and enabled else None
        inputs = inputs_for_records(processor, current)
        for i, row in enumerate(current):
            ids = unpad_input_ids(inputs['input_ids'][i].tolist(), inputs['attention_mask'][i].tolist())
            expected = model.config.image_seq_length if row['modality'] == 'image' else 0
            if ids.count(model.config.image_token_index) != expected or len(ids)+16 > args.max_length:
                raise ValueError('Image tokens/context budget mismatch: '+row['id'])
        inputs = common.move_batch(inputs, dtype)
        width = inputs['input_ids'].shape[1]
        with torch.inference_mode(), block.conditioned(context, enabled=enabled and bool(images)):
            tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id)
        answers = processor.batch_decode(tokens[:, width:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        for row, answer in zip(current, answers):
            prediction = polarity.format_prediction(row, answer)
            prediction.update(data_sha256=common.sha(args.data/f'{split}.json'), smoke_only=smoke,
                decoding='free', base_model_id='llava-hf/llava-1.5-7b-hf', base_revision=common.BASE_REVISION,
                adapter_sha256=common.baseline_sha(args), precision=str(dtype), max_new_tokens=16,
                vision_weight_sha256=WEIGHT_SHA256, integration_method=METHOD if enabled else BASE_METHOD,
                guidance_sha256=guidance_hash if enabled else None, guidance_enabled=enabled,
                condition_source=('image_predicted_scene_semantics' if mode == 'predicted' else
                                  'provided_same_scene_caption0') if enabled else None,
                evaluation_protocol=protocol, external_image_caption_access=protocol == PAIRED,
                scene_caption_sha256=(hashlib.sha256(captions[row['scene_id']].encode()).hexdigest()
                    if protocol == PAIRED and row['modality'] == 'image' else None))
            outputs.append(prediction)
        print(f'{split} {mode} guided={enabled}: {min(offset+len(current),len(records))}/{len(records)}', flush=True)
    return outputs


def score(raw, outputs, audit=None, *, diagnostic=False):
    from moral_llava.scene_guided_vision import INDEPENDENT, PAIRED
    protocols = {p['evaluation_protocol'] for p in outputs}
    if len(protocols) != 1 or next(iter(protocols)) not in (INDEPENDENT, PAIRED):
        raise ValueError('Mixed or missing scene evaluation protocol')
    protocol = next(iter(protocols))
    if any(p['external_image_caption_access'] != (protocol == PAIRED) for p in outputs):
        raise ValueError('Caption access disagrees with evaluation protocol')
    metrics = common.metrics_for(raw, outputs, audit, diagnostic=diagnostic)
    metrics.update(evaluation_protocol=protocol, comparable_to_independent_ticr=protocol == INDEPENDENT)
    if protocol == PAIRED:
        metrics.update(protocol=PAIRED, metric_name='Caption-assisted agreement', formal_test_result=False)
        metrics['notes'].append('Image encoder reads scene text; this is not the independent-input TICR protocol.')
    return metrics


def comparison(args, raw, baseline, guided, metrics):
    from moral_llava.guided_report import paired_intervals
    b, g = [metrics[name]['overall']['primary_macro'] for name in ('baseline', 'guided')]
    keys = ('ticr', 'image_accuracy', 'text_accuracy', 'image_balanced_accuracy', 'joint_accuracy')
    delta = {key: g[key]-b[key] for key in keys}
    original = json.loads((ROOT/'results/benchmarks/original-tower/ticr-metrics.json').read_text())
    original = original.get('metrics', original)['overall']['primary_macro']
    return {'scope': 'independent image-only inference; external scene text used during training only',
        'baseline': b, 'guided': g, 'delta': delta,
        'paired_scene_bootstrap_delta95': paired_intervals(raw, baseline, guided),
        'passes_point_estimate_goal': delta['ticr'] > 0 and delta['image_accuracy'] >= 0 and delta['joint_accuracy'] > 0,
        'archived_original_tower_reference': original,
        'exceeds_archived_original_tower_ticr': g['ticr'] > original['ticr'],
        'meets_requested_ticr_target': g['ticr'] > max(b['ticr'], original['ticr'])
            and delta['image_accuracy'] >= 0 and delta['joint_accuracy'] > 0,
        'reference_note': 'Original-tower reference is historical; not a new same-runtime paired comparison.'}


def train_batch(args, model, processor, block, dtype, encoder, captions, records, text_rows, collator):
    """Text teacher -> independent image student, plus actual-caption guided branch."""
    import torch
    from moral_llava.scene_guided_vision import SceneContext, answer_logits, supervised_and_distilled
    target = scene_features(encoder, records, captions)
    device = next(model.parameters()).device
    def move(batch):
        return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                for k, v in batch.items()}
    teacher_inputs = move(collator(text_rows))
    with torch.no_grad():
        teacher_output = model(**teacher_inputs, use_cache=False)
        teacher = [(logits.detach().clone(), labels.detach().clone()) for logits, labels in
                   answer_logits(teacher_output.logits, teacher_inputs['labels'])]
    del teacher_output, teacher_inputs
    inputs = move(collator(records))
    with block.conditioned(SceneContext('predicted', target)):
        output = model(**inputs, use_cache=False)
        ce, kl, gate = supervised_and_distilled(output.logits, inputs['labels'], teacher,
                                               temperature=args.temperature)
        alignment, penalty = block.last_alignment, block.last_penalty
        loss = ce + args.distill_weight*kl + args.alignment_weight*alignment + args.preservation_weight*penalty
        if not torch.isfinite(loss):
            raise ValueError('Non-finite independent scene training loss')
        # Backward inside the scope, including non-reentrant decoder checkpointing.
        (loss*args._batch_weight).backward()
        report = {'image_ce': float(ce.detach()), 'teacher_kl': float(kl.detach()),
            'teacher_gate_fraction': float(gate.detach()), 'scene_alignment': float(alignment.detach()),
            'preservation': float(penalty.detach()), 'independent_loss': float(loss.detach())}
    del output, loss, teacher, inputs, ce, kl, alignment, penalty
    # A separate forward/backward avoids retaining both 7B activation graphs.
    if args.paired_weight:
        inputs = move(collator(records))
        with block.conditioned(SceneContext('paired', target)):
            output = model(**inputs, use_cache=False)
            aligned = answer_logits(output.logits, inputs['labels'])
            paired_ce = torch.stack([torch.nn.functional.cross_entropy(logits, labels)
                                     for logits, labels in aligned]).mean()
            loss = paired_ce + args.preservation_weight*block.last_penalty
            if not torch.isfinite(loss):
                raise ValueError('Non-finite paired scene training loss')
            (args.paired_weight*args._batch_weight*loss).backward()
            report['paired_ce'] = float(paired_ce.detach())
    return report


def experiment(args):
    import torch
    from moral_llava.polarity_runtime import PolaritySFTCollator
    from moral_llava.scene_guided_vision import (
        METHOD, caption_index, paired_text_rows, save_scene_guidance, load_scene_guidance)
    from moral_llava.text_guided_vision import verify_original_vision
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; scene 7B training did not start. CPU checks are not formal TICR.')
    env, raw, rows, audit = common.preflight(args)
    env.update(conditioning='same-scene caption0 during training; image-predicted scene condition during independent inference')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'starting', 'started_at': common.now(), 'pid': os.getpid(), 'method': METHOD}
    common.save(args.output/'status.json', state)
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, processor, block, index, integration, dtype, encoder = load_runtime(args, need_text=True)
        smoke = args.command == 'smoke'
        train = common.balanced_subset(rows['train'], 2) if smoke else [r for r in rows['train'] if r['modality'] == 'image']
        validation = common.balanced_subset(rows['validation'], 1, paired=True) if smoke else rows['validation']
        ids = {r['id'] for r in validation}
        validation_raw = [r for r in raw['validation'] if r['id'] in ids]
        captions = caption_index(raw['train'], split='train')
        source_paths = [Path(__file__), ROOT/'moral_llava/scene_guided_vision.py', ROOT/'run_text_guided_vision.py',
                        ROOT/'moral_llava/text_guided_vision.py', ROOT/'moral_llava/polarity.py',
                        ROOT/'moral_llava/polarity_runtime.py', ROOT/'moral_llava/train.py']
        config = {'method': METHOD, 'environment': env, 'guidance_config': asdict(block.adapter.config),
            'wrapped_block_index': index, 'trainable_parameters': sum(p.numel() for p in block.adapter.parameters()),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'source_sha256': {str(p.relative_to(ROOT)): common.sha(p) for p in source_paths},
            'data_sha256': {s: common.sha(args.data/f'{s}.json') for s in raw},
            'selection': 'independent validation TICR; accuracy/BA/joint/validity guards; zero epoch eligible',
            'target': 'beat same-run direct baseline TICR with preserved accuracy; also report archived original-tower reference'}
        common.save(args.output/'frozen_config.json', config)
        baseline = predictions(args, model, processor, block, dtype, validation,
                               enabled=False, split='validation', smoke=smoke)
        common.write_predictions(args.output/'baseline-validation.jsonl', baseline)
        baseline_metrics = score(validation_raw, baseline, diagnostic=smoke)
        common.save(args.output/'baseline-validation-metrics.json', baseline_metrics)
        zero = predictions(args, model, processor, block, dtype, validation[:20], enabled=True,
                           split='validation', smoke=True, guidance_hash='zero-init')
        if [p['raw_prediction'] for p in zero] != [p['raw_prediction'] for p in baseline[:20]]:
            raise ValueError('Scene zero initialization changed baseline generation')
        metadata = {'baseline_adapter_sha256': common.baseline_sha(args)}
        save_scene_guidance(args.output/'epoch-0', block, dict(metadata, epoch=0))
        selected, best = args.output/'epoch-0', scene_selection_key(baseline_metrics, baseline_metrics)
        optimizer = torch.optim.AdamW(block.adapter.parameters(), lr=args.learning_rate, weight_decay=0)
        collator = PolaritySFTCollator(processor, image_token_id=model.config.image_token_index,
            expected_image_tokens=model.config.image_seq_length, max_length=args.max_length)
        losses = []
        step = 0
        for epoch in range(1, (1 if smoke else args.epochs)+1):
            shuffled = list(train)
            random.Random(args.seed+epoch).shuffle(shuffled)
            if smoke:
                shuffled = shuffled[:args.smoke_steps*args.train_batch_size]
            model.eval()
            model.get_base_model().language_model.model.training = True
            block.adapter.train()
            processor.tokenizer.padding_side = 'right'
            batches = [shuffled[i:i+args.train_batch_size] for i in range(0, len(shuffled), args.train_batch_size)]
            optimizer.zero_grad()
            for i, batch in enumerate(batches):
                group_start = (i//args.gradient_accumulation)*args.gradient_accumulation
                group_count = sum(len(b) for b in batches[group_start:group_start+args.gradient_accumulation])
                args._batch_weight = len(batch)/group_count
                loss = train_batch(args, model, processor, block, dtype, encoder, captions, batch,
                                   paired_text_rows(batch, rows['train']), collator)
                if (i+1) % args.gradient_accumulation == 0 or i == len(batches)-1:
                    trainable = [p for p in block.adapter.parameters() if p.requires_grad]
                    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable):
                        raise ValueError('No gradient reached scene encoder')
                    torch.nn.utils.clip_grad_norm_(trainable, 1.)
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                losses.append(dict(loss, epoch=epoch, batch=i, optimizer_step=step))
                if i % 10 == 0:
                    state.update(status='training', epoch=epoch, batch=i, batches=len(batches), optimizer_step=step)
                    common.save(args.output/'status.json', state)
                    common.save(args.output/'training-loss.json', losses)
                    print(json.dumps(losses[-1]), flush=True)
            path = args.output/f'epoch-{epoch}'
            saved = save_scene_guidance(path, block, dict(metadata, epoch=epoch))
            generated = predictions(args, model, processor, block, dtype, validation, enabled=True,
                split='validation', smoke=smoke, guidance_hash=saved['weights_sha256'])
            common.write_predictions(args.output/f'epoch-{epoch}-validation.jsonl', generated)
            metrics = score(validation_raw, generated, diagnostic=smoke)
            common.save(args.output/f'epoch-{epoch}-validation-metrics.json', metrics)
            key = scene_selection_key(metrics, baseline_metrics)
            if key is not None and key > best:
                selected, best = path, key
        common.save(args.output/'training-loss.json', losses)
        saved = load_scene_guidance(selected, block)
        common.save(args.output/'selection.json', {'checkpoint': str(selected),
            'epoch': saved['metadata']['epoch'], 'weights_sha256': saved['weights_sha256'],
            'validation_key': best, 'locked_at': common.now()})
        if not smoke and not args.validation_only:
            all_outputs, all_metrics = {}, {}
            for name, enabled in (('baseline', False), ('guided', True)):
                output = predictions(args, model, processor, block, dtype, rows['test'], enabled=enabled,
                    split='test', smoke=False, guidance_hash=saved['weights_sha256'] if enabled else None)
                common.write_predictions(args.output/f'{name}-test.jsonl', output)
                metrics = score(raw['test'], output, audit)
                if not metrics['formal_test_result']:
                    raise ValueError('Incomplete independent formal test')
                common.save(args.output/f'{name}-metrics.json', metrics)
                all_outputs[name], all_metrics[name] = output, metrics
            unchanged_text_branch(all_outputs['baseline'], all_outputs['guided'])
            common.save(args.output/'comparison.json', comparison(args, raw['test'],
                all_outputs['baseline'], all_outputs['guided'], all_metrics))
            if args.evaluate_paired:
                paired = predictions(args, model, processor, block, dtype, rows['test'], enabled=True,
                    split='test', smoke=False, guidance_hash=saved['weights_sha256'], mode='paired',
                    encoder=encoder, captions=caption_index(raw['test'], split='test'))
                unchanged_text_branch(all_outputs['baseline'], paired)
                common.write_predictions(args.output/'paired-assisted-test.jsonl', paired)
                common.save(args.output/'paired-assisted-metrics.json', score(raw['test'], paired, audit))
        verified = verify_original_vision(model, integration)
        for filename, checksum in config['source_sha256'].items():
            if common.sha(ROOT/filename) != checksum:
                raise ValueError('Source changed during experiment: '+filename)
        state.update(status='complete', finished_at=common.now(), seconds=time.monotonic()-start,
            gpu_training_executed=True, formal_ticr_available=not smoke and not args.validation_only,
            selected_epoch=saved['metadata']['epoch'], frozen_visual_tensors_verified=verified,
            truncated_scene_captions=len(encoder.truncated_captions),
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        common.save(args.output/'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error), finished_at=common.now())
        common.save(args.output/'status.json', state)
        raise


def unchanged_text_branch(baseline, guided):
    index = {r['id']: r for r in baseline}
    if any(r['raw_prediction'] != index[r['id']]['raw_prediction'] for r in guided if r['modality'] == 'text'):
        raise ValueError('Frozen text-only branch changed')


def infer(args):
    from moral_llava.scene_guided_vision import load_scene_guidance, caption_index
    if args.guidance is None:
        raise ValueError('infer requires --guidance')
    raw, rows, audit = common.ensure_data(args)
    saved = json.loads((args.guidance/'guidance.json').read_text())
    if saved['metadata']['baseline_adapter_sha256'] != common.baseline_sha(args):
        raise ValueError('Scene module was trained with a different baseline adapter')
    args.rank = saved['config']['rank']
    args.output.mkdir(parents=True, exist_ok=False)
    model, processor, block, _, _, dtype, encoder = load_runtime(args, need_text=args.evaluate_paired)
    saved = load_scene_guidance(args.guidance, block)
    outputs, metrics = {}, {}
    for name, enabled in (('baseline', False), ('guided', True)):
        outputs[name] = predictions(args, model, processor, block, dtype, rows['test'], enabled=enabled,
            split='test', smoke=False, guidance_hash=saved['weights_sha256'] if enabled else None)
        common.write_predictions(args.output/f'{name}-test.jsonl', outputs[name])
        metrics[name] = score(raw['test'], outputs[name], audit)
        common.save(args.output/f'{name}-metrics.json', metrics[name])
    unchanged_text_branch(outputs['baseline'], outputs['guided'])
    common.save(args.output/'comparison.json', comparison(args, raw['test'], outputs['baseline'], outputs['guided'], metrics))
    if args.evaluate_paired:
        output = predictions(args, model, processor, block, dtype, rows['test'], enabled=True,
            split='test', smoke=False, guidance_hash=saved['weights_sha256'], mode='paired', encoder=encoder,
            captions=caption_index(raw['test'], split='test'))
        unchanged_text_branch(outputs['baseline'], output)
        common.write_predictions(args.output/'paired-assisted-test.jsonl', output)
        common.save(args.output/'paired-assisted-metrics.json', score(raw['test'], output, audit))


def predict(args):
    """Deployment on one user image, with an optional genuine scene description."""
    import torch
    from moral_llava.polarity import question
    from moral_llava.polarity_runtime import inputs_for_records
    from moral_llava.scene_guided_vision import SceneContext, load_scene_guidance, METHOD
    if args.guidance is None or args.image is None or not args.image.is_file():
        raise ValueError('predict requires --guidance and an existing --image')
    if args.scene_text is not None and not args.scene_text.strip():
        raise ValueError('A supplied scene description cannot be empty')
    saved = json.loads((args.guidance/'guidance.json').read_text())
    if saved['metadata']['baseline_adapter_sha256'] != common.baseline_sha(args):
        raise ValueError('Scene module was trained with a different baseline adapter')
    args.rank = saved['config']['rank']
    paired = args.scene_text is not None
    model, processor, block, _, _, dtype, encoder = load_runtime(args, need_text=paired)
    load_scene_guidance(args.guidance, block)
    processor.tokenizer.padding_side = 'left'
    record = {'modality': 'image', 'image_path': str(args.image),
              'prompt': f'USER: {question(args.task, "image")} ASSISTANT:'}
    inputs = inputs_for_records(processor, [record])
    if inputs['input_ids'].shape[1]+16 > args.max_length:
        raise ValueError('Deployment input exceeds context budget')
    features = encoder.encode([args.scene_text]) if paired else None
    context = SceneContext('paired' if paired else 'predicted', features)
    inputs = common.move_batch(inputs, dtype)
    width = inputs['input_ids'].shape[1]
    with torch.inference_mode(), block.conditioned(context):
        tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id)
    result = {'method': METHOD, 'task': args.task,
        'condition_source': 'provided_scene_description' if paired else 'image_predicted_scene_semantics',
        'raw_prediction': processor.batch_decode(tokens[:, width:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0],
        'formal_benchmark_result': False, 'guidance_sha256': saved['weights_sha256']}
    common.save(args.output/'prediction.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inspect', 'preflight', 'smoke', 'experiment', 'infer', 'predict'))
    p.add_argument('--output', type=Path)
    p.add_argument('--base-model', type=Path, default=Path(os.environ.get('BASE_SNAPSHOT', str(ROOT/'models/base/llava'))))
    p.add_argument('--moralclip-model', type=Path, default=Path(os.environ.get('MORALCLIP_MODEL', str(ROOT/'models/base/moralclip'))))
    p.add_argument('--baseline-adapter', type=Path, default=ROOT/'models/moralclip-llava-polarity')
    p.add_argument('--expected-baseline-sha256')
    p.add_argument('--guidance', type=Path)
    p.add_argument('--data', type=Path, default=ROOT/'data/polarity-v1')
    p.add_argument('--image-root', type=Path, default=ROOT/'data/raw/SMID_moralclip')
    p.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--train-batch-size', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=2)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--smoke-steps', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--distill-weight', type=float, default=.2)
    p.add_argument('--alignment-weight', type=float, default=.1)
    p.add_argument('--preservation-weight', type=float, default=.05)
    p.add_argument('--paired-weight', type=float, default=.5)
    p.add_argument('--temperature', type=float, default=2.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--evaluate-paired', action='store_true', help='Also report caption-assisted mode separately')
    p.add_argument('--validation-only', action='store_true', help='Train/select without reading test captions or generating test answers')
    p.add_argument('--image', type=Path, help='User image for predict')
    p.add_argument('--scene-text', help='Optional user-provided scene description for predict')
    p.add_argument('--task', choices=('care', 'fairness', 'loyalty', 'authority', 'sanctity'), default='care')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    for key in ('epochs', 'rank', 'train_batch_size', 'eval_batch_size', 'gradient_accumulation', 'max_length', 'smoke_steps'):
        if getattr(args, key) <= 0:
            p.error(key+' must be positive')
    for key in ('learning_rate', 'temperature', 'distill_weight', 'alignment_weight', 'preservation_weight', 'paired_weight'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key in ('learning_rate', 'temperature') and value == 0):
            p.error(key+' must be finite and nonnegative (positive for learning rate/temperature)')
    if args.command == 'inspect':
        report = inspect_data(args)
        if args.output:
            common.save(args.output/'data-inspection.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == 'preflight':
        report, *_ = common.preflight(args)
        report.update(conditioning='same-scene text supervision; image-predicted or paired-scene inference',
                      gpu_training_executed=False)
        if args.output:
            common.save(args.output/'preflight.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.output is None:
        p.error('--output is required for training/inference')
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'predict':
        if args.output.exists():
            p.error('predict output directory must be new')
        predict(args)
    else:
        experiment(args)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print('Scene-guided encoder:', error, file=sys.stderr)
        raise SystemExit(1)
