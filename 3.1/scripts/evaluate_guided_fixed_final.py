"""Report fixed final-epoch weights separately from validation selection.

This optional comparison is planned before final testing and never selects a
checkpoint or changes the primary report. Requires a complete primary run.
"""
import argparse
from argparse import Namespace
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from run_text_guided_vision import (load_runtime, preflight, predictions,
        metrics_for, write_predictions, save, sha, now)
    from moral_llava.text_guided_vision import load_guidance, verify_original_vision, question_conditions
    from moral_llava.binary_runtime import inputs_for_records
    from moral_llava.guided_report import read_jsonl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    cli = parser.parse_args()
    root = cli.run_root.resolve()
    primary = root / 'formal'
    plan = json.loads((root / 'fixed-final-evaluation-plan.json').read_text())
    status = json.loads((primary / 'status.json').read_text())
    selection = json.loads((primary / 'selection.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Primary experiment must complete first')
    if datetime.fromisoformat(plan['created_at']) >= datetime.fromisoformat(selection['locked_at']):
        raise ValueError('Supplemental evaluation must be planned before selection/test')
    already_evaluated = selection['epoch'] == plan['epoch'] and plan['skip_if_selected_epoch_is_3']
    config = json.loads((primary / 'frozen_config.json').read_text())
    args = Namespace(**config['arguments'])
    for key in ('base_model', 'baseline_adapter', 'moralclip_model', 'data', 'image_root', 'output'):
        setattr(args, key, Path(getattr(args, key)))
    _, raw, rows, audit = preflight(args)
    if sha(args.data / 'test.json') != plan['test_data_sha256']:
        raise ValueError('Supplemental frozen test mismatch')
    for name, checksum in config['source_sha256'].items():
        if sha(ROOT / name) != checksum:
            raise ValueError('Primary source changed: ' + name)
    destination = root / 'fixed-final-evaluation'
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    state = {'status': 'running', 'started_at': now(), 'epoch': plan['epoch'],
             'role': 'fixed_final_epoch_supplement_not_used_for_selection',
             'evaluator_sha256': sha(Path(__file__)), 'primary_selected_epoch': selection['epoch'],
             'reused_primary_predictions': already_evaluated}
    save(destination / 'status.json', state)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, processor, block, _, integration, dtype = load_runtime(args)
        metadata = load_guidance(primary / f'epoch-{plan["epoch"]}', block)
        if already_evaluated:
            outputs = read_jsonl(primary / 'guided-test.jsonl')
        else:
            outputs = predictions(args, model, processor, block, dtype, rows['test'], enabled=True,
                guidance_hash=metadata['weights_sha256'], split='test', smoke=False)
        write_predictions(destination / 'final-trained-test.jsonl', outputs)
        metrics = metrics_for(raw['test'], outputs, audit)
        if not metrics['formal_test_result']:
            raise ValueError('Incomplete fixed final test')
        baseline = {p['id']: p for p in read_jsonl(primary / 'baseline-test.jsonl')}
        if any(p['raw_prediction'] != baseline[p['id']]['raw_prediction']
               for p in outputs if p['modality'] == 'text'):
            raise ValueError('Frozen text branch changed in fixed-final evaluation')
        save(destination / 'metrics.json', metrics)
        save(destination / 'guidance-metadata.json', metadata)
        # A training image, with two different task questions, verifies the
        # actual trained encoder's condition dependence in deployment dtype.
        original = next(r for r in rows['train'] if r['modality'] == 'image')
        swapped = next(r for r in rows['train'] if r['modality'] == 'image'
                       and r['task'] != original['task'])
        pixels = inputs_for_records(processor, [original])['pixel_values'].to('cuda', dtype=dtype)
        features, conditions = [], []
        with torch.inference_mode():
            for question in (original, swapped):
                condition = question_conditions(model, processor.tokenizer, [question])
                conditions.append(condition)
                with block.conditioned(condition):
                    encoded = model.get_base_model().vision_tower(pixel_values=pixels, output_hidden_states=True)
                    features.append(encoded.hidden_states[model.config.vision_feature_layer][:, 1:].float())
        difference = (features[0] - features[1]).abs()
        save(destination / 'question-conditioning-check.json', {
            'image_record_id': original['id'], 'swapped_question_record_id': swapped['id'],
            'image_sha256': sha(original['image_path']), 'image_unchanged': True,
            'condition_mean_abs_difference': float((conditions[0]-conditions[1]).abs().mean()),
            'selected_visual_feature_mean_abs_difference': float(difference.mean()),
            'selected_visual_feature_max_abs_difference': float(difference.max()),
            'selected_visual_feature_changed_fraction': float((difference > 0).float().mean()),
            'precision': str(dtype),
            'interpretation': 'Condition-path diagnostic on one training image; not evidence of accuracy or TICR improvement.'})
        state.update(status='complete', finished_at=now(), seconds=time.monotonic()-started,
            frozen_visual_tensors_verified=verify_original_vision(model, integration),
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(destination / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
    except BaseException as error:
        state.update(status='failed', error=str(error), finished_at=now())
        save(destination / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
