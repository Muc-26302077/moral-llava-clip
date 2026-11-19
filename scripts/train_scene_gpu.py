"""Time-bounded GPU research: recoverable training, validation selection, one test."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_training_report import read, atomic, render, TZ

TRAIN_END = datetime(2026, 11, 17, 7, 15, tzinfo=TZ)
HARD_END = datetime(2026, 11, 17, 7, 55, tzinfo=TZ)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume existing candidate checkpoints')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    prior = read(run/'orchestrator-status.json') or {}
    if args.resume and prior.get('phase') == 'complete':
        audit = read(run/'final/audit.json') or {}
        selection = read(run/'final-selection.json') or {}
        import hashlib
        weights = Path(selection['checkpoint'])/'guidance.safetensors'
        if (not audit.get('score_replay_matches') or not audit.get('text_branch_unchanged')
                or hashlib.sha256(weights.read_bytes()).hexdigest() != audit.get('locked_weight_sha256')
                or selection.get('weights_sha256') != audit.get('locked_weight_sha256')):
            raise RuntimeError('Completed run evidence changed; refusing a repeated test')
        render(run)
        print('Run already completed and audited; no training or test repeated.', flush=True)
        return
    state = {'pid': os.getpid(), 'phase': 'GPU恢复检查', 'candidates': []}
    def update(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now(TZ).isoformat(), **extra)
        atomic(run/'orchestrator-status.json', json.dumps(state, ensure_ascii=False, indent=2))
        render(run)
        print(phase, flush=True)
    def call(command, name):
        with (run/(name+'.log')).open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', 'run_scene_guided_vision.py', *command],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(name, child_pid=child.pid)
            try:
                code = child.wait(timeout=max(1, (HARD_END-datetime.now(TZ)).total_seconds()-10))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise RuntimeError('07:55 deadline reached; child stopped; inspect last recovery checkpoint')
        if code:
            raise RuntimeError(f'{name} exited with {code}; see {run/(name+".log")}')
    base = ['--base-model', os.environ['BASE_SNAPSHOT'], '--moralclip-model', os.environ['MORALCLIP_MODEL'],
        '--expected-baseline-sha256', '318b609354f115162f14de9136005ded3cde31865dda8823559b1e0b79f79e2c',
        '--eval-batch-size', '1']
    try:
        import torch
        smoke_name, batch, accumulation, smoke = 'smoke-b4', 4, 2, None
        for cached_batch in (4, 2, 1):
            cached_name = f'smoke-b{cached_batch}'
            cached = read(run/cached_name/'status.json') or {}
            if cached.get('status') == 'complete' and cached.get('resumed_optimizer_step') == 1:
                smoke_name, batch, accumulation, smoke = cached_name, cached_batch, 8//cached_batch, cached
                break
        if smoke is None:
            import shutil
            for batch in (4, 2, 1):
                accumulation, smoke_name = 8//batch, f'smoke-b{batch}'
                smoke_command = ['smoke', *base, '--output', str(run/smoke_name),
                    '--train-batch-size', str(batch), '--gradient-accumulation', str(accumulation),
                    '--smoke-steps', str(2*accumulation)]
                try:
                    call(smoke_command+['--max-optimizer-steps', '1'], smoke_name+'-initial')
                    shutil.copy2(run/smoke_name/'training-state.pt', run/'smoke-step-1.pt')
                    call(smoke_command+['--resume', str(run/smoke_name/'training-state.pt')], smoke_name+'-resume')
                    smoke = read(run/smoke_name/'status.json')
                    if not smoke or smoke.get('status') != 'complete':
                        raise RuntimeError('Smoke recovery did not complete')
                    break
                except RuntimeError:
                    logs = '\n'.join(p.read_text() for p in run.glob(smoke_name+'*.log'))
                    if batch == 1 or ('out of memory' not in logs.lower() and 'OutOfMemoryError' not in logs):
                        raise
        first = torch.load(run/'smoke-step-1.pt', map_location='cpu', weights_only=False)
        final = torch.load(run/smoke_name/'training-state.pt', map_location='cpu', weights_only=False)
        changed = any(not torch.equal(v, final['adapter'][k]) for k, v in first['adapter'].items())
        verification = {'initial_step': first['step'], 'recovered_final_step': final['step'],
            'parameters_changed_after_resume': changed,
            'frozen_visual_tensors_verified': smoke['frozen_visual_tensors_verified'],
            'batch_size': batch, 'cuda_peak_allocated_gib': smoke['cuda_peak_allocated_gib']}
        if final['step'] <= first['step'] or not changed or smoke['frozen_visual_tensors_verified'] != 199:
            raise RuntimeError('GPU recovery verification failed')
        atomic(run/'recovery-verification.json', json.dumps(verification, indent=2))
        del first, final
        common = base+['--train-batch-size', str(batch), '--gradient-accumulation', str(accumulation)]
        def train(name, *, epochs, rank, lr, preservation):
            state['candidates'].append(name)
            command = ['experiment', *common, '--validation-only', '--epochs', str(epochs),
                '--rank', str(rank), '--learning-rate', str(lr), '--preservation-weight', str(preservation),
                '--checkpoint-every', '50', '--stop-at', TRAIN_END.isoformat(), '--output', str(run/name)]
            if (run/name).exists():
                if not args.resume:
                    raise RuntimeError('Candidate already exists; use --resume or a new run directory')
                prior = read(run/name/'status.json') or {}
                if prior.get('status') == 'complete':
                    return
                command += ['--resume', str(run/name/'training-state.pt')]
            call(command, name)
        train('candidate-a', epochs=3, rank=32, lr=2e-4, preservation=.05)
        import run_scene_guided_vision as runner
        def selected(name):
            selection = read(run/name/'selection.json')
            b = read(run/name/'baseline-validation-metrics.json')
            if not selection or not b:
                raise RuntimeError('Incomplete baseline/selection evidence: '+name)
            baseline = b['overall']['primary_macro']
            return selection, baseline
        selection, baseline = selected('candidate-a')
        a_key = tuple(selection['validation_key'])
        if a_key[0] <= baseline['ticr']:
            metrics = [read(p) for p in (run/'candidate-a').glob('epoch-*-validation-metrics.json')]
            quality_regression = any(m and m['overall']['primary_macro']['ticr'] > baseline['ticr']
                and runner.scene_selection_key(m, read(run/'candidate-a/baseline-validation-metrics.json')) is None for m in metrics)
            losses = read(run/'candidate-a/training-loss.json') or []
            # Conservative throughput forecast; include full validation per epoch and model load.
            seconds_per_batch = ((losses[-1]['elapsed_seconds']-losses[0]['elapsed_seconds'])/max(1, len(losses)-1)) if len(losses)>1 else float('inf')
            forecast = 2*(1226*seconds_per_batch+360)+120
            now = datetime.now(TZ)
            if now.hour < 7 and forecast < (TRAIN_END-now).total_seconds():
                if quality_regression:
                    train('candidate-b-conservative', epochs=2, rank=32, lr=5e-5, preservation=.1)
                else:
                    train('candidate-b-capacity', epochs=2, rank=64, lr=5e-4, preservation=.05)
        best_name, best_selection = 'candidate-a', selection
        for name in state['candidates'][1:]:
            current, _ = selected(name)
            if tuple(current['validation_key']) > tuple(best_selection['validation_key']):
                best_name, best_selection = name, current
        lock = dict(best_selection, candidate=best_name, selection_policy='validation only; earlier candidate wins ties',
                    locked_at=datetime.now(TZ).isoformat())
        atomic(run/'final-selection.json', json.dumps(lock, indent=2))
        call(['infer', *common, '--guidance', best_selection['checkpoint'], '--evaluate-paired',
              '--output', str(run/'final')], 'final')
        raw = read(ROOT/'data/polarity-v1/test.json')
        audit = read(ROOT/'data/polarity-v1/audit.json')
        predictions = {}
        for name in ('baseline', 'guided'):
            rows = [json.loads(line) for line in (run/f'final/{name}-test.jsonl').read_text().splitlines()]
            result = runner.score(raw, rows, audit)
            if len(rows) != 576 or not result['formal_test_result'] or result != read(run/f'final/{name}-metrics.json'):
                raise RuntimeError('Independent score replay failed: '+name)
            if any(r['adapter_sha256'] != '318b609354f115162f14de9136005ded3cde31865dda8823559b1e0b79f79e2c' for r in rows):
                raise RuntimeError('Prediction baseline hash mismatch')
            if name == 'guided' and any(r['guidance_sha256'] != best_selection['weights_sha256'] for r in rows):
                raise RuntimeError('Prediction guidance hash mismatch')
            predictions[name] = rows
        runner.unchanged_text_branch(predictions['baseline'], predictions['guided'])
        from run_text_guided_vision import sha
        if sha(Path(best_selection['checkpoint'])/'guidance.safetensors') != best_selection['weights_sha256']:
            raise RuntimeError('Locked model weight hash mismatch')
        atomic(run/'final/audit.json', json.dumps({'score_replay_matches': True,
            'predictions_per_model': 576, 'locked_weight_sha256': best_selection['weights_sha256'],
            'text_branch_unchanged': True, 'selection_used_test_results': False}, indent=2))
        update('complete', finished_at=datetime.now(TZ).isoformat())
    except BaseException as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
