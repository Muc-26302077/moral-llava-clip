"""Independent raw-answer recount and scene bootstrap."""
import math
import random
from moral_llava import polarity

def independent_audit(data, predictions, metrics):
    """Recompute metrics from raw label strings, independently of the scorer."""
    index = {r['id']: r for r in predictions}
    if len(index) != len(predictions) or set(index) != {r['id'] for r in data}:
        raise ValueError('Independent prediction coverage check failed')
    answers = {k: (p['raw_prediction'].strip() if isinstance(p['raw_prediction'], str) and
                   p['raw_prediction'].strip() in polarity.ANSWERS[p['task']].values() else None)
               for k, p in index.items()}
    bootstrap = {}
    for group in ('overall', 'A', 'B'):
        selected = [r for r in data if group == 'overall' or r['group'] == group]
        by_task = {}
        scene_pairs = {}
        for task in polarity.TASKS:
            rows = [r for r in selected if r['task'] == task]
            truth = {r['scene_id']: r['conversations'][1]['value'] for r in rows}
            pairs = {}
            for row in rows:
                pairs.setdefault(row['scene_id'], {})[row['modality']] = answers[row['id']]
            n = len(pairs)
            consistent = {s: p['image'] is not None and p['image'] == p['text'] for s, p in pairs.items()}
            ticr = sum(consistent.values())/n
            joint = sum(p['image'] == truth[s] and p['text'] == truth[s] for s, p in pairs.items())/n
            t = {'ticr': ticr, 'joint_accuracy': joint}
            for modality in ('image', 'text'):
                t[modality+'_accuracy'] = sum(p[modality] == truth[s] for s, p in pairs.items())/n
                recalls = [sum(p[modality] == label for s, p in pairs.items() if truth[s] == label)/
                           sum(v == label for v in truth.values()) for label in polarity.ANSWERS[task].values()]
                t[modality+'_balanced_accuracy'] = sum(recalls)/2
            by_task[task] = t
            for scene, value in consistent.items():
                scene_pairs.setdefault(scene, {})[task] = value
        ref = metrics['overall'] if group == 'overall' else metrics['groups'][group]
        for key in next(iter(by_task.values())):
            value = sum(t[key] for t in by_task.values())/5
            if not math.isclose(value, ref['primary_macro'][key], abs_tol=1e-12):
                raise ValueError(f'Independent recomputation differs: {group}/{key}')
        # Sample scenes together; five task denominators remain conditional and
        # unequal. Never pool all scene-task pairs as independent observations.
        rng = random.Random(42)
        scenes = sorted(scene_pairs)
        samples = []
        while len(samples) < 2000:
            draws = rng.choices(scenes, k=len(scenes))
            rates = []
            for task in polarity.TASKS:
                values = [scene_pairs[s][task] for s in draws if task in scene_pairs[s]]
                if not values:
                    break
                rates.append(sum(values)/len(values))
            if len(rates) == 5:
                samples.append(sum(rates)/5)
        samples.sort()
        bootstrap[group] = [samples[49], samples[1949]]
    return bootstrap
