from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(
    'logs/A1_jump_ppo/May16_16-56-20_v1', size_guidance={'scalars': 0}
)
ea.Reload()
tags = sorted(t for t in ea.Tags()['scalars'] if t.startswith('Episode/rew_'))
checkpoints = [200, 1000, 5000, 15000, 29999]
hdr = f"{'reward':<28}" + ''.join(f'{c:>11}' for c in checkpoints)
print(hdr)
print('-' * len(hdr))
tp = {c: 0.0 for c in checkpoints}
tn = {c: 0.0 for c in checkpoints}
for tag in tags:
    short = tag.replace('Episode/rew_', '')
    es = ea.Scalars(tag)
    vals = {e.step: e.value for e in es}
    line = f'{short:<28}'
    for c in checkpoints:
        best = min(
            vals.keys(),
            key=lambda k: abs(k - c) if k <= c + 50 else 999999,
        )
        v = vals[best]
        line += f'{v:>11.4f}'
        if v < 0:
            tn[c] += v
        else:
            tp[c] += v
    print(line)
print('-' * len(hdr))
print(f"{'TOTAL +':<28}" + ''.join(f'{tp[c]:>11.4f}' for c in checkpoints))
print(f"{'TOTAL -':<28}" + ''.join(f'{tn[c]:>11.4f}' for c in checkpoints))
