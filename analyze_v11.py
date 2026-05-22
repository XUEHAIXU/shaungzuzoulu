"""Quick analysis of v11 (3-segment leg-driven jump) training run."""
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

ea = event_accumulator.EventAccumulator(
    'logs/A1_jump_ppo/May18_16-25-17_v1', size_guidance={'scalars': 0}
)
ea.Reload()


def s(tag):
    return [(e.step, e.value) for e in ea.Scalars(tag)] if tag in ea.Tags()['scalars'] else []


ep_len = s('Train/mean_episode_length')
ep_rew = s('Train/mean_reward')
noise = s('Policy/mean_noise_std')

if not ep_len:
    print("No data yet")
    exit()

last_iter = ep_len[-1][0]
print(f"v11 run state: iter {last_iter}, mean_rew {ep_rew[-1][1]:.2f}, "
      f"ep_len {ep_len[-1][1]:.1f}, noise {noise[-1][1]:.4f}")

cps = sorted(set([0, 50, 100, 200, 300, 400, last_iter]))
cps = [c for c in cps if c <= last_iter]
print(f"\n{'iter':>6} {'len':>8} {'rew':>10} {'noise':>10}")
for c in cps:
    j = next((i for i, (st, _) in enumerate(ep_len) if st >= c), len(ep_len)-1)
    print(f"  {ep_len[j][0]:>5d} {ep_len[j][1]:>8.2f} {ep_rew[j][1]:>10.3f} {noise[j][1]:>10.4f}")

# Per-term reward at first/middle/last available checkpoint
tags = sorted(t for t in ea.Tags()['scalars'] if t.startswith('Episode/rew_'))
print("\nPer-term episode reward:")
print(f"  {'term':<24}" + ''.join(f'{c:>10d}' for c in cps))
for tag in tags:
    es = ea.Scalars(tag)
    vals = {e.step: e.value for e in es}
    line = f"  {tag.replace('Episode/rew_',''):<24}"
    for c in cps:
        # nearest step <= c
        ks = [k for k in vals.keys() if k <= c + 50]
        if not ks:
            line += f"{'-':>10}"
            continue
        line += f"{vals[max(ks)]:>10.4f}"
    print(line)

# Stability metrics over last 100 iters
print("\nStability (last 100 iters rolling-window CV):")
def cv(x, w=50):
    if len(x) < w + 1:
        return None
    out = []
    for i in range(w, len(x)):
        seg = x[i-w:i]
        m = abs(seg.mean()) + 1e-6
        out.append(seg.std() / m)
    return np.array(out)

rew_arr = np.array([v for _, v in ep_rew])
len_arr = np.array([v for _, v in ep_len])
for label, arr in [('mean_reward', rew_arr), ('ep_length', len_arr)]:
    c = cv(arr, 50)
    if c is not None:
        print(f"  {label:<14} median CV = {np.median(c[-50:]):.3f}, p90 = {np.percentile(c[-50:],90):.3f}")
