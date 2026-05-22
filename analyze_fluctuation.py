"""Compare per-iteration reward fluctuation between v9 and v10 runs.

Computes: for each reward tag, the rolling-window CV (coefficient of variation)
and the iteration-to-iteration absolute change, normalised by mean. Also dumps
mean episode length and noise std evolution so we can see whether the
"fluctuation" is driven by termination variance.
"""
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

RUNS = {
    'A1 walking (May12 01:00)': 'logs/A1_ppo/May12_01-00-01_v1',
    'A1 walking (May09 11:19)': 'logs/A1_ppo/May09_11-19-25_v1',
    'v10 jump (May18 11:51)': 'logs/A1_jump_ppo/May18_11-51-12_v1',
}


def load(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={'scalars': 0})
    ea.Reload()
    return ea


def vec(ea, tag):
    return np.array([e.value for e in ea.Scalars(tag)]) if tag in ea.Tags()['scalars'] else None


def stats(name, ea, max_iter=3400):
    iters = np.array([e.step for e in ea.Scalars('Train/mean_reward')])
    mask = iters <= max_iter
    iters = iters[mask]
    print(f"\n===== {name} (first {mask.sum()} iters, max_step={iters[-1]}) =====")
    rew = vec(ea, 'Train/mean_reward')[mask]
    ep_len = vec(ea, 'Train/mean_episode_length')[mask]
    noise = vec(ea, 'Policy/mean_noise_std')[mask]
    # Window 200-iter standard deviation, normalized by |mean|
    def cv(x, w=200):
        out = []
        for i in range(w, len(x)):
            seg = x[i-w:i]
            m = abs(seg.mean()) + 1e-6
            out.append(seg.std() / m)
        return np.array(out)
    print(f"  Final mean_reward = {rew[-1]:.2f}, ep_len = {ep_len[-1]:.1f}, noise = {noise[-1]:.3f}")
    print(f"  mean_reward rolling-200 CV: median={np.median(cv(rew)):.3f}, p90={np.percentile(cv(rew),90):.3f}")
    print(f"  ep_length   rolling-200 CV: median={np.median(cv(ep_len)):.3f}, p90={np.percentile(cv(ep_len),90):.3f}")

    # Per-reward-term variance over iters 1500-3400 (after settle)
    s = max(1500, w := 200)
    e = min(3400, len(iters))
    print(f"\n  Per-term rolling-{w} CV in [{s},{e}] (top-10 most volatile, abs(mean)>0.001):")
    items = []
    for tag in sorted(t for t in ea.Tags()['scalars'] if t.startswith('Episode/rew_')):
        v = vec(ea, tag)[mask]
        if len(v) < e:
            continue
        seg = v[s:e]
        m = abs(seg.mean())
        if m < 0.001:
            continue
        items.append((tag.replace('Episode/rew_',''), seg.mean(), seg.std(), seg.std()/(m+1e-6)))
    items.sort(key=lambda r: -r[3])
    print(f"    {'tag':<22} {'mean':>10} {'std':>10} {'CV':>8}")
    for tag, m, s_, c in items[:15]:
        print(f"    {tag:<22} {m:>10.4f} {s_:>10.4f} {c:>8.2f}")


for name, path in RUNS.items():
    try:
        ea = load(path)
        stats(name, ea)
    except Exception as ex:
        print(f"\n[skip {name}: {ex}]")
