from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(
    'logs/A1_jump_ppo/May17_21-18-23_v1', size_guidance={'scalars': 0}
)
ea.Reload()


def s(tag):
    return [(e.step, e.value) for e in ea.Scalars(tag)]


ep_len = s('Train/mean_episode_length')
ep_rew = s('Train/mean_reward')
noise = s('Policy/mean_noise_std')
print(f"Total iterations: {len(ep_len)}, last iter: {ep_len[-1][0]}")
print(f"\n{'iter':>6} {'len':>8} {'rew':>10} {'noise':>10}")
last = ep_len[-1][0]
checkpoints = sorted(set([0, 50, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, last]))
checkpoints = [c for c in checkpoints if c <= last]
for c in checkpoints:
    j = next((i for i, (st, _) in enumerate(ep_len) if st >= c), len(ep_len) - 1)
    print(f"  {ep_len[j][0]:>5d} {ep_len[j][1]:>8.2f} {ep_rew[j][1]:>10.3f} {noise[j][1]:>10.4f}")

print('\nNoise trajectory (every 1000 iters):')
for i in range(0, len(noise), 1000):
    print(f"  iter {noise[i][0]:>5d}: noise={noise[i][1]:.4f}")
