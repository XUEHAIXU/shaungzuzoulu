"""A1-legs_V1 (12-DoF biped) MuJoCo sim2sim deployment.

Port of https://github.com/.../mujoco_sim/deploy_a1_legs.py to this repo.
Obs layout (47-dim, frame_stack=15) matches A1Cfg training; MJCF joint order
matches URDF traversal (R1..R6, L1..L6) so no permutation is needed.

Run under the `mujoco` conda env (Python 3.11 + mujoco 3.5 + torch 2.x + pyyaml).

Keyboard (non-blocking, raw stdin):
  w / s : vx +/- 0.1
  a / d : yaw +/- 0.1 (a=left, d=right)
  j / l : vy +/- 0.1 (j=left, l=right)
  space : reset cmd
  r     : reset robot state
"""

import math
import os
import sys
import time
import fcntl
import termios
import tty
from string import Template

import numpy as np
import mujoco
import mujoco.viewer
import torch
import yaml

from scipy.spatial.transform import Rotation as R

from humanoid import LEGGED_GYM_ROOT_DIR


# ------------------------------ obs history (gym, time-major) ------------------------------
class ObsHistoryTimeMajor:
    def __init__(self, num_obs: int, hist_len: int):
        self.num_obs = num_obs
        self.hist_len = hist_len
        self.buf = np.zeros(num_obs * hist_len, dtype=np.float32)

    def reset(self):
        self.buf[:] = 0.0

    def update(self, new_obs: np.ndarray) -> np.ndarray:
        self.buf[:-self.num_obs] = self.buf[self.num_obs:]
        self.buf[-self.num_obs:] = new_obs.astype(np.float32)
        return self.buf.copy()


# ------------------------------ keyboard (termios non-blocking) ------------------------------
_fd = sys.stdin.fileno()
_old_term = termios.tcgetattr(_fd)
tty.setcbreak(_fd)
_old_flags = fcntl.fcntl(_fd, fcntl.F_GETFL)
fcntl.fcntl(_fd, fcntl.F_SETFL, _old_flags | os.O_NONBLOCK)


def restore_terminal():
    termios.tcsetattr(_fd, termios.TCSADRAIN, _old_term)
    fcntl.fcntl(_fd, fcntl.F_SETFL, _old_flags)


x_vel_cmd = 0.0
y_vel_cmd = 0.0
yaw_vel_cmd = 0.0
reset_requested = False


def handle_key():
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, reset_requested
    try:
        ch = sys.stdin.read(1)
    except IOError:
        return
    if not ch:
        return
    if ch == 'w':
        x_vel_cmd += 0.1
    elif ch == 's':
        x_vel_cmd -= 0.1
    elif ch == 'a':
        yaw_vel_cmd += 0.1
    elif ch == 'd':
        yaw_vel_cmd -= 0.1
    elif ch == 'j':
        y_vel_cmd += 0.1
    elif ch == 'l':
        y_vel_cmd -= 0.1
    elif ch == ' ':
        x_vel_cmd = y_vel_cmd = yaw_vel_cmd = 0.0
    elif ch == 'r':
        reset_requested = True


# ------------------------------ math helpers ------------------------------
def quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    # mujoco stores quat as [w, x, y, z]; scipy uses [x, y, z, w]
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    eu = r.as_euler('xyz', degrees=False)
    eu = np.where(eu > math.pi, eu - 2 * math.pi, eu)
    return eu.astype(np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _expand_path(p: str) -> str:
    return Template(p).safe_substitute(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)


# ------------------------------ main ------------------------------
def main():
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, reset_requested

    config_file = sys.argv[1] if len(sys.argv) > 1 else 'a1.yaml'
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', config_file)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    policy_path = _expand_path(cfg['policy_path'])
    xml_path = _expand_path(cfg['xml_path'])
    sim_duration = float(cfg['simulation_duration'])
    sim_dt = float(cfg['simulation_dt'])
    decimation = int(cfg['control_decimation'])

    num_actions = int(cfg['num_actions'])
    num_single_obs = int(cfg['num_single_obs'])
    frame_stack = int(cfg['frame_stack'])
    cycle_time = float(cfg['cycle_time'])
    action_scale = float(cfg['action_scale'])
    clip_obs = float(cfg['clip_observations'])
    clip_act = float(cfg['clip_actions'])

    lin_vel_scale = float(cfg['lin_vel_scale'])
    ang_vel_scale = float(cfg['ang_vel_scale'])
    dof_pos_scale = float(cfg['dof_pos_scale'])
    dof_vel_scale = float(cfg['dof_vel_scale'])
    quat_scale = float(cfg['quat_scale'])

    kps = np.array(cfg['kps'], dtype=np.float64)
    kds = np.array(cfg['kds'], dtype=np.float64)
    tau_limit = np.array(cfg['tau_limit'], dtype=np.float64)
    default_angles = np.array(cfg['default_angles'], dtype=np.float64)
    cmd_init = np.array(cfg['cmd_init'], dtype=np.float32)
    warmup_steps = int(cfg.get('warmup_steps', 100))

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = float(cmd_init[0]), float(cmd_init[1]), float(cmd_init[2])

    print(f'[A1 deploy] xml_path    = {xml_path}')
    print(f'[A1 deploy] policy_path = {policy_path}')
    print(f'[A1 deploy] num_single_obs = {num_single_obs}, frame_stack = {frame_stack}')

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # free joint occupies qpos[0:7], qvel[0:6]
    d.qpos[7:7 + num_actions] = default_angles
    mujoco.mj_forward(m, d)
    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()

    policy = torch.jit.load(policy_path)
    policy.eval()

    obs_hist = ObsHistoryTimeMajor(num_single_obs, frame_stack)
    actions = np.zeros(num_actions, dtype=np.float32)
    target_q = np.zeros(num_actions, dtype=np.float64)  # policy target before default offset

    counter = 0
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    print('Controls: w/s vx, a/d yaw, j/l vy, space=stop, r=reset')

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.5])

        start_wall = time.time()
        while viewer.is_running() and time.time() - start_wall < sim_duration:
            step_start = time.time()

            if reset_requested:
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
                d.xfrc_applied[:] = 0.0
                actions[:] = 0.0
                target_q[:] = 0.0
                obs_hist.reset()
                x_vel_cmd = y_vel_cmd = yaw_vel_cmd = 0.0
                counter = 0
                mujoco.mj_forward(m, d)
                viewer.sync()
                reset_requested = False
                print('\n[reset] robot state reset')
                continue

            handle_key()

            q = d.qpos[7:7 + num_actions].astype(np.float64)
            dq = d.qvel[6:6 + num_actions].astype(np.float64)
            quat_wxyz = d.qpos[3:7].astype(np.float64)
            omega_body = d.qvel[3:6].astype(np.float64)
            eu = quat_wxyz_to_euler_xyz(quat_wxyz)

            if counter < warmup_steps:
                tau = pd_control(default_angles, q, kps, np.zeros(num_actions), dq, kds)
            else:
                tau = pd_control(target_q + default_angles, q, kps,
                                 np.zeros(num_actions), dq, kds)
            tau = np.clip(tau, -tau_limit, tau_limit)
            d.ctrl[:] = tau

            mujoco.mj_step(m, d)

            if counter % decimation == 0:
                obs = np.zeros(num_single_obs, dtype=np.float32)
                phase = counter * sim_dt / cycle_time
                obs[0] = math.sin(2.0 * math.pi * phase)
                obs[1] = math.cos(2.0 * math.pi * phase)
                obs[2] = x_vel_cmd * lin_vel_scale
                obs[3] = y_vel_cmd * lin_vel_scale
                obs[4] = yaw_vel_cmd * ang_vel_scale
                obs[5:17] = (q - default_angles) * dof_pos_scale
                obs[17:29] = dq * dof_vel_scale
                obs[29:41] = actions
                obs[41:44] = omega_body * ang_vel_scale
                obs[44:47] = eu * quat_scale

                obs = np.clip(obs, -clip_obs, clip_obs)
                stacked = obs_hist.update(obs)

                with torch.no_grad():
                    out = policy(torch.from_numpy(stacked).unsqueeze(0).float()).squeeze(0)
                actions = np.clip(out.detach().cpu().numpy(), -clip_act, clip_act).astype(np.float32)
                target_q = actions.astype(np.float64) * action_scale

            print(f'\rvx={x_vel_cmd:+.2f}  vy={y_vel_cmd:+.2f}  yaw={yaw_vel_cmd:+.2f}  '
                  f'base_z={d.qpos[2]:.3f}  step={counter}   ', end='')

            viewer.sync()
            counter += 1

            elapsed = time.time() - step_start
            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)

    print()


if __name__ == '__main__':
    try:
        main()
    finally:
        restore_terminal()
