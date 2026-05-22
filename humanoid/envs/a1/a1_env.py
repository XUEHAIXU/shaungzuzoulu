from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot
from humanoid.envs.a1.a1_config import A1Cfg
from humanoid.utils.terrain import HumanoidTerrain


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class A1FreeEnv(LeggedRobot):

    def __init__(
        self, cfg: A1Cfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
        )

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )
        self.reset_buf |= torch.any(
            torch.abs(self.projected_gravity[:, 0:1]) > 0.8, dim=1
        )
        self.reset_buf |= torch.any(
            torch.abs(self.projected_gravity[:, 1:2]) > 0.8, dim=1
        )
        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.35, dim=1)
        self.reset_buf |= self.time_out_buf

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0])
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0])
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 0] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 6] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_action = 2 * self.ref_dof_pos

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:5] = 0.0
        noise_vec[5:17] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[17:29] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[29:41] = 0.0
        noise_vec[41:44] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[44:47] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += (
            self.cfg.domain_rand.dynamic_randomization
            * torch.randn_like(actions)
            * actions
        )
        return super().step(actions)

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0]).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase + self.random_half_phase[0]).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1
        )

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,
                (self.dof_pos - self.default_joint_pd_target)
                * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                diff,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_euler_xyz * self.obs_scales.quat,
                self.rand_push_force[:, :2],
                self.rand_push_torque,
                self.env_frictions,
                self.body_mass / 30.0,
                stance_mask,
                contact_mask,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.command_input,
                q,
                dq,
                self.actions,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_euler_xyz * self.obs_scales.quat,
            ),
            dim=-1,
        )

        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = (
                obs_buf.clone()
                + torch.randn_like(obs_buf)
                * self.noise_scale_vec
                * self.cfg.noise.noise_level
            )
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1
        )

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ========== Rewards ========== #
    def _reward_joint_pos(self):
        selected_columns = [0, 3, 4, 6, 9, 10]
        diff = (self.dof_pos - self.ref_dof_pos)[:, selected_columns]
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(
            diff, dim=1
        ).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return torch.exp(-torch.abs(d_min) * 20) * torch.exp(-torch.abs(d_max) * 20)

    def _reward_knee_distance(self):
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist * 2.0
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return torch.exp(-torch.abs(d_min) * 20) * torch.exp(-torch.abs(d_max) * 20)

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(
            self.rigid_state[:, self.feet_indices, 10:12], dim=2
        )
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(
            torch.logical_or(contact, stance_mask), self.last_contacts
        )
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_default_hip_roll_joint_pos(self):
        # R2, L2 hip roll — pure squared penalty (any deviation hurts).
        # Pair with a negative scale in the config.
        return torch.sum(self.dof_pos[:, [1, 7]] ** 2, dim=1)

    def _reward_ankle_roll_flat(self):
        # R6, L6 ankle roll — penalise so policy can't use ankles to
        # compensate for hip adduction.
        return torch.sum(self.dof_pos[:, [5, 11]] ** 2, dim=1)

    def _reward_default_thigh_joint_pos(self):
        selected_columns = [2, 8]
        _yaw_roll = self.dof_pos[:, selected_columns]
        yaw_roll = torch.norm(_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll, 0, 50)
        return torch.exp(-yaw_roll / 0.1)

    def _reward_base_height(self):
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1
        ) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)
        c_update = (lin_mismatch + ang_mismatch) / 2.0
        return c_update

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    def _reward_track_vel_hard(self):
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1
        )
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        rew = self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_lin).sum(dim=1) / 2
        return rew

    def _reward_tracking_ang_vel(self):
        error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, 2]))
        rew = self._neg_sqrd_exp(error, a=self.cfg.rewards.tracking_sigma_ang)
        return rew

    def _reward_feet_clearance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        feet_z = self.rigid_state[:, self.feet_indices - 1, 2] - 0.0635
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_gait_phase()
        rew_pos = (
            torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        )
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(
            self.commands[:, 0]
        )
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.0
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_collision(self):
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_default_ankle_roll_pos(self):
        e_1 = get_euler_xyz_tensor(self.rigid_state[:, self.feet_indices[0], 3:7])
        e_2 = get_euler_xyz_tensor(self.rigid_state[:, self.feet_indices[1], 3:7])
        feet_eular_0 = torch.abs(e_1[:, 0])
        feet_eular_1 = torch.abs(e_2[:, 0])
        rew = torch.exp(-((feet_eular_0 + feet_eular_1) / 2) / 0.1)
        feet_eular_0 = torch.abs(e_1[:, 1])
        feet_eular_1 = torch.abs(e_2[:, 1])
        rew += torch.exp(-((feet_eular_0 + feet_eular_1) / 2) / 0.1)
        return rew / 2

    def _reward_termination(self):
        return -(self.reset_buf * ~self.time_out_buf).float()

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _neg_exp(self, x, a=1):
        return torch.exp(-(x / a) / a)

    def _neg_sqrd_exp(self, x, a=1):
        return torch.exp(-torch.square(x / a) / a)
