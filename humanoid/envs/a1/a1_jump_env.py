import torch

from humanoid.envs.a1.a1_env import A1FreeEnv
from humanoid.envs.a1.a1_jump_config import A1JumpCfg


class A1JumpEnv(A1FreeEnv):
    """Bunny-hop emerging from walking framework + feet-sync hard constraints.

    Strategy: keep all walking-task rewards (proven to train) and
      (1) replace the anti-phase sin gait reference with a SYNC reference
          (both legs crouch and extend together)
      (2) replace anti-phase stance_mask with a SYNC stance_mask (both feet
          in stance or in swing simultaneously)
      (3) add feet_sync_z and feet_sync_contact penalties making sync a hard
          constraint at the kinematic level.

    With command tracking pulling forward and the feet forced into lockstep,
    the only way to translate is to push off and land with both feet at once
    -- a bunny-hop. The cycle_time, foot_air_time, feet_clearance machinery
    inherited from walking handles the timing.

    Note: _reward_base_height is overridden because the parent class form
    divides by sum(stance_mask) which is 0 when both feet are in swing
    (legal under sync) -> NaN. We use absolute root z vs target instead.
    """

    cfg: A1JumpCfg

    def _get_gait_phase(self):
        # SYNC stance mask: both feet in the same phase. Walking has
        # stance_mask[:,0] = (sin >= 0), stance_mask[:,1] = (sin < 0) -- one
        # leg in stance while the other swings. Here we mirror both columns
        # to the same value so the policy must put both feet down or lift
        # both up simultaneously.
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0])
        in_stance = (sin_pos < 0).float()
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = in_stance
        stance_mask[:, 1] = in_stance
        # Near zero crossings count as stance for both feet (transition
        # window where contact ambiguity is OK).
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def compute_ref_state(self):
        # SYNC sin gait reference: both legs follow the same crouch-extend
        # curve. Walking uses sin_pos_l (negative-half clipped) for R and
        # sin_pos_r (positive-half clipped) for L, producing antiphase
        # stride. Here both legs use the same negative-half clip so they
        # crouch together (vertical pump) instead of striding alternately.
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase + self.random_half_phase[0])
        sin_sync = sin_pos.clone()
        sin_sync[sin_sync > 0] = 0  # crouch on the negative half only
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        # Right leg (indices 0,3,4 = hip pitch, knee, ankle pitch)
        self.ref_dof_pos[:, 0] = sin_sync * scale_1
        self.ref_dof_pos[:, 3] = -sin_sync * scale_2
        self.ref_dof_pos[:, 4] = sin_sync * scale_1
        # Left leg (6,9,10) -- SAME sign as right (sync, not antiphase)
        self.ref_dof_pos[:, 6] = sin_sync * scale_1
        self.ref_dof_pos[:, 9] = -sin_sync * scale_2
        self.ref_dof_pos[:, 10] = sin_sync * scale_1
        # Stand straight at zero crossings.
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_action = 2 * self.ref_dof_pos

    # ---------- foot-sync hard constraints ----------

    def _reward_feet_sync_z(self):
        # Quadratic penalty on z difference between feet (paired with negative
        # scale). Walking-style alternating gait produces |Δz| up to ~0.15m
        # -> 0.0225; sync mode wants this near 0. Strong penalty here is the
        # "hard constraint" that converts walking framework into bunny-hop.
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        return torch.square(feet_z[:, 0] - feet_z[:, 1])

    def _reward_feet_sync_contact(self):
        # XOR penalty on contact states (paired with negative scale). Returns
        # 1.0 when exactly one foot is in contact (alternation = walking) and
        # 0.0 when both are in contact (stance) or both off (flight). Forces
        # both feet to leave/land together.
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        xor = (contact[:, 0] ^ contact[:, 1]).float()
        return xor

    # ---------- override base_height (parent NaNs when both feet in swing) ----------

    def _reward_base_height(self):
        # Quadratic deviation from base_height_target using absolute root z.
        # Plane terrain only -- ground is at z=0 throughout, so this is the
        # straight error. Used with negative scale.
        target = self.cfg.rewards.base_height_target
        return torch.square(self.root_states[:, 2] - target)

    # ---------- left/right symmetry (kept) ----------

    def _reward_symmetry(self):
        # Mirror residual across the sagittal plane.
        # pitch joints (axis y, indices 0,3,4): R == L
        # roll/yaw joints (axis x or z, indices 1,2,5): R == -L
        R = self.dof_pos[:, 0:6]
        L = self.dof_pos[:, 6:12]
        pitch_mask = torch.tensor(
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            device=self.device,
            dtype=R.dtype,
        )
        diff = pitch_mask * (R - L) + (1.0 - pitch_mask) * (R + L)
        return torch.sum(diff * diff, dim=1)
