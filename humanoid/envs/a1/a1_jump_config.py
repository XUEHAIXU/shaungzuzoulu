from humanoid.envs.a1.a1_config import A1Cfg, A1CfgPPO


class A1JumpCfg(A1Cfg):
    """Bunny-hop via foot-sync constraint on the walking framework (v12).

    Inherits walking config in full (commands, gait timing, all rewards) and
    overrides only what's needed to convert walking -> bunny-hop:
      - sync sin gait reference + sync stance_mask (in env)
      - strong feet_sync_z + feet_sync_contact penalties
      - base_height override to absolute root z (parent class NaNs when both
        feet in swing under sync)
      - disable shapers that conflict with sync (default_thigh_joint_pos uses
        an exp on hip yaw which is fine; feet_distance/knee_distance still
        useful to prevent leg crossing)

    Walking commands pull forward; sync forbids alternation; the only feasible
    translation is bunny-hop forward.
    """

    class env(A1Cfg.env):
        episode_length_s = 24

    class commands(A1Cfg.commands):
        # Inherit walking ranges (lin_vel_x [-0.3, 0.6], heading_command=True)
        # so policy gets the same forward-walking signal walking trains under.
        pass

    class domain_rand(A1Cfg.domain_rand):
        # Inherit walking domain_rand (friction [0.1,2.0], push 0.2/0.4).
        pass

    class rewards(A1Cfg.rewards):
        # base_height_target, cycle_time (0.64s), target_feet_height (0.15),
        # min/max_dist, target_joint_pos_scale all inherited from A1Cfg.

        class scales(A1Cfg.rewards.scales):
            # ===== feet sync hard constraints (the v12 idea) =====
            # |Δz_feet|^2 -- typical walking |Δz| peaks ~0.15m -> 0.0225. Scale
            # -50 -> -0.0225*50*0.02 = -0.0225/step at peak, so a fully
            # walking-style episode pays -0.0225 * 0.5 * 1200 ~= -13.5 over
            # the swing phase. Strong: roughly 15% of walking's positive
            # reward total.
            feet_sync_z = -50.0
            # XOR contact: 1 when feet in different contact states. Walking
            # has feet alternating ~70% of the cycle -> per-step 0.7 -> with
            # scale -2.0: -2.0*0.02*0.7*1200 = -33.6/ep when fully walking.
            # This is the dominant penalty -- it's the constraint that
            # forbids walking-style locomotion outright.
            feet_sync_contact = -2.0

            # ===== walking shapers that need adjustment for sync mode =====
            # Walking's _reward_feet_contact_number uses the (now sync)
            # stance_mask -- both feet must match the same in/out label.
            # Keep at walking weight; it now rewards both-down and both-up
            # rather than alternation.
            feet_contact_number = 1.2
            # _reward_feet_air_time inherited; cycle_time 0.64s gives swing
            # ~0.32s -> reward fires when both feet just landed after >=0.5s
            # combined airborne. Wait, with sync both feet leave/land
            # together, individual air_time is the same as flight duration.
            # Reward already clamps at 0.5s. Keep walking weight.
            feet_air_time = 1.0
            # _reward_feet_clearance rewards feet at target height during
            # swing. Swing is now both-feet-airborne -- target_feet_height
            # 0.15m is the apex.
            feet_clearance = 1.0
            # _reward_joint_pos rewards matching ref_dof_pos. With sync ref
            # both legs simultaneously crouch then extend.
            joint_pos = 1.0

            # ===== walking shapers that conflict with bunny-hop (disable) =====
            # foot_slip penalises in-stance foot motion. In bunny-hop landings
            # are hard and brief slip is unavoidable; keep but weaken.
            foot_slip = -0.05
            # feet_distance and knee_distance keep stance wide. Bunny-hop
            # tolerates narrower stance (legs together is a valid hop posture).
            # Reduce so they don't fight tight-leg jumps.
            feet_distance = 0.2
            knee_distance = 0.2
            # low_speed and track_vel_hard reward continuous forward motion at
            # commanded speed. With bunny-hop momentum is intermittent -- vel
            # spikes at takeoff and decays during flight. Disable to avoid
            # punishing the natural hop velocity profile.
            low_speed = 0.0
            track_vel_hard = 0.0
            # vel_mismatch_exp rewards low base z velocity and roll/pitch.
            # Bunny-hop necessarily has nonzero base vz; disable.
            vel_mismatch_exp = 0.0

            # ===== overrides on walking penalties =====
            # base_height (overridden in env to absolute root z): keep walking
            # weight -2.0. With base_height_target 0.5 and ballistic apex
            # ~0.05m the per-step penalty is small.
            base_height = -2.0
            # Inherit walking's orientation -10 (quadratic projected_gravity_xy)
            # and ang_vel_xy -1 (quadratic). These keep base flat during hop.
            orientation = -10.0
            ang_vel_xy = -1.0
            # default_hip_roll_joint_pos -10 strong (kept).
            default_hip_roll_joint_pos = -10.0
            # default_thigh_joint_pos / default_ankle_roll_pos / ankle_roll_flat
            # all inherited.

            # ===== keep symmetry (v9 addition) =====
            symmetry = -0.5

            # ===== disable v11 3-segment design entirely =====
            stance_contact = 0.0
            flight_air = 0.0
            feet_height_jump = 0.0
            flight_grounded = 0.0
            lin_vel_z = 0.0
            default_pos = 0.0
            jump = 0.0

            # termination inherited (=1.0); _reward_termination returns -1
            # on early death, so 1.0 * 0.02 * (-1) = -0.02 per termination.
            # Walking-level signal.
            termination = 1.0


class A1JumpCfgPPO(A1CfgPPO):
    seed = 5

    class policy(A1CfgPPO.policy):
        # Walking trains stably from default 1.0 noise; sync gait is a
        # walking-shaped task so the same starting noise should work.
        init_noise_std = 1.0

    class algorithm(A1CfgPPO.algorithm):
        # Inherit walking's 0.01 -- walking provides advantage signal
        # throughout training (gait refinement) so noise std stays bounded.
        # Sync framework should behave similarly. v9-v11 problems with
        # entropy_coef 0.01 came from the 3-segment design saturating
        # advantage; doesn't apply here.
        entropy_coef = 0.01

    class runner(A1CfgPPO.runner):
        experiment_name = "A1_jump_ppo"
        run_name = "v12_sync"
        max_iterations = 30000
        save_interval = 200
