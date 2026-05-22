from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1Cfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 20
        use_ref_actions = False

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/A1-legs_V1/A1-legs_V1.urdf"
        name = "A1"
        foot_name = "6"
        knee_name = "4"
        terminate_after_contacts_on = ["base"]
        penalize_contacts_on = ["base"]
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.57]
        default_joint_angles = {
            "joint_R1": -0.1,
            "joint_R2": 0.0,
            "joint_R3": 0.0,
            "joint_R4": 0.3,
            "joint_R5": -0.2,
            "joint_R6": 0.0,
            "joint_L1": -0.1,
            "joint_L2": 0.0,
            "joint_L3": 0.0,
            "joint_L4": 0.3,
            "joint_L5": -0.2,
            "joint_L6": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD gains use substring matching: key "1" matches joint_R1 and joint_L1
        stiffness = {
            "1": 30.0,   # hip pitch
            "2": 30.0,   # hip roll
            "3": 30.0,   # hip yaw
            "4": 30.0,   # knee
            "5": 20.0,   # ankle pitch (lower effort motor on A1)
            "6": 20.0,   # ankle roll
        }
        damping = {
            "1": 1.0,
            "2": 1.0,
            "3": 1.0,
            "4": 1.0,
            "5": 0.5,
            "6": 0.5,
        }
        action_scale = 0.25
        decimation = 20

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 30
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 4.0
        heading_command = True

        class ranges:
            lin_vel_x = [-0.3, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.5
        # default standing foot/knee y-separation is 0.322 m. Put the penalty
        # wall at 0.25 (vs 0.20) so the policy gets gradient before legs collapse
        # into a V — see _reward_feet_distance / _reward_knee_distance.
        min_dist = 0.25
        max_dist = 0.40
        target_joint_pos_scale = 0.25#0.17
        target_feet_height = 0.15
        cycle_time = 0.64
        only_positive_rewards = False
        tracking_sigma_ang = 0.25
        tracking_sigma_lin = 0.25
        max_contact_force = 200

        class scales:
            joint_pos = 1.0
            feet_clearance = 1.0
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.0
            foot_slip = -0.1
            feet_distance = 0.6          # was 0.2 -- stronger shaping for wider stance
            knee_distance = 0.4          # was 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            vel_mismatch_exp = 1.0       # was 0.5 -- reward low roll/pitch rates more
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_hip_roll_joint_pos = -10.0  # pure squared penalty on R2/L2 hip roll
            default_thigh_joint_pos = 0.5
            default_ankle_roll_pos = 0.5     # was 0.3 -- stabilize ankle_roll against lateral rocking
            ankle_roll_flat = -0.5           # penalise ankle roll joint to block inversion/eversion compensation
            orientation = -10.0              # was -5.0 -- stronger flat-base penalty (roll/pitch)
            ang_vel_xy = -1.0  #-0.1              # direct quadratic penalty on roll/pitch angular velocity
            base_height = -2.0
            base_acc = 0.3                   # was 0.2 -- smoother base motion
            # energy / regularization
            action_smoothness = -0.005       # was -0.003 -- smoother actions, less rocking
            torques = -1e-4
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            dof_pos_limits = -2.0
            collision = -1.0

            termination = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0


class A1CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 100000
        save_interval = 100
        experiment_name = "A1_ppo"
        run_name = "v1"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
