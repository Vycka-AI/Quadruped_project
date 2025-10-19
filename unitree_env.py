import gymnasium as gym
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

class UnitreeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # --- Store initial state for resets and pose penalty ---
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        
        self.n_substeps = 10
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        # --- Task & Command Parameters (from repo) ---
        self.target_velocity = np.zeros(3) # Will be randomized on reset
        self.target_height = 0.3 # Target base height from reward_cfg

        # --- Define Action and Observation Spaces ---
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.get_observation()),), dtype=np.float32
        )

    def get_observation(self):
        joint_pos = self.data.qpos[7:].copy()
        joint_vel = self.data.qvel[6:].copy()
        torso_quat = self.data.sensor('imu_quat').data.copy()
        base_lin_vel = self.data.qvel[:3].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        fl_force = self.data.sensor('FL_foot_force').data.copy()
        fr_force = self.data.sensor('FR_foot_force').data.copy()
        rl_force = self.data.sensor('RL_foot_force').data.copy()
        rr_force = self.data.sensor('RR_foot_force').data.copy()
            
        foot_forces = np.concatenate([fl_force, fr_force, rl_force, rr_force])

        # We add target height to the observation as per the repo's command structure
        commands = np.concatenate([self.target_velocity, [self.target_height]])
        
        observation = np.concatenate([
            joint_pos, joint_vel, base_lin_vel, base_ang_vel, torso_quat, foot_forces,
            commands, self.last_action
        ])
        return observation

    def step(self, action):
        true_control_range = self.model.actuator_ctrlrange
        low_bound, high_bound = true_control_range[:, 0], true_control_range[:, 1]
        rescaled_action = low_bound + (action + 1.0) * 0.5 * (high_bound - low_bound)
        self.data.ctrl[:] = np.clip(rescaled_action, low_bound, high_bound)
        
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        self.last_action = action.copy()

        observation = self.get_observation()
        reward = self.calculate_reward(action)
        terminated = self.is_terminated()
        
        if terminated:
            reward = -50.0 # Apply a large penalty for termination

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)

        # Command Randomization
        lin_vel_x = np.random.uniform(0.3, 0.7)
        ang_vel_z = np.random.uniform(-0.3, 0.3)
        self.target_velocity = np.array([lin_vel_x, 0.0, ang_vel_z])

        self.last_action.fill(0)
        return self.get_observation(), {}

    def calculate_reward(self, action):
        # --- State ---
        base_lin_vel = self.data.qvel[:3]
        base_ang_vel = self.data.qvel[3:6]
        joint_pos = self.data.qpos[7:]
        current_height = self.data.qpos[2]
        
        # --- 1. Velocity Tracking Rewards (from repo) ---
        tracking_sigma = 0.25 # A scaling factor they use
        lin_vel_error = np.sum(np.square(self.target_velocity[:2] - base_lin_vel[:2]))
        lin_vel_reward = 1.0 * np.exp(-lin_vel_error / tracking_sigma)
        
        ang_vel_error = np.square(self.target_velocity[2] - base_ang_vel[2])
        ang_vel_reward = 0.2 * np.exp(-ang_vel_error / tracking_sigma)

        # --- 2. Penalties (matching the repo's reward_scales) ---
        # Height Penalty
        height_error = np.square(current_height - self.target_height)
        height_penalty = -50.0 * height_error

        # Action Rate Penalty (Jerk)
        action_rate_penalty = -0.005 * np.sum(np.square(action - self.last_action))

        # Pose Similarity Penalty
        default_pose = self.initial_qpos[7:]
        pose_similarity_penalty = -0.1 * np.sum(np.square(joint_pos - default_pose))
        
        # Vertical Velocity Penalty
        vertical_velocity_penalty = -1.0 * np.square(base_lin_vel[2])

        # Roll and Pitch Stabilization Penalty
        torso_quat_mujoco = self.data.sensor('imu_quat').data
        rot = Rotation.from_quat([torso_quat_mujoco[1], torso_quat_mujoco[2], torso_quat_mujoco[3], torso_quat_mujoco[0]])
        roll, pitch = rot.as_euler('xyz', degrees=False)[0:2]
        roll_pitch_penalty = -1.0 * (np.square(roll) + np.square(pitch)) # Assuming a weight of -1.0

        # --- Total Reward ---
        total_reward = (
            lin_vel_reward +
            ang_vel_reward +
            height_penalty +
            action_rate_penalty +
            pose_similarity_penalty +
            vertical_velocity_penalty +
            roll_pitch_penalty
        )
        return total_reward

    def is_terminated(self):
        torso_height = self.data.qpos[2]
        torso_quat_mujoco = self.data.sensor('imu_quat').data
        rot = Rotation.from_quat([torso_quat_mujoco[1], torso_quat_mujoco[2], torso_quat_mujoco[3], torso_quat_mujoco[0]])
        roll, pitch = rot.as_euler('xyz', degrees=False)[0:2]
        
        # Using a slightly looser threshold than the repo for initial training
        is_flipped = (abs(roll) > np.pi / 2.5) or (abs(pitch) > np.pi / 2.5)
        is_fallen = torso_height < 0.18
        
        return is_fallen or is_flipped

    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

            #use CPG