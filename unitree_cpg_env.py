# --- START OF FILE unitree_cpg_env.py ---

import gymnasium as gym
import numpy as np
import mujoco
from cpg_controller import CPGController # Import our new CPG class

class UnitreeCPGEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        # --- Standard MuJoCo and rendering setup ---
        self.render_mode = render_mode
        self.viewer = None
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # --- Store initial state ---
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        
        # --- CPG Integration ---
        # The simulation timestep is crucial for the CPG's phase update
        self.n_substeps = 10
        cpg_dt = self.model.opt.timestep * self.n_substeps
        self.cpg = CPGController(dt=cpg_dt)

        
        self.last_cpg_action = np.zeros(4, dtype=np.float32)

        # --- PD Controller Gains ---
        # These gains determine how stiffly the joints follow the CPG targets
        self.kp = 100.0  # Proportional gain (stiffness)
        self.kd = 5.0   # Derivative gain (damping)

        # --- Task & Command Parameters ---
        self.target_velocity = np.zeros(3) # [lin_vel_x, 0.0, ang_vel_z]

        self.base_body_height = 0.3 # Target base height from your original env
        # --- NEW: Define CPG Action and Observation Spaces ---
        # The agent now controls 4 CPG parameters instead of 12 joints
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.get_observation()),), dtype=np.float32
        )

    def get_observation(self):
        # We can use the same observation as before, but it's helpful
        # to add the CPG phase information so the agent knows the state of the gait.
        joint_pos = self.data.qpos[7:].copy()
        joint_vel = self.data.qvel[6:].copy()
        base_lin_vel = self.data.qvel[:3].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        torso_quat = self.data.sensor('imu_quat').data.copy()
        
        # Combine CPG state with robot state
        cpg_phases = np.sin(self.cpg.phases) # Use sin of phases for continuous representation
        commands = np.concatenate([self.target_velocity[:], [self.base_body_height]])
        
        observation = np.concatenate([
            joint_pos, joint_vel, base_lin_vel, base_ang_vel, torso_quat,
            commands, self.last_cpg_action, cpg_phases
        ])
        return observation

    def step(self, action):
        # --- CPG LOGIC ---
        # 1. Get target joint positions from the CPG
        target_qpos = self.cpg.update(action)
        
        # --- PD CONTROL LOGIC ---
        # 2. Calculate the torque required to reach the target positions
        current_qpos = self.data.qpos[7:]
        current_qvel = self.data.qvel[6:]
        
        # Torque = Kp * (target_pos - current_pos) - Kd * current_vel
        torque = self.kp * (target_qpos - current_qpos) - self.kd * current_qvel
        
        # 3. Apply the calculated torque
        self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        
        # 4. Step the simulation
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        self.last_cpg_action = action.copy()

        # --- Standard RL loop ---
        observation = self.get_observation()
        # The same reward function should work well! The goal hasn't changed.
        reward = self.calculate_reward(action)
        terminated = self.is_terminated()
        
        if terminated:
            reward = -50.0

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)

        # Reset the CPG controller
        self.cpg.reset()

        # Randomize commands
        lin_vel_x = np.random.uniform(0.3, 0.7)
        ang_vel_z = np.random.uniform(-0.3, 0.3)
        self.target_velocity = np.array([lin_vel_x, 0.0, ang_vel_z])

        self.last_cpg_action.fill(0)
        return self.get_observation(), {}

    # --- REWARD AND TERMINATION (Can be copied from unitree_env.py) ---
    def calculate_reward(self, action):
        # This reward function is already excellent for this task.
        # It correctly incentivizes matching a target velocity while penalizing instability.
        base_lin_vel = self.data.qvel[:3]
        base_ang_vel = self.data.qvel[3:6]
        
        tracking_sigma = 0.25
        lin_vel_error = np.sum(np.square(self.target_velocity[:2] - base_lin_vel[:2]))
        lin_vel_reward = 1.0 * np.exp(-lin_vel_error / tracking_sigma)
        
        ang_vel_error = np.square(self.target_velocity[2] - base_ang_vel[2])
        ang_vel_reward = 0.5 * np.exp(-ang_vel_error / tracking_sigma) # Slightly increased turning reward

        # Penalize jerky high-level commands
        action_rate_penalty = -0.01 * np.sum(np.square(action - self.last_cpg_action))
        
        return lin_vel_reward + ang_vel_reward + action_rate_penalty

    def is_terminated(self):
        torso_height = self.data.qpos[2]
        torso_quat_mujoco = self.data.sensor('imu_quat').data
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([torso_quat_mujoco[1], torso_quat_mujoco[2], torso_quat_mujoco[3], torso_quat_mujoco[0]])
        roll, pitch = rot.as_euler('xyz', degrees=False)[0:2]
        
        is_flipped = (abs(roll) > np.pi / 2.5) or (abs(pitch) > np.pi / 2.5)
        is_fallen = torso_height < 0.18
        
        return is_fallen or is_flipped

    # --- Rendering and Closing (Unchanged) ---
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