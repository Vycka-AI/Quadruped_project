import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer # Import the viewer
from scipy.spatial.transform import Rotation # 1. Import the Rotation tool

class UnitreeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path='path/to/your/unitree_model.xml', render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # --- BUG FIX #1: CORRECT INITIALIZATION ORDER ---
        # 1. Define action space and last_action variable FIRST
        action_shape = (self.model.nu,)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=action_shape, dtype=np.float32
        )
        self.last_action = np.zeros(action_shape, dtype=np.float32)

        # 2. Now, with all components ready, define the observation space
        observation_size = len(self.get_observation())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32
        )

    # --- PERFORMANCE IMPROVEMENT: OPTIMIZED get_observation ---
    def get_observation(self):
        """ Gathers data efficiently using direct array access. """
        # Joint positions (qpos[7:] starts after the 7D root pose)
        joint_pos = self.data.qpos[7:].copy()
        
        # Joint velocities (qvel[6:] starts after the 6D root velocity)
        joint_vel = self.data.qvel[6:].copy()
        
        # Actuator forces/torques
        joint_torques = self.data.actuator_force.copy()

        # IMU and Frame data (using sensors is fine here as it's not in a tight loop)
        imu_quat = self.data.sensor('imu_quat').data.copy()
        imu_gyro = self.data.sensor('imu_gyro').data.copy()
        imu_acc = self.data.sensor('imu_acc').data.copy()
        frame_pos = self.data.sensor('frame_pos').data.copy()
        frame_vel = self.data.sensor('frame_vel').data.copy()
        
        observation = np.concatenate([
            joint_pos, joint_vel, joint_torques,
            imu_quat, imu_gyro, imu_acc, frame_pos, frame_vel,
            self.last_action
        ])
        return observation

    def step(self, action):
        # Action rescaling is correct
        true_control_range = self.model.actuator_ctrlrange
        low_bound, high_bound = true_control_range[:, 0], true_control_range[:, 1]
        rescaled_action = low_bound + (action + 1.0) * 0.5 * (high_bound - low_bound)
        self.data.ctrl[:] = np.clip(rescaled_action, low_bound, high_bound)
        
        # Store the normalized action
        self.last_action = action.copy()

        mujoco.mj_step(self.model, self.data, nstep=5)
        
        if self.render_mode == "human":
            self.render()

        observation = self.get_observation()
        reward = self.calculate_reward(action)
        terminated = self.is_terminated()
        
        # Suggestion: Add a large penalty for termination
        if terminated:
            reward -= 2000.0 # Example penalty

        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility
        mujoco.mj_resetData(self.model, self.data)
        
        # --- BUG FIX #2: RESET last_action ---
        self.last_action.fill(0)
        
        return self.get_observation(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def calculate_reward(self, current_action):
        # Your reward function is very well designed! No changes needed here.
        # It correctly balances multiple objectives.
        target_velocity = 1.0
        forward_velocity = self.data.qvel[0]
        velocity_error = np.abs(target_velocity - forward_velocity)
        reward_forward = 4.0 * np.exp(-5.0 * velocity_error)

        alive_bonus = 50
        torque_penalty = 0.0001 * np.sum(np.square(self.data.ctrl))
        angular_velocity_penalty = 0.01 * np.sum(np.square(self.data.qvel[3:6]))

        # NEW: Penalty for standing still (inaction)
        stillness_threshold = 0.05  # m/s
        if abs(forward_velocity) < stillness_threshold:
            stillness_penalty = 0.5
        else:
            stillness_penalty = 0.0

        jerk_penalty = 0.5 * np.sum(np.square(current_action - self.last_action))


        # 3. NEW: Orientation Penalty (penalizes tilting) 🧭
        # Get the torso's orientation as a rotation matrix from the IMU quaternion
        torso_quat_mujoco = self.data.sensor('imu_quat').data # This is in [w, x, y, z] format
    
        # Scipy expects quaternions in [x, y, z, w] format, so we reorder it
        torso_quat_scipy = [torso_quat_mujoco[1], torso_quat_mujoco[2], torso_quat_mujoco[3], torso_quat_mujoco[0]]
        
        # Create a Rotation object from the quaternion
        rot = Rotation.from_quat(torso_quat_scipy)
        
        # Convert to Euler angles [roll, pitch, yaw] in radians
        euler_angles = rot.as_euler('xyz', degrees=False)
        roll, pitch = euler_angles[0], euler_angles[1]
        orientation_penalty = (roll**2 + pitch**2)

        # This shaped penalty is a good idea
        if self.data.qpos[2] < 0.2:
            down_penalty = (0.2 - self.data.qpos[2]) * 10
        else:
            down_penalty = 0.0

        total_reward = reward_forward + alive_bonus - torque_penalty - angular_velocity_penalty - down_penalty - orientation_penalty - stillness_penalty - jerk_penalty
        return total_reward

    def is_terminated(self):
        # Your termination condition is fine. The height is a tunable parameter.
        return self.data.qpos[2] < 0.1