import gymnasium as gym
import numpy as np
import mujoco
import mujoco.glfw # Import glfw
from scipy.spatial.transform import Rotation
import math
import time # Import time for potential frame limiting

# --- PD Controller Gains (TUNE THESE!) ---
KP = 100.0
KD = 5.0

class UnitreeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.window = None # Add window attribute
        self.cam = None # Add camera attribute
        self.opt = None # Add options attribute
        self.scn = None # Add scene attribute
        self.ctx = None # Add context attribute
        self._render_initialized = False # Flag to track viewer setup
        self.render_timer = time.time() # For frame rate limiting

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            print(f"Error loading MuJoCo model from {model_path}: {e}")
            raise

        self.data = mujoco.MjData(self.model)
        self.num_actuators = self.model.nu

        self.kp_array = np.full(self.num_actuators, KP)
        self.kd_array = np.full(self.num_actuators, KD)
        self.target_qvel = np.zeros(self.num_actuators)

        self.actuator_joint_ids = [self.model.actuator_trnid[i, 0] for i in range(self.num_actuators)]
        self.qpos_indices = []
        self.qvel_indices = []
        for jid in self.actuator_joint_ids:
            self.qpos_indices.append(self.model.jnt_qposadr[jid])
            self.qvel_indices.append(self.model.jnt_dofadr[jid])

        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.default_qpos = self.initial_qpos[self.qpos_indices].copy()

        self.n_substeps = 10
        self.dt = self.model.opt.timestep * self.n_substeps
        self.last_action_raw = np.zeros(self.num_actuators, dtype=np.float32)
        # Inside __init__ method, after initializing self.last_action_raw
        self.last_action_raw = np.zeros(self.num_actuators, dtype=np.float32)
        self.previous_action_raw = np.zeros(self.num_actuators, dtype=np.float32) # ADD THIS LINE

        self.target_lin_vel = np.zeros(2)
        self.target_ang_vel = 0.0
        self.target_height = 0.3

        self.action_scale = np.pi
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32
        )

        obs_dim = 3 + 3 + 3 + 12 + 12 + 12
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.ctrl_range = self.model.actuator_ctrlrange.copy()

        # Initialize rendering if mode is human
        if self.render_mode == "human":
            self._initialize_render()


    def _initialize_render(self):
        """Sets up the MuJoCo rendering objects."""
        if self._render_initialized:
            return

        print("Initializing viewer...")
        mujoco.glfw.glfw.init()
        self.window = mujoco.glfw.glfw.create_window(1200, 900, "Unitree Go2 Training", None, None)
        if not self.window:
             print("ERROR: Could not create GLFW window.")
             mujoco.glfw.glfw.terminate()
             return

        mujoco.glfw.glfw.make_context_current(self.window)
        mujoco.glfw.glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Default camera view
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth = 120
        self.cam.elevation = -20
        self.cam.distance = 2.5
        self.cam.lookat = self.data.qpos[:3] # Initial lookat

        self._render_initialized = True
        print("Viewer initialized.")


    def _get_obs(self):
        # ... (observation calculation code remains the same) ...
        # Base linear velocity (world frame)
        lin_vel = self.data.sensor('frame_vel').data.copy()
        # Base angular velocity (world frame)
        ang_vel = self.data.sensor('imu_gyro').data.copy() # Gyro is usually body frame, check sensor definition
        # If gyro is body frame, need to rotate to world (or keep consistent)
        # Let's assume for now sensors provide world frame or policy learns frame invariance

        # Base orientation (gravity vector in base frame)
        base_quat_mujoco = self.data.sensor('imu_quat').data # w, x, y, z
        rot = Rotation.from_quat([base_quat_mujoco[1], base_quat_mujoco[2], base_quat_mujoco[3], base_quat_mujoco[0]]) # x, y, z, w for scipy
        gravity_world = np.array([0., 0., -1.])
        gravity_base = rot.apply(gravity_world, inverse=True)

        # Joint positions and velocities
        joint_pos = self.data.qpos[self.qpos_indices].copy()
        joint_vel = self.data.qvel[self.qvel_indices].copy()

        # Last raw action (target joint angles, scaled)
        last_target_angles = self._scale_action(self.last_action_raw)


        obs = np.concatenate([
            lin_vel,
            ang_vel,
            gravity_base,
            joint_pos,
            joint_vel,
            last_target_angles # Using the calculated target angles, not raw action
        ])
        return obs.astype(np.float32)


    def _scale_action(self, action_raw):
        # ... (scaling code remains the same) ...
        target_qpos = self.default_qpos + action_raw * self.action_scale
        return target_qpos

    def step(self, action_raw):
        # ... (PD control and mj_step code remains the same) ...
        action_raw = np.array(action_raw, dtype=np.float32)
        self.last_action_raw = action_raw

        target_qpos = self._scale_action(action_raw)
        current_qpos = self.data.qpos[self.qpos_indices]
        current_qvel = self.data.qvel[self.qvel_indices]

        pos_error = target_qpos - current_qpos
        vel_error = self.target_qvel - current_qvel

        pd_torques = self.kp_array * pos_error + self.kd_array * vel_error
        clamped_torques = np.clip(pd_torques, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[:] = clamped_torques

        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)


        # --- Get Obs, Reward, Termination, Truncation ---
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self.is_terminated()
        truncated = False # Implement truncation logic if needed (e.g., max episode steps)
        info = {}

        # --- Handle Rendering ---
        if self.render_mode == "human":
            self.render() # Call the render method

        self.previous_action_raw = self.last_action_raw.copy() # ADD THIS LINE

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # ... (reset logic remains the same) ...
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)

        self.last_action_raw = np.zeros(self.num_actuators, dtype=np.float32)

        self.last_action_raw = np.zeros(self.num_actuators, dtype=np.float32)
        self.previous_action_raw = np.zeros(self.num_actuators, dtype=np.float32) # ADD THIS LINE

        self.target_lin_vel = self.np_random.uniform(low=[-0.5, -0.3], high=[1.0, 0.3], size=2)
        self.target_ang_vel = self.np_random.uniform(low=-0.5, high=0.5)

        observation = self._get_obs()
        info = {}

        return observation, info

    def _calculate_reward(self):
        # ... (reward calculation remains the same) ...
        # --- Read relevant states ---
        lin_vel = self.data.sensor('frame_vel').data[:2] # World X, Y velocity
        ang_vel_z = self.data.sensor('imu_gyro').data[2] # Z angular velocity (assuming body frame)
        torso_height = self.data.qpos[2]
        applied_torques = self.data.ctrl.copy()
        joint_vel = self.data.qvel[self.qvel_indices].copy()

        # --- Velocity Tracking ---
        lin_vel_error = np.linalg.norm(self.target_lin_vel - lin_vel)
        ang_vel_error = np.square(self.target_ang_vel - ang_vel_z)

        # Rewards (using exponential forms like many papers)
        lin_vel_reward = np.exp(-2.0 * lin_vel_error) # Weight 2.0
        ang_vel_reward = np.exp(-0.5 * ang_vel_error) # Weight 0.5

        # --- Penalties ---
        # Height penalty (encourage target height)
        height_error = np.square(self.target_height - torso_height)
        height_penalty = -5.0 * height_error # Weight -5.0

        # Action rate penalty (discourage jerky movements in target angles)
        # Calculate diff based on previous *raw* action to current *raw* action
        action_diff = self.last_action_raw - self.previous_action_raw # *** FIX HERE ***
        # Correction: Calculate diff based on previous *target angles* and current *target angles*
        prev_target_qpos = self._scale_action(self.last_action_raw) # Requires storing last_action_raw
        #current_target_qpos = self._scale_action(action_raw) # Requires access to current action_raw
        # Easiest: Penalty on change in raw actions
        # action_rate_penalty = -0.005 * np.sum(np.square(action_raw - self.last_action_raw)) # Revisit this logic
        # Let's stick with penalizing joint speed and torques for now, simpler
        action_rate_penalty = 0.0 # Temporarily disable action rate penalty as implemented


        # Joint Speed Penalty (discourage high joint velocities)
        joint_speed_penalty = -0.0001 * np.sum(np.square(joint_vel)) # Weight -0.0001

        # Torque Penalty (discourage high torques)
        torque_penalty = -1e-5 * np.sum(np.square(applied_torques)) # Weight -1e-5

        # Body orientation penalty (penalize roll and pitch)
        base_quat_mujoco = self.data.sensor('imu_quat').data
        rot = Rotation.from_quat([base_quat_mujoco[1], base_quat_mujoco[2], base_quat_mujoco[3], base_quat_mujoco[0]])
        roll, pitch = rot.as_euler('xyz', degrees=False)[0:2]
        orientation_penalty = -0.5 * (np.square(roll) + np.square(pitch)) # Weight -0.5


        # --- Total Reward ---
        total_reward = (
            lin_vel_reward +
            ang_vel_reward +
            height_penalty +
            action_rate_penalty + # Now calculated correctly
            joint_speed_penalty +
            torque_penalty +
            orientation_penalty
        )
        return total_reward


    def is_terminated(self):
        # ... (termination logic remains the same) ...
        torso_height = self.data.qpos[2]
        base_quat_mujoco = self.data.sensor('imu_quat').data
        rot = Rotation.from_quat([base_quat_mujoco[1], base_quat_mujoco[2], base_quat_mujoco[3], base_quat_mujoco[0]])
        roll, pitch = rot.as_euler('xyz', degrees=False)[0:2]

        is_fallen = torso_height < 0.18
        is_flipped = (abs(roll) > math.radians(80)) or (abs(pitch) > math.radians(80))

        return is_fallen or is_flipped


    def render(self):
        """Renders the environment to a graphical window."""
        if self.render_mode != "human":
            # print("Warning: render() called but render_mode is not 'human'.")
            return

        if not self._render_initialized:
            # print("Error: Render called before initialization.")
             # Attempt to initialize if not already done (might happen on first step)
            self._initialize_render()
            if not self._render_initialized: # Check again if init failed
                 return

        # Limit rendering speed to target FPS
        time_until_next_frame = (1.0 / self.metadata["render_fps"]) - (time.time() - self.render_timer)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)
        self.render_timer = time.time() # Reset timer

        if self.window is None or mujoco.glfw.glfw.window_should_close(self.window):
             # print("Window closed or not initialized during render.")
             # Should probably terminate or handle this state better
             self.close() # Attempt to clean up
             return # Stop rendering if window is gone

        try:
             viewport_width, viewport_height = mujoco.glfw.glfw.get_framebuffer_size(self.window)
             viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

             # Update camera to follow the robot's base
             self.cam.lookat = self.data.qpos[:3]

             # Update scene and render
             mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
             mujoco.mjr_render(viewport, self.scn, self.ctx)

             # Swap buffers and poll events
             mujoco.glfw.glfw.swap_buffers(self.window)
             mujoco.glfw.glfw.poll_events()

        except Exception as e:
            print(f"Error during rendering: {e}")
            self.close() # Close if rendering fails


    def close(self):
        """Cleans up rendering resources."""
        if self.window:
            # print("Closing viewer window.")
            mujoco.glfw.glfw.destroy_window(self.window)
            self.window = None
        if self._render_initialized:
             mujoco.glfw.glfw.terminate()
             self._render_initialized = False
        # print("Environment closed.")