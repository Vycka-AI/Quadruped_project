import gymnasium as gym
import numpy as np
import mujoco

class Go2SuspendedEnv(gym.Env):
    """
    A Gymnasium environment for the Unitree Go2 robot suspended in the air.
    The robot's base is fixed, and the agent controls the 12 leg joints.
    This version uses PD position control, expecting actions as offsets
    from a default pose, just like in unitree-rl.
    """
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Get the ID of the "home" keyframe from the XML (optional, not used in reset)
        self.key_home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')

        # --- ADDED: Define default pose from GO2RoughCfg ---
        # NOTE: The order (FL, FR, RL, RR) must match your MuJoCo XML actuator order.
        # This order is based on the standard unitree_ros XMLs.
        self.default_dof_pos = np.array([
            0.1,  0.8, -1.5,  # FL_hip_joint, FL_thigh_joint, FL_calf_joint
           -0.1,  0.8, -1.5,  # FR_hip_joint, FR_thigh_joint, FR_calf_joint
            0.1,  1.0, -1.5,  # RL_hip_joint, RL_thigh_joint, RL_calf_joint
           -0.1,  1.0, -1.5   # RR_hip_joint, RR_thigh_joint, RR_calf_joint
        ])

        # --- ADDED: Control parameters from unitree-rl ---
        self.action_scale = 0.25
        # Assuming stiffness=20, damping=0.5 for all joints (like your previous code)
        self.p_gains = np.full(self.model.nu, 20.0)
        self.d_gains = np.full(self.model.nu, 0.5)

        # --- Define Action and Observation Spaces ---
        # Action space: 12 actuators for the 12 joints (from go2.xml)
        # Action is an offset, so [-1, 1] is appropriate.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation space: 12 joint positions + 12 joint velocities
        # self.model.nu is 12 (number of actuators/DoFs)
        # Total obs size = 12 (pos) + 12 (vel) = 24
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.model.nu,), dtype=np.float32
        )

    def _get_obs(self):
        """
        Returns the current observation, which is the joint positions and velocities.
        """
        # qpos[7:] slices away the 7-DoF freejoint of the base
        joint_pos = self.data.qpos[7:].copy()
        # qvel[6:] slices away the 6-DoF velocity of the base
        joint_vel = self.data.qvel[6:].copy()
        
        # --- MODIFIED: Observation is now relative to default pose ---
        # This is standard for unitree-rl: obs = (pos - default_pos), vel
        return np.concatenate([joint_pos - self.default_dof_pos, joint_vel])

    def step(self, action):
        # --- ENTIRELY MODIFIED: Use PD Position Control ---
        
        # Clip action from policy
        clipped_action = np.clip(action, -1.0, 1.0)
        
        # Calculate target joint positions
        # action is an offset from the default pose
        target_dof_pos = (clipped_action * self.action_scale) + self.default_dof_pos
        
        # Get current state
        current_dof_pos = self.data.qpos[7:]
        current_dof_vel = self.data.qvel[6:]
        
        # Calculate torques using PD controller
        position_error = target_dof_pos - current_dof_pos
        velocity_error = -current_dof_vel # Damping term
        
        torques = self.p_gains * position_error + self.d_gains * velocity_error
        
        # Get actuator torque limits
        ctrl_limit = self.model.actuator_ctrlrange[:, 1]
        applied_torques = np.clip(torques, -ctrl_limit, ctrl_limit)
        
        # Apply torques
        self.data.ctrl[:] = applied_torques
        
        # --- End of modification ---

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        
        # For now, reward is a placeholder.
        reward = 0.0
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- MODIFIED: Reset to default joint angles ---
        
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set joint positions to the default pose
        self.data.qpos[7:] = self.default_dof_pos
        
        # Set joint velocities to zero
        self.data.qvel[6:] = 0.0

        # Forward simulation to update sensor data and kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # --- End of modification ---

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None