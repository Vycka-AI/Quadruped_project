import gymnasium as gym
import numpy as np
import mujoco

class Go2SuspendedEnv(gym.Env):
    """
    A Gymnasium environment for the Unitree Go2 robot suspended in the air.
    The robot's base is fixed, and the agent controls the 12 leg joints.
    """
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Get the ID of the "home" keyframe from the XML for easy resets
        self.key_home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')

        # --- Define Action and Observation Spaces ---
        # Action space: 12 actuators for the 12 joints (from go2.xml)
        # self.model.nu gives the number of actuators, which is 12.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation space: 12 joint positions + 12 joint velocities
        # self.model.njnt gives the number of joints.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.model.njnt,), dtype=np.float32
        )

    def _get_obs(self):
        """
        Returns the current observation, which is the joint positions and velocities.
        """
        # qpos[7:] slices away the 7-DoF freejoint of the base
        joint_pos = self.data.qpos[7:].copy()
        # qvel[6:] slices away the 6-DoF velocity of the base
        joint_vel = self.data.qvel[6:].copy()
        return np.concatenate([joint_pos, joint_vel])

    def step(self, action):
        # Scale the agent's [-1, 1] actions to the actuator's control range
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = ctrl_range[:, 0] + (action + 1.0) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[:] = np.clip(scaled_action, ctrl_range[:, 0], ctrl_range[:, 1])

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        
        # For now, reward is a placeholder. For a real RL task, you would
        # calculate a reward based on the robot's state and actions.
        reward = 0.0
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the simulation to the "home" keyframe defined in go2.xml
        #mujoco.mj_resetDataKey(self.model, self.data, self.key_home_id)

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