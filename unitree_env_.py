import gymnasium as gym
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

class UnitreeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

class UnitreeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        
        self.n_substeps = 10
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        self.foot_geom_ids = [
            self.model.geom(name).id for name in ["FL", "FR", "RL", "RR"]
        ]
        # This was the missing line:
        self.floor_geom_id = self.model.geom("floor").id

        # --- NEW: Episode step counter ---
        self.episode_step = 0

        self.target_velocity = np.zeros(3)

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
        
        # --- NEW: Get foot force feedback ---
        # Read the 3-axis force from each of the four new sensors
        fl_force = self.data.sensor('FL_foot_force').data.copy()
        fr_force = self.data.sensor('FR_foot_force').data.copy()
        rl_force = self.data.sensor('RL_foot_force').data.copy()
        rr_force = self.data.sensor('RR_foot_force').data.copy()



        observation = np.concatenate([
            joint_pos, joint_vel, base_lin_vel, base_ang_vel, torso_quat,
            self.target_velocity, self.last_action
        ])
        return observation

    def step(self, action):
        true_control_range = self.model.actuator_ctrlrange
        low_bound, high_bound = true_control_range[:, 0], true_control_range[:, 1]
        rescaled_action = low_bound + (action + 1.0) * 0.5 * (high_bound - low_bound)
        self.data.ctrl[:] = np.clip(rescaled_action, low_bound, high_bound)
        
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        self.last_action = action.copy()
        
        # --- NEW: Increment step counter ---
        self.episode_step += 1

        observation = self.get_observation()
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

        # --- NEW: Reset step counter ---
        self.episode_step = 0
        
        lin_vel_x = np.random.uniform(0.3, 0.7)
        ang_vel_z = np.random.uniform(-0.3, 0.3)
        self.target_velocity = np.array([lin_vel_x, 0.0, ang_vel_z])

        self.last_action.fill(0)
        return self.get_observation(), {}

    
    def calculate_reward(self, action):
        base_lin_vel = self.data.qvel[:3]
        base_ang_vel = self.data.qvel[3:6]
        joint_vel = self.data.qvel[6:]

        lin_vel_reward = np.exp(-2.0 * np.sum(np.square(self.target_velocity[:2] - base_lin_vel[:2])))
        ang_vel_reward = np.exp(-1.0 * np.square(self.target_velocity[2] - base_ang_vel[2]))

        # --- NEW: Exponential Time-Based Survival Reward --- 📈
        # The reward for surviving grows exponentially with the number of steps.
        # The (np.exp(...) - 1) form ensures the reward starts at 0.
        # The 0.1 and 0.005 are scaling factors to keep it from exploding.
        survival_reward = 0.3 * (np.exp(0.005 * self.episode_step) - 1)

        # --- Penalties ---
        torque_penalty = 0.0002 * np.sum(np.square(self.data.ctrl))
        joint_vel_penalty = 0.001 * np.sum(np.square(joint_vel))
        jerk_penalty = 0.01 * np.sum(np.square(action - self.last_action))
        other_ang_vel_penalty = 0.1 * (np.square(base_ang_vel[0]) + np.square(base_ang_vel[1]))
        other_lin_vel_penalty = 0.1 * (np.square(base_lin_vel[1]) + np.square(base_lin_vel[2]))


        # --- NEW: Contact Reward and Penalty --- 🐾
        foot_contact_reward, body_contact_penalty = self._check_contacts()


        total_reward = (
            lin_vel_reward +
            ang_vel_reward +
            survival_reward + # Add the new survival reward
            -torque_penalty -
            -joint_vel_penalty -
            -jerk_penalty -
            -other_ang_vel_penalty -
            -other_lin_vel_penalty +
            0.1 * foot_contact_reward + # Add the new reward
            -body_contact_penalty
        )
        return total_reward

    # --- NEW HELPER FUNCTION ---
    def _check_contacts(self):
        """
        Checks for foot-ground and body-ground contacts.
        """
        foot_contacts = 0
        body_contacts = 0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if one of the geoms is the floor
            if geom1 == self.floor_geom_id or geom2 == self.floor_geom_id:
                # Find which geom is NOT the floor
                other_geom = geom2 if geom1 == self.floor_geom_id else geom1
                
                # If the other geom is a foot, it's a good contact
                if other_geom in self.foot_geom_ids:
                    foot_contacts += 1
                # If the other geom is part of the robot but not a foot, it's a bad contact
                elif self.model.geom_bodyid[other_geom] > 0: # bodyid > 0 means it's part of the robot
                    body_contacts += 1
        
        # Reward having 1 or 2 feet on the ground, penalize having 0 or >2
        # This encourages a walking/trotting gait
        foot_reward = 0
        if foot_contacts == 1 or foot_contacts == 2:
            foot_reward = foot_contacts
        elif foot_contacts > 2:
            foot_reward = - (foot_contacts - 2)
            
        # A large penalty for any body-ground contact
        body_penalty = 2.0 * body_contacts

        return foot_reward, body_penalty

    def is_terminated(self):
        torso_height = self.data.qpos[2]
        torso_quat_mujoco = self.data.sensor('imu_quat').data
        rot = Rotation.from_quat([torso_quat_mujoco[1], torso_quat_mujoco[2], torso_quat_mujoco[3], torso_quat_mujoco[0]])
        roll, pitch = rot.as_euler('xyz', degrees=False)[0:2]
        
        is_flipped = (abs(roll) > -0.15 + np.pi / 2) or (abs(pitch) > -0.15 +  np.pi / 2)
        is_fallen = torso_height < 0.15
        
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