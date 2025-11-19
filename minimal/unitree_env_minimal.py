import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

class UnitreeEnv(gym.Env):
    def __init__(self, model_path, render_mode=None, test_mode=False, frame_skip=4, **kwargs):
        super().__init__()
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.test_mode = test_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 0.325
        self.init_qvel = self.data.qvel.copy()

        # --- PD and Action Scaling ---
        self.action_scale = 0.15
        self.p_gains = np.full(self.model.nu, 15.0)
        self.d_gains = np.full(self.model.nu, 3.0)

        # --- Default pose ---
        self.default_dof_pos = np.array([
            -0.15,  0.8, -1.5,
             0.15,  0.8, -1.5,
            -0.15,  1.0, -1.5,
             0.15,  1.0, -1.5
        ])
        self.init_qpos[7:] = self.default_dof_pos

        # --- Joint limits ---
        hard_limits = self.model.jnt_range[1:13].copy()
        hard_min = hard_limits[:, 0]
        hard_max = hard_limits[:, 1]
        margin = 0.05
        total_amplitude = hard_max - hard_min
        dynamic_offset = total_amplitude * margin
        self.soft_jnt_min = hard_min + dynamic_offset
        self.soft_jnt_max = hard_max - dynamic_offset
        self.scale_to_max = self.soft_jnt_max - self.default_dof_pos
        self.scale_to_min = self.default_dof_pos - self.soft_jnt_min

        # --- Observation/Action Space ---
        obs_high = np.inf * np.ones(52)
        obs_low = -obs_high
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        action_high = np.ones(self.model.nu)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        # --- Command buffer ---
        self.commands = np.zeros(3)
        self.last_actions = np.zeros(self.action_space.shape)
        self.last_dof_vel = np.zeros(self.model.nu)
        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.zeros(4, dtype=bool)
        self.step_counter = 0
        self.max_episode_length = 6000

        # --- Reward config ---
        self.reward_scales = {
            "tracking_lin_vel": 5.0,
            "living_bonus": 0.2,
            "termination": -100.0,
            "joint_center_penalty": -2.0,  # Start with -1.0, tune as needed
            "side_velocity": -2.0,   # Penalize sideways movement
            "yaw_rate": -2.0,        # Penalize spinning
            "foot_clearance": 2.0,
            "contact_evenness": -2.5,  
        }
        self.tracking_sigma = 0.25

        # --- Indices ---
        self.feet_indices = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'FL_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'FR_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'RL_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'RR_foot')
        ])

        self.dt = frame_skip * self.model.opt.timestep


        self.termination_geom_indices = []
        self.penalised_geom_indices = []

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        
        # Get IDs for penalized bodies (from GO2RoughCfg.asset.penalize_contacts_on)
        penalized_body_names = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", 
                                "RL_thigh", "RL_calf", "RR_thigh", "RR_calf",
                                "base_1, base_2, base_3"]
        self.penalized_body_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) 
                            for name in penalized_body_names}

        for i in range(self.model.ngeom):
            geom_body_id = self.model.geom_bodyid[i]
            
            # Check if this geom's parent body is the 'base_link'
            if geom_body_id == self.base_body_id:
                # We also check its 'contype' to make sure it's a collision geom
                # (contype > 0 means it can collide)
                if self.model.geom_contype[i] > 0:
                    self.termination_geom_indices.append(i)
                
            # Check if this geom's parent body is in the penalized list
            if geom_body_id in self.penalized_body_ids:
                if self.model.geom_contype[i] > 0:
                    self.penalised_geom_indices.append(i)
        
        self.termination_geom_indices = np.array(self.termination_geom_indices)
        self.penalised_geom_indices = np.array(self.penalised_geom_indices)

        self.contact_history_len = 500  # Number of steps to look back
        self.foot_contact_history = np.zeros((self.contact_history_len, 4), dtype=np.float32)
        self.contact_history_ptr = 0


    def _map_actions_to_targets(self, action):
        positive_actions = np.clip(action, 0.0, 1.0)
        negative_actions = np.clip(action, -1.0, 0.0)
        target_dof_pos = (self.default_dof_pos +
                          positive_actions * self.scale_to_max +
                          negative_actions * self.scale_to_min)
        return target_dof_pos

    def print_base_velocities(self):
        """
        Print the robot's base linear and angular velocities in the body frame.
        Call this method after _get_obs() in each step.
        """
        # Ensure _get_obs() has been called so self.base_lin_vel and self.base_ang_vel are up to date
        lin_vel = self.base_lin_vel  # [vx, vy, vz] in body frame
        ang_vel = self.base_ang_vel  # [wx, wy, wz] in body frame (rad/s)
        print(f"Linear velocity (body frame):  x={lin_vel[0]: .3f}  y={lin_vel[1]: .3f}  z={lin_vel[2]: .3f}")
        print(f"Angular velocity (body frame): roll={ang_vel[0]: .3f}  pitch={ang_vel[1]: .3f}  yaw={ang_vel[2]: .3f}")

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        self.dof_pos = qpos[7:]
        self.dof_vel = qvel[6:]
        base_quat = self.data.sensor('imu_quat').data.copy()
        base_rot_mat_flat = np.zeros(9)
        mujoco.mju_quat2Mat(base_rot_mat_flat, base_quat)
        base_rot_mat = base_rot_mat_flat.reshape(3, 3)
        base_lin_vel_world = self.data.qvel[:3].copy()
        self.base_lin_vel = base_rot_mat.T @ base_lin_vel_world
        self.base_ang_vel = self.data.sensor('imu_gyro').data.copy()
        gravity_world = np.array([0, 0, -9.81])
        self.projected_gravity = base_rot_mat.T @ gravity_world

        # --- Foot Contacts (vertical force) ---
        contact_threshold = 1.0
        fl_force = self.data.sensor('FL_foot_force').data
        fr_force = self.data.sensor('FR_foot_force').data
        rl_force = self.data.sensor('RL_foot_force').data
        rr_force = self.data.sensor('RR_foot_force').data
        fl_contact = float(np.abs(fl_force[2]) > contact_threshold)
        fr_contact = float(np.abs(fr_force[2]) > contact_threshold)
        rl_contact = float(np.abs(rl_force[2]) > contact_threshold)
        rr_contact = float(np.abs(rr_force[2]) > contact_threshold)
        foot_contacts_float = np.array([fl_contact, fr_contact, rl_contact, rr_contact], dtype=np.float32)
        self.current_foot_contacts = foot_contacts_float

        obs = np.concatenate((
            self.base_lin_vel * 2.0,
            self.base_ang_vel * 0.25,
            self.projected_gravity,
            self.commands,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * 0.05,
            self.last_actions,
            foot_contacts_float
        )).astype(np.float32)
        return obs

    def _reward_contact_evenness(self):
        # Compute average contact for each foot over the window
        avg_contacts = np.mean(self.foot_contact_history, axis=0)
        # Penalize the difference between max and min contact fraction
        penalty = np.max(avg_contacts) - np.min(avg_contacts)
        return penalty  # Use a negative scale in reward_scales


    def step(self, action):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        target_dof_pos = self._map_actions_to_targets(clipped_action)
        current_dof_pos = self.data.qpos[7:]
        current_dof_vel = self.data.qvel[6:]
        position_error = (target_dof_pos - current_dof_pos)
        velocity_error = -current_dof_vel
        torques = self.p_gains * position_error + self.d_gains * velocity_error
        ctrl_limit = self.model.actuator_ctrlrange[:, 1]
        applied_torques = np.clip(torques, -ctrl_limit, ctrl_limit)
        self.data.ctrl[:] = applied_torques
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_counter += 1

        # --- Command resampling every 8 seconds ---
        resampling_time_steps = int(8.0 / self.dt)
        if self.step_counter % resampling_time_steps == 0:
            self._resample_commands()

        observation = self._get_obs()

        self.foot_contact_history[self.contact_history_ptr] = self.current_foot_contacts
        self.contact_history_ptr = (self.contact_history_ptr + 1) % self.contact_history_len

        terminated, truncated = self._check_termination()
        reward = self._compute_reward(terminated, truncated)
        self.last_actions = clipped_action.copy()

        # --- NaN/Inf checks ---
        if not np.isfinite(observation).all():
            print("WARNING: NaN or Inf in observation!")
        if not np.isfinite(reward):
            print("WARNING: NaN or Inf in reward!")

        info = {}
        return observation, reward, terminated, truncated, info

    def _reward_termination(self, terminated, time_out):
        # Only penalize if terminated by failure, not by timeout
        return terminated * (not time_out)

    def _compute_reward(self, terminated, time_out):
        # Only forward velocity tracking and living bonus
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]))
        tracking_lin_vel = np.exp(-lin_vel_error / self.tracking_sigma)

        living_bonus = 1.0
        reward = self.reward_scales["tracking_lin_vel"] * tracking_lin_vel + \
                 self.reward_scales["living_bonus"] * living_bonus
        
        if "termination" in self.reward_scales:
            term_rew = self._reward_termination(terminated, time_out) * self.reward_scales["termination"]
            reward += term_rew

        if "joint_center_penalty" in self.reward_scales:
            joint_penalty = self._reward_joint_center_penalty() * self.reward_scales["joint_center_penalty"]
            reward += joint_penalty

        if "side_velocity" in self.reward_scales:
            side_vel_penalty = self._reward_side_velocity() * self.reward_scales["side_velocity"]
            reward += side_vel_penalty
        if "yaw_rate" in self.reward_scales:
            yaw_rate_penalty = self._reward_yaw_rate() * self.reward_scales["yaw_rate"]
            reward += yaw_rate_penalty
        if "foot_clearance" in self.reward_scales:
            foot_clearance_reward = self._reward_foot_clearance() * self.reward_scales["foot_clearance"]
            reward += foot_clearance_reward
        if "contact_evenness" in self.reward_scales:
            contact_evenness_penalty = self._reward_contact_evenness() * self.reward_scales["contact_evenness"]
            reward += contact_evenness_penalty  # Subtract since it's a penalty
        return reward
    
    def _reward_foot_clearance(self):
        # Get foot site positions (z)
        foot_sites = ['FL_site', 'FR_site', 'RL_site', 'RR_site']
        clearance = 0.0
        for i, site in enumerate(foot_sites):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
            foot_z = self.data.site_xpos[site_id][2]
            # Only reward if foot is NOT in contact
            if self.current_foot_contacts[i] < 0.5:
                clearance += foot_z
        return clearance / len(foot_sites)

    def _reward_side_velocity(self):
        # Only penalize if commanded to walk forward
        if abs(self.commands[0]) > 0.2 and abs(self.commands[1]) < 0.1 and abs(self.commands[2]) < 0.1:
            return np.square(self.base_ang_vel[2])
        return 0.0

    def _reward_yaw_rate(self):
        if abs(self.commands[0]) > 0.2 and abs(self.commands[1]) < 0.1 and abs(self.commands[2]) < 0.1:
            return np.square(self.base_ang_vel[2])
        return 0.0


    def _reward_joint_center_penalty(self):
        """
        Penalizes joint positions that deviate too far from default positions.
        No penalty within ±0.26 radians; exponential penalty beyond.
        """

        free_ranges = np.array([
            0.1, 0.15, 0.2,
            0.1, 0.15, 0.2,
            0.1, 0.15, 0.2,
            0.1, 0.15, 0.2,
        ])

        #free_joint_radius = 0.26  # radians
        joint_deviations = np.abs(self.dof_pos - self.default_dof_pos)
        excess_deviations = np.maximum(0.0, joint_deviations - free_ranges)

        
        joint_weights = np.array([1.5, 1.0, 3.0] * 4)  # Example: hips > thighs > calves
        weighted_penalties = joint_weights * (np.exp(excess_deviations) - 1.0)
        return np.sum(weighted_penalties)

    def _check_termination(self):
        # Loosened orientation and height thresholds
        roll, pitch, yaw = self._quat_to_rpy(self.data.sensor('imu_quat').data)
        orientation_violated = abs(roll) > 1.2 or abs(pitch) > 1.5
        base_height = self.data.qpos[2]
        body_too_low = base_height < 0.25
        truncated = self.step_counter >= self.max_episode_length

        # --- 1. Check for Base Contact (Termination) ---
        contacts, contact_geom_ids = self._get_contact_info()
        base_contact = False
        contact_threshold = 1.0  # From Isaac Gym

        # Check against the GEOM list from __init__
        for geom_id in self.termination_geom_indices:
            if geom_id in contacts: # 'contacts' is the dictionary {geom_id: force}
                if np.linalg.norm(contacts[geom_id]) > contact_threshold:
                    base_contact = True
                    if self.render_mode == 'human':
                        print(f"TERMINATION: Base Contact (Geom ID {geom_id} hit ground with force {np.linalg.norm(contacts[geom_id]):.2f})")
                    break


        return base_contact or orientation_violated or body_too_low or truncated, truncated

    def _get_contact_info(self):
        """Helper to get contact forces and identify contacting geoms."""
        # Dictionary: geom_id -> total_force_vector
        contacts = {} 
        # Set: geom_id
        contact_geom_ids = set() 
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if one geom is ground (geomid=0) and the other is part of the robot
            robot_geom = -1
            if geom1 == 0 and geom2 > 0:
                robot_geom = geom2
            elif geom2 == 0 and geom1 > 0:
                 robot_geom = geom1
                 
            if robot_geom != -1:
                 # Calculate force vector in world frame
                 force_vector = np.zeros(6)
                 mujoco.mj_contactForce(self.model, self.data, i, force_vector)
                 contact_force_world = force_vector[0:3] # Only translational force

                 if robot_geom not in contacts:
                      contacts[robot_geom] = np.zeros(3)
                 contacts[robot_geom] += contact_force_world
                 contact_geom_ids.add(robot_geom)

        # This function returns TWO values
        return contacts, contact_geom_ids

    def _quat_to_rpy(self, q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._resample_commands()
        self.step_counter = 0
        observation = self._get_obs()
        if not np.isfinite(observation).all():
            print("WARNING: NaN or Inf in reset observation!")
        info = {}
        return observation, info

    def _resample_commands(self):
        # Only forward command for now
        self.commands[:] = 0.0
        self.commands[0] = np.random.uniform(0.1, 0.7)  # Forward velocity only

    def render(self):
        if self.render_mode == "human" and self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception as e:
                if "Window" in str(e) or "glfw" in str(e).lower():
                    print("Viewer window closed.")
                    self.viewer = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
