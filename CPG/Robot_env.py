import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer # <-- ADD THIS
# Add torch if you want rewards calculated with it, otherwise use numpy
import torch 
import time
from CPG_Network import CPGNetwork

class UnitreeEnv(gym.Env):
    def __init__(self, model_path, render_mode=None, frame_skip=4, **kwargs):
        super().__init__()
        #self.calf_joint_indices = np.array([2, 5, 8, 11])
        #self.calf_joint_indices = np.array([1, 4, 7, 10])
        self.calf_joint_indices = np.array([1, 4, 7, 10])
        
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.init_qpos = self.data.qpos.copy()
        #self.init_qpos[2] = 0.325
        
        self.init_qvel = self.data.qvel.copy()
        
        self.default_dof_pos = np.array([
            0.15,  0.8, -1.5,  # FR_hip_joint, FR_thigh_joint, FR_calf_joint
            -0.15,  0.8, -1.5,  # FL_hip_joint, FL_thigh_joint, FL_calf_joint
            0.15,  1.0, -1.5,  # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            -0.15,  1.0, -1.5   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
        ])

        self.init_qpos[7:] = self.default_dof_pos

        self.p_gains = np.full(self.model.nu, 70.0)  # VERY low P-gain
        self.d_gains = np.full(self.model.nu, 2.5)  # VERY low D-gain
        #self.p_gains = np.full(self.model.nu, 0.0)  # VERY low P-gain
        #self.d_gains = np.full(self.model.nu, 0.0)  # VERY low D-gain

        self.obs_scales_lin_vel = 2.0
        self.obs_scales_ang_vel = 0.25
        self.obs_scales_dof_pos = 1.0
        self.obs_scales_dof_vel = 0.05

        self.step_counter = 0
        self.max_episode_length = 3000

        self.termination_geom_indices = []
        self.penalised_geom_indices = []
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        
        penalized_body_names = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", 
                                "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]
        self.penalized_body_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) 
                           for name in penalized_body_names}

        for i in range(self.model.ngeom):
            geom_body_id = self.model.geom_bodyid[i]
            if geom_body_id == self.base_body_id:
                if self.model.geom_contype[i] > 0:
                    self.termination_geom_indices.append(i)
            if geom_body_id in self.penalized_body_ids:
                if self.model.geom_contype[i] > 0:
                    self.penalised_geom_indices.append(i)
        
        self.termination_geom_indices = np.array(self.termination_geom_indices)
        self.penalised_geom_indices = np.array(self.penalised_geom_indices)
        
        print(f"Found {len(self.termination_geom_indices)} termination geoms (base).")
        print(f"Found {len(self.penalised_geom_indices)} penalized geoms (thighs/calves).")

        self.dt = frame_skip * self.model.opt.timestep
        self.reward_scales = {
             # ... (your reward scales)
        }
        
        # --- (your other __init__ code) ---
        
        self.tracking_sigma = 0.25
        self.base_height_target = 0.25
        self.soft_dof_pos_limit = 0.9
        self.soft_dof_vel_limit = 1.0
        self.soft_torque_limit = 1.0
        self.max_contact_force = 100.0


        self.reward_scales = {
            "lin_vel_z": -0.2,
            "ang_vel_xy": -0.05,
            "orientation": -1.0, # Default is 0
            "base_height": -5.0, # Default is 0
            "torques": -0.0002, # From GO2RoughCfg
            "dof_vel": -0.01, # Default is 0
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "collision": -0.1,
            "termination": -12.0, # Default is 0
            "dof_pos_limits": -10.0, # From GO2RoughCfg
            "dof_vel_limits": 0.0, # Default not specified or 0
            "torque_limits": 0.0, # Default not specified or 0
            "tracking_lin_vel": 4.0,# * self.dt,
            "tracking_ang_vel": 1.0,# * self.dt,
            "feet_air_time": 1.0,
            "stumble": -0.0, # Default is 0
            "stand_still": -0.1, # Default is 0
            "feet_contact_forces": 0.0, # Default not specified or 0
            "living_bonus": 0.0,# * self.dt, # <-- ADD THIS REWARD
            "feet_stuck": -1.0,
            "large_tracking_error": -1.0
        }

        self.active_reward_scales = {k: v for k, v in self.reward_scales.items() if v != 0.0}

        self.feet_indices = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'FL_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'FR_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'RL_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'RR_foot')
        ])
        
        cpg_state_dim = 8
        # --- CPG-RL SPACE DEFINITIONS ---

        # 1. DEFINE ACTION SPACE (8-dim)
        # The agent outputs 8 values: 4 for amplitude, 4 for frequency
        # All actions are in the normalized range [-1, 1]
        action_dim = 8 
        action_high = np.ones(action_dim)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        # 2. DEFINE OBSERVATION SPACE (52 + 8 = 60-dim)
        # Original obs size = 52
        # We add 8 CPG states (x, y for each of 4 oscillators)
        cpg_state_dim = 4
        obs_dim = 52 + cpg_state_dim # 60
        
        obs_high = np.inf * np.ones(obs_dim)
        obs_low = -obs_high
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        # --- BUFFERS ---
        self.commands = np.zeros(3) 
        # last_actions is now 8-dim
        self.last_actions = np.zeros(self.action_space.shape) 
        # Buffer for the *previous* action (for action_rate reward)
        self.prev_last_actions = np.zeros(self.action_space.shape)
        # Buffer to store the CPG state for the observation
        self.cpg_states = np.zeros(cpg_state_dim)
        self.last_dof_vel = np.zeros(self.model.nu)
        self.feet_air_time = np.zeros(len(self.feet_indices))
        self.last_contacts = np.zeros(len(self.feet_indices), dtype=bool)
        
        self.dof_pos = None
        self.dof_vel = None
        self.torques = np.zeros(self.model.nu)

        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.zeros(3)

        self.step_counter = 0
        self.last_target_dof_pos = self.default_dof_pos.copy()
        # ======================================================================
        # --- CPG INITIALIZATION ---
        # Create the CPG network, passing it the sim dt and base pose
        self.cpg_network = CPGNetwork(dt=self.dt, base_positions=self.default_dof_pos)
        # ======================================================================

    # ... (your _get_contact_info, _get_obs, _sample_value functions) ...
    # === PD CONTROLLER TEST PARAMETERS ===
        self.run_oscillation_test = True # Set to True to run the test
        self.test_amplitude = 0.5  # Radians (how far the joint moves)
        self.test_frequency = 1.0  # Hz (how fast it moves, 1.0 = 1 cycle/sec)
    def cyclic_step(self):
        """
        Calculates a cyclical (sine wave) target position for the calf joints.
        Used for testing the PD controller's tracking performance.
        """
        # 1. Calculate the current time in the simulation
        current_time = self.step_counter * self.dt
        
        # 2. Calculate the cyclical offset (a sine wave)
        #    This moves from -amplitude to +amplitude and back
        offset = self.test_amplitude * np.sin(
            2 * np.pi * self.test_frequency * current_time
        )
        
        # 3. Start with the default "home" pose
        target_dof_pos = self.default_dof_pos.copy()
        
        # 4. Add the offset to the default position of the calf joints
        #    so they oscillate around their 'home' position
        target_dof_pos[self.calf_joint_indices] += offset
        
        return target_dof_pos

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
            
            robot_geom = -1
            if geom1 == 0 and geom2 > 0:
                robot_geom = geom2
            elif geom2 == 0 and geom1 > 0:
                robot_geom = geom1
                
            if robot_geom != -1:
                force_vector = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force_vector)
                contact_force_world = force_vector[0:3] 

                if robot_geom not in contacts:
                    contacts[robot_geom] = np.zeros(3)
                contacts[robot_geom] += contact_force_world
                contact_geom_ids.add(robot_geom)

        return contacts, contact_geom_ids

    def _get_obs(self):
        """ Computes and returns the observation vector. """
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

        contact_threshold = 1.0 
        fl_force = self.data.sensor('FL_foot_force').data
        fr_force = self.data.sensor('FR_foot_force').data
        rl_force = self.data.sensor('RL_foot_force').data
        rr_force = self.data.sensor('RR_foot_force').data

        fl_contact = float(np.abs(fl_force[2]) > contact_threshold) if len(fl_force) >=3 else float(np.abs(fl_force) > contact_threshold)
        fr_contact = float(np.abs(fr_force[2]) > contact_threshold) if len(fr_force) >=3 else float(np.abs(fr_force) > contact_threshold)
        rl_contact = float(np.abs(rl_force[2]) > contact_threshold) if len(rl_force) >=3 else float(np.abs(rl_force) > contact_threshold)
        rr_contact = float(np.abs(rr_force[2]) > contact_threshold) if len(rr_force) >=3 else float(np.abs(rr_force) > contact_threshold)

        foot_contacts_float = np.array([fl_contact, fr_contact, rl_contact, rr_contact], dtype=np.float32)
        self.current_foot_contacts = foot_contacts_float

        obs = np.concatenate((
            self.base_lin_vel * self.obs_scales_lin_vel,
            self.base_ang_vel * self.obs_scales_ang_vel,
            self.projected_gravity,
            self.commands,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales_dof_pos,
            self.dof_vel * self.obs_scales_dof_vel,
            self.last_actions,
            foot_contacts_float,
            self.cpg_states  # <-- ADD THIS
        )).astype(np.float32)
        return obs

    def _sample_value(self, min_val, max_val, dead_zone=0.2):
        """Samples a value from a range, ensuring it's outside a 'dead zone'"""
        val = np.random.uniform(min_val, max_val)
        if 0 < abs(val) < dead_zone:
            val = dead_zone * np.sign(val)
        return val
    
    # ==========================================================================
    # 4. REPLACE YOUR STEP FUNCTION
    # ==========================================================================
    def step(self, action):
        """ Applies CPG action, simulates, calculates rewards, and returns results. """
        
        # --- CPG CONTROLLER ---
        # The 'action' from the agent is IGNORED.
        # We get target positions directly from our hard-coded CPG.
        #target_dof_pos = self.default_dof_pos
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        target_dof_pos, self.cpg_states = self.cpg_network.step(rl_action=clipped_action)
        #target_dof_pos = self.cyclic_step()
        #print(f"CPG Target DOF Positions:\n{target_dof_pos.reshape(4,3)}") # Debug print
        
        # --- For Reward/Obs Buffers ---
        # We still need to populate these buffers for compatibility.
        # We'll just pretend the "action" was zero.
        # --- CPG-RL CONTROLLER ---
        # 1. Clip the raw RL action (this is the 8-dim CPG param vector)
        
        
        # 2. Store the "previous" action for the action_rate reward
        self.prev_last_actions = self.last_actions.copy()

        # 3. Pass the action to the CPG network
        #    (Make sure your CPGNetwork.step returns two values!)
        #target_dof_pos, self.cpg_states = self.cpg_network.step(rl_action=clipped_action)
        
        # 4. Store this action for the *next* observation and action_rate
        #self.last_actions = clipped_action
        
        # --- PD Controller (Unchanged) ---
        # (Your PD controller code is here)
        # ---------------------------
        
        # PD Controller: Calculate torques
        current_dof_pos = self.data.qpos[7:]
        
        #print(f"Current DOF Positions:\n{current_dof_pos.reshape(4,3)}") # Debug print
        current_dof_vel = self.data.qvel[6:]
        
        position_error = (target_dof_pos - current_dof_pos)
        
        #velocity_error = (target_dof_vel - current_dof_vel)
        velocity_error = -current_dof_vel 
        
        self.torques = self.p_gains * position_error + self.d_gains * velocity_error
        
        ctrl_limit = self.model.actuator_ctrlrange[:, 1]
        #print(f"Control Limits:\n{ctrl_limit.reshape(4,3)}") # Debug print
        applied_torques = np.clip(self.torques, -ctrl_limit, ctrl_limit)
        
        # --- MODIFIED DEBUG BLOCK ---
        """
        if self.render_mode == "human" and self.step_counter % 50 == 0: # Print every 50 steps
            print("\n--- CPG Controller Debug (Step", self.step_counter, ") ---")
            # Print CPG target position (Front-Left leg)
            print(f"  CPG Target Pos (FL leg):   {target_dof_pos[3:6]}") # FL is index 1 (joints 3,4,5)

            print(f" Current Pos (FL leg):   {current_dof_pos[3:6]}") # FL is index 1 (joints 3,4,5)
            # Print default position (Front-Left leg)
            print(f"  Default Pos (FL leg):    {self.default_dof_pos[3:6]}")

            #print(f"  Position error (FL leg):    {position_error[3:6]}")
            # Print final torque (Front-Left leg)
            print(f"  Applied Torque (FL leg): {applied_torques[3:6]}")

            print(f"Current dt: {self.dt}")
        """
        self.step_counter += 1
        # -------------------------------

        resampling_time_steps = int(3.0 / self.dt) 
        if self.step_counter % resampling_time_steps == 0:
            self._resample_commands()


        # --- Simulate ---
        self.data.ctrl[:] = applied_torques
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        
        # --- Get Observation ---
        observation = self._get_obs() 

        # --- Calculate Rewards (Dummy values for now) ---
        # (Reward calculation is complex and not implemented in your snippet)

        terminated = False 

        terminated, truncated = self._check_termination() # This already checks max_episode_length
        reward, reward_info = self._compute_reward(terminated, truncated)
        
        # --- Check Terminations (Dummy values for now) ---
        
        truncated = self.step_counter >= self.max_episode_length
        
        # --- Render ---
        if self.render_mode == "human":
            self.render() # Call the new render method

        # --- Update Buffers (for next step's obs/rewards) ---
        
        info = {}
        info.update(reward_info)

        #if terminated or truncated:
        #    self.reset()
        # Add any other relevant info

        return observation, reward, terminated, truncated, info

    def _check_termination(self):
        """ Checks if the episode should terminate. """
        
        # Get GEOM-based contact info
        contacts, contact_geom_ids = self._get_contact_info()

        # --- 1. Check for Base Contact (Termination) ---
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
        
        # --- Check orientation ---
        # Use projected gravity calculated in _get_obs
        orientation_limit_roll = 0.8 # From legged_robot.py check_termination
        orientation_limit_pitch = 1.0 # From legged_robot.py check_termination
        
        # Need roll/pitch from projected gravity or quaternion
        # Example: Get roll/pitch from quat
        roll, pitch, yaw = self._quat_to_rpy(self.data.sensor('imu_quat').data)
        orientation_violated = abs(roll) > orientation_limit_roll or abs(pitch) > orientation_limit_pitch

        # --- 3. ADDED: Check for low body height ---
        low_height_threshold = 0.1  # 15cm. Tune this value as needed!
        base_height = self.data.qpos[2] # Assuming z-height is index 2
        body_too_low = base_height < low_height_threshold

        self.step_counter += 1
        truncated = self.step_counter >= self.max_episode_length

        # --- Check for rendering and print termination reason ---
        if self.render_mode == 'human':
            if base_contact:
                # This will print right when the robot's base hits the ground
                print("TERMINATION: Base Contact (Torso hit the ground)")
            if orientation_violated:
                # This will print if the robot flips over
                print(f"TERMINATION: Orientation Violation (Roll: {roll:.2f}, Pitch: {pitch:.2f})")
            if body_too_low:
                print(f"TERMINATION: Body Too Low (Height: {base_height:.2f} < {low_height_threshold})")
            # -----------------------------------
        # --------------------------------------------------------

        terminated = base_contact or orientation_violated or body_too_low
        return terminated, truncated


    def _compute_reward(self, terminated, time_out):
        """ Calculates the reward based on active reward functions. """
        total_reward = 0.0
        reward_info = {}

        # Use torch for calculations if desired for consistency with legged_gym
        # Or keep using numpy
        
        # --- Call individual reward functions ---
        # Note: These functions now need to exist in this class and use MuJoCo data access
        # Ensure internal state variables (self.base_lin_vel etc.) are updated in _get_obs or step
        
        # Example using a helper function for potentially missing rewards:
        def get_reward_or_zero(name):
             func_name = f"_reward_{name}"
             if hasattr(self, func_name) and name in self.active_reward_scales:
                 # Calculate reward using numpy/torch
                 # For torch, convert numpy arrays to tensors first:
                 # rew = getattr(self, func_name)(torch.from_numpy(self.some_state)).numpy()
                 # For numpy:
                 rew = getattr(self, func_name)()
                 scaled_rew = rew * self.active_reward_scales[name]
                 reward_info[f"reward_{name}"] = scaled_rew
                 return scaled_rew
             return 0.0

        for name in self.active_reward_scales.keys():
            if name != "termination": # Termination handled separately
                 total_reward += get_reward_or_zero(name)

        # --- Clip negative rewards if configured ---
        # if self.cfg.rewards.only_positive_rewards: # Need to add this config
        #    total_reward = np.clip(total_reward, a_min=0.0, a_max=None)

        # --- Add termination reward ---
        # Need termination check logic first to set self.reset_buf equivalent
        # Need time_out logic similar to legged_gym
        #time_out = False # Replace with actual check
        if "termination" in self.active_reward_scales:
            term_rew = self._reward_termination(terminated, time_out) * self.active_reward_scales["termination"]
            total_reward += term_rew
            reward_info["reward_termination"] = term_rew

        return total_reward, reward_info

    def _reward_large_tracking_error(self):
        """
        Penalizes the agent for a large velocity tracking error
        when a non-zero command is given.
        """
        command_norm = np.linalg.norm(self.commands[:2])
        if command_norm > 0.2: # Only penalize if commanded to move
            
            # Calculate the squared error, just like in _reward_tracking_lin_vel
            lin_vel_error_sq = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]))
            
            # Define a threshold for "too high" error.
            # 1.0 m/s error squared = (1.0)^2 = 1.0
            # 0.8 m/s error squared = (0.8)^2 = 0.64
            # Let's use 0.5 m/s as the "failure" threshold
            error_threshold_sq = 0.25 # (0.5 m/s)^2
            
            if lin_vel_error_sq > error_threshold_sq:
                # Penalize based on how bad the error is
                # This returns a value from 0 up to 1.0
                penalty = np.clip((lin_vel_error_sq - error_threshold_sq) / (4.0 - error_threshold_sq), 0, 1.0)
                return penalty
        
        return 0.0 # No penalty

    def _reward_feet_stuck(self):
        """
        Penalizes the agent for keeping all feet on the ground
        when a non-zero command is given. This forces it to learn a gait.
        """
        command_norm = np.linalg.norm(self.commands[:2])
        if command_norm > 0.2:
            # Check if all 4 feet are on the ground
            # self.current_foot_contacts is a float array [0. or 1.]
            all_feet_in_contact = np.all(self.current_foot_contacts > 0)
            if all_feet_in_contact:
                return 1.0 # Return 1.0 (will be a penalty)
        
        return 0.0 # No penalty

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[2]) # Index 2 for z

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.base_ang_vel[:2])) # Indices 0 and 1 for x, y

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # Uses projected gravity calculated in _get_obs
        return np.sum(np.square(self.projected_gravity[:2])) # Indices 0 and 1 for x, y component

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.data.qpos[2] # Assuming z is index 2 for base position
        return np.square(base_height - self.base_height_target)

    def _reward_torques(self):
        # Penalize torques (use applied torques)
        return np.sum(np.square(self.data.ctrl)) # Use applied control torques

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # Use dof_vel calculated in _get_obs
        return np.sum(np.square(self.dof_vel))

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        # Use dof_vel (current) and last_dof_vel (previous step)
        acceleration = (self.dof_vel - self.last_dof_vel) / self.dt
        return np.sum(np.square(acceleration))

    def _reward_action_rate(self):
        # Penalize changes in actions
        # Use last_actions (current) and prev_last_actions (previous step)
        return np.sum(np.square(self.last_actions - self.prev_last_actions))

    def _reward_collision(self):
        # Penalize collisions on selected bodies (geoms)
        
        # Get GEOM-based contact info
        contacts, contact_geom_ids = self._get_contact_info() 
        
        collision_count = 0.0
        contact_threshold = 0.1 # From legged_gym _reward_collision
        
        # Check against the GEOM list from __init__
        for geom_id in self.penalised_geom_indices: 
             if geom_id in contacts: # 'contacts' is the dictionary {geom_id: force}
                 force_norm = np.linalg.norm(contacts[geom_id])
                 if force_norm > contact_threshold:
                     collision_count += 1.0 
        
        return collision_count

    def _reward_termination(self, terminated, time_out):
        # Terminal reward / penalty (use flags from step/check_termination)
        # Assumes terminated=True if reset, time_out=True if max episode length reached
        return terminated * (not time_out) # 1.0 if terminated by condition, 0.0 if time_out

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # Need self.dof_pos_limits (soft limits) defined in __init__
        # Assuming self.dof_pos_limits is shape (num_dof, 2) [min, max]
        # And self.dof_pos is shape (num_dof,)
        # out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit -> numpy
        # out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.) # upper limit -> numpy
        
        # Read hard limits from model for calculation
        model_limits = self.model.jnt_range # Shape (num_jnt, 2)
        # Assume joint order matches dof order
        hard_limits = model_limits[:self.model.nu] # Get limits for actuated joints
        
        midpoint = (hard_limits[:, 0] + hard_limits[:, 1]) / 2
        range_ = hard_limits[:, 1] - hard_limits[:, 0]
        soft_limit_min = midpoint - 0.5 * range_ * self.soft_dof_pos_limit
        soft_limit_max = midpoint + 0.5 * range_ * self.soft_dof_pos_limit
        
        out_of_limits_lower = np.maximum(0, soft_limit_min - self.dof_pos)
        out_of_limits_upper = np.maximum(0, self.dof_pos - soft_limit_max)
        
        return np.sum(out_of_limits_lower + out_of_limits_upper)


    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # Need self.dof_vel_limits (hard limits * soft factor)
        # Read hard limits from model
        hard_vel_limits = self.model.actuator_velocity # Max velocity (assuming symmetric +/-)
        soft_vel_limit_val = hard_vel_limits * self.soft_dof_vel_limit

        # clip to max error = 1 rad/s per joint to avoid huge penalties
        # excess_vel = (np.abs(self.dof_vel) - soft_vel_limit_val).clip(min=0., max=1.) -> numpy
        excess_vel = np.maximum(0, np.abs(self.dof_vel) - soft_vel_limit_val)
        excess_vel = np.minimum(excess_vel, 1.0) # Clip max error contribution per joint
        return np.sum(excess_vel)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        # Need self.torque_limits (hard limits * soft factor)
        # Read hard limits from model
        hard_torque_limits = self.model.actuator_ctrlrange[:, 1] # Assuming symmetric +/- limit is at index 1
        soft_torque_limit_val = hard_torque_limits * self.soft_torque_limit

        # excess_torque = (np.abs(self.torques) - soft_torque_limit_val).clip(min=0.) -> numpy
        excess_torque = np.maximum(0, np.abs(self.torques) - soft_torque_limit_val)
        return np.sum(excess_torque)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # Use self.commands and self.base_lin_vel
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]))
        return np.exp(-lin_vel_error / self.tracking_sigma) # Use tracking_sigma from config

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        # Use self.commands and self.base_ang_vel
        ang_vel_error = np.square(self.commands[2] - self.base_ang_vel[2]) # Index 2 for yaw
        return np.exp(-ang_vel_error / self.tracking_sigma) # Use tracking_sigma from config

    def _reward_feet_air_time(self):
        # Reward long steps
        
        # --- REMOVE OLD LOGIC ---
        # contacts, contact_geom_ids = self._get_contact_info()
        # contact_threshold = 1.0
        # current_contacts = np.zeros(len(self.foot_geom_indices), dtype=bool) # <-- This was shape (0,)
        # for i, geom_id in enumerate(self.foot_geom_indices):
        #      if geom_id in contacts and np.abs(contacts[geom_id][2]) > contact_threshold:
        #          current_contacts[i] = True
        # --- END REMOVE ---

        # --- USE NEW SENSOR-BASED STATE ---
        # self.current_foot_contacts was set in _get_obs()
        current_contacts = self.current_foot_contacts # This has shape (4,)
        # ----------------------------------
                 
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # legged_gym does OR with last_contacts. Let's replicate.
        contact_filt = np.logical_or(current_contacts, self.last_contacts) # This will now work (4,) or (4,)
        
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        
        # reward only on first contact with the ground, scaled by air time
        # legged_gym uses (self.feet_air_time - 0.5)
        rew_airTime = np.sum((self.feet_air_time - 0.5) * first_contact)
        
        # No reward for zero command
        command_norm = np.linalg.norm(self.commands[:2])
        if command_norm < 0.1:
            rew_airTime = 0.0
            
        # Reset air time for feet currently in contact (using filtered contact)
        self.feet_air_time[contact_filt] = 0.0
        
        # Update last contacts
        self.last_contacts = current_contacts
        
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        contacts, contact_geom_ids = self._get_contact_info()
        is_stumbling = False
        stumble_threshold_multiplier = 5.0 # From legged_gym

        for i, geom_id in enumerate(self.foot_geom_indices):
            if geom_id in contacts:
                force_vector = contacts[geom_id]
                horizontal_force_norm = np.linalg.norm(force_vector[:2])
                vertical_force_abs = np.abs(force_vector[2])
                # Check if horizontal force is significantly larger than vertical force
                if horizontal_force_norm > stumble_threshold_multiplier * vertical_force_abs:
                     is_stumbling = True
                     break # Only need one foot to stumble

        return float(is_stumbling) # Return 1.0 if stumbling, 0.0 otherwise (penalty comes from negative scale)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        command_norm = np.linalg.norm(self.commands[:2])
        if command_norm < 0.1:
            # Calculate deviation from default pose
            dof_pos_deviation = np.sum(np.abs(self.dof_pos - self.default_dof_pos))
            return dof_pos_deviation
        else:
            return 0.0 # No penalty if command is not near zero

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        contacts, contact_geom_ids = self._get_contact_info()
        total_excess_force = 0.0

        for i, geom_id in enumerate(self.foot_geom_indices):
            if geom_id in contacts:
                 force_norm = np.linalg.norm(contacts[geom_id])
                 # excess_force = (force_norm - self.max_contact_force).clip(min=0.) -> numpy
                 excess_force = np.maximum(0, force_norm - self.max_contact_force)
                 total_excess_force += excess_force
                 
        return total_excess_force
    def _reward_living_bonus(self):
        # A constant reward for every step the agent is alive
        return 1.0

    def _quat_to_rpy(self, q):
         # MuJoCo sensors ('imu_quat') are scalar-first (w, x, y, z)
         # OLD: qx, qy, qz, qw = q[0], q[1], q[2], q[3]
         qw, qx, qy, qz = q[0], q[1], q[2], q[3] # NEW: Correct scalar-first order
         
         # Roll (x-axis rotation)
         sinr_cosp = 2 * (qw * qx + qy * qz)
         cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
         roll = np.arctan2(sinr_cosp, cosr_cosp)
         
         # Pitch (y-axis rotation)
         sinp = 2 * (qw * qy - qz * qx)
         if abs(sinp) >= 1:
             pitch = np.copysign(np.pi / 2, sinp) # Use 90 degrees if out of range
         else:
             pitch = np.arcsin(sinp)
             
         # Yaw (z-axis rotation)
         siny_cosp = 2 * (qw * qz + qx * qy)
         cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
         yaw = np.arctan2(siny_cosp, cosy_cosp)
         
         return roll, pitch, yaw

    # ==========================================================================
    # 5. ADD RESET, RENDER, AND CLOSE METHODS
    # ==========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for Gymnasium compatibility
        
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        # ... (your existing sim reset code) ...
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        # --- CPG RESET ---
        self.cpg_network.reset() # This sets the initial phases
        
        # --- GET INITIAL CPG STATE ---
        # (This requires a get_state() method in your HopfOscillator class)
        try:
            self.cpg_states = np.concatenate(
                [osc.get_state() for osc in self.cpg_network.oscillators]
            )
        except AttributeError:
            print("ERROR: Your HopfOscillator class is missing a 'get_state()' method.")
            print("Please add: def get_state(self): return np.array([self.x, self.y])")
            raise
        # -------------------------------

        # --- Reset buffers ---
        self.last_actions = np.zeros(self.action_space.shape) # Now 8-dim
        self.prev_last_actions = np.zeros(self.action_space.shape) # Add this
        # ... (rest of your buffer resets) ...
        mujoco.mj_forward(self.model, self.data) # Update kinematics after setting state
        # -------------------------------

        # --- Reset buffers ---
        # ... (rest of your reset function) ...

        self._resample_commands()

        self.step_counter = 0

        observation = self._get_obs()
        info = {} # Add any necessary reset info

        return observation, info

    def render(self):
        """ Syncs the passive viewer. """
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        """ Closes the viewer. """
        if self.viewer:
            self.viewer.close()
    
    def _resample_commands(self):
        """
        Randomly select commands.
        - 60% chance for a 'standalone' command (stand, forward, sideways, rotate)
        - 40% chance for a 'mixture' command (a combination of 2)
        """
        
        # --- Define Command Ranges (from legged_robot_config.py) ---
        lin_vel_x_range = [-2.0, 2.0] # 2x forward speed
        lin_vel_y_range = [-0.5, 0.5] # Keep sideways speed low (it's harder)
        ang_vel_yaw_range = [-1.5, 1.5] # Increase rotation speed
        # -----------------------------------------------------------

        # --- Zero out all commands first ---
        self.commands[:] = 0.0
        
        # --- Define the modes ---
        standalone_modes = ['stand', 'forward', 'sideways', 'rotate']
        # 'stand' is not a moving mode, so we don't mix it
        moving_modes = ['forward', 'sideways', 'rotate'] 

        # --- Decide between Standalone (60%) or Mixture (40%) ---
        if self.np_random.random() < 0.6:
            # --- STANDALONE (60%) ---
            # Pick one of the 4 standalone modes with equal probability
            chosen_mode = self.np_random.choice(standalone_modes)
            
            if chosen_mode == 'stand':
                # Leave commands at [0, 0, 0]
                # This will activate your _reward_stand_still
                pass
            
            elif chosen_mode == 'forward':
                self.commands[0] = self._sample_value(lin_vel_x_range[0], lin_vel_x_range[1])

            elif chosen_mode == 'sideways':
                self.commands[1] = self._sample_value(lin_vel_y_range[0], lin_vel_y_range[1])

            elif chosen_mode == 'rotate':
                self.commands[2] = self._sample_value(ang_vel_yaw_range[0], ang_vel_yaw_range[1])

        else:
            # --- MIXTURE (40%) ---
            # Pick 2 different moving modes
            chosen_modes = self.np_random.choice(moving_modes, size=2, replace=False)
            
            for mode in chosen_modes:
                if mode == 'forward':
                    self.commands[0] = self._sample_value(lin_vel_x_range[0], lin_vel_x_range[1])
                
                elif mode == 'sideways':
                    self.commands[1] = self._sample_value(lin_vel_y_range[0], lin_vel_y_range[1])
                
                elif mode == 'rotate':
                    self.commands[2] = self._sample_value(ang_vel_yaw_range[0], ang_vel_yaw_range[1])


# ==============================================================================
# 6. ADD A MAIN BLOCK TO RUN THE SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- !! IMPORTANT !! ---
    # --- SET YOUR MODEL PATH HERE ---
    XML_MODEL_PATH = "../../unitree_mujoco/unitree_robots/go2/scene_ground.xml" # <--- UPDATE THIS
    # -----------------------

    print("Launching CPG-controlled quadruped in MuJoCo...")
    
    try:
        env = UnitreeEnv(model_path=XML_MODEL_PATH, 
                         render_mode="human", 
                         frame_skip=1) 
        
        obs, info = env.reset(seed=42)
        dummy_action = np.zeros(env.action_space.shape)
        
        # --- NEW: FPS Counter Setup ---
        print("\nStarting simulation loop. Press Ctrl+C to stop.")
        start_time = time.time()
        num_steps = 0
        print_interval = 2000 # Print FPS every 2000 steps
        # -----------------------------

        for i in range(100000): # Run for more steps
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
            if terminated or truncated:
                obs, info = env.reset()
            # --- NEW: FPS Calculation ---
            num_steps += 1
            if (i + 1) % print_interval == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = num_steps / elapsed_time
                
                print(f"\n--- FPS Report (Step {i+1}) ---")
                print(f"    Time: {elapsed_time:.2f} s")
                print(f"    Steps: {num_steps}")
                print(f"    FPS: {fps:.2f} steps/sec")
                print("------------------------------")
                
                # Reset for next batch
                start_time = time.time()
                num_steps = 0
            # --- END NEW ---
                
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps.")
                obs, info = env.reset()
                
    except KeyboardInterrupt: # Added this to catch Ctrl+C
        print("\nTraining interrupted by user.")
    #except FileNotFoundError:
    #   ... (rest of your file) ...
    