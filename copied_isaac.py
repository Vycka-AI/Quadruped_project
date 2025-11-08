import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer # <-- ADD THIS
# Add torch if you want rewards calculated with it, otherwise use numpy
import torch 
import time
# Or use numpy directly:
# import numpy as torch # Alias numpy to torch for easier copy-pasting

# Assuming you have your UnitreeEnv class inheriting from gym.Env or similar

class UnitreeEnv(gym.Env): # Or your specific base class
    def __init__(self, model_path, render_mode=None, frame_skip=1, **kwargs):
        super().__init__()
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.render_mode == "human":
            # Launch the passive viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # --- Store initial state for resets and pose penalty ---
        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 0.325
        self.init_qvel = self.data.qvel.copy()

        # --- Define Constants (Adapt these from legged_gym configs) ---
        self.action_scale = 0.05 # From GO2RoughCfg.control
        # Example default pose (Adapt from GO2RoughCfg.init_state.default_joint_angles)
        # Ensure the order matches your MuJoCo joint order!
        
        self.default_dof_pos = np.array([
            -0.1,  0.8, -1.5,  # FR_hip_joint, FR_thigh_joint, FR_calf_joint
            0.1,  0.8, -1.5,  # FL_hip_joint, FL_thigh_joint, FL_calf_joint
            -0.1,  1.0, -1.5,  # RR_hip_joint, RR_thigh_joint, RR_calf_joint
            0.1,  1.0, -1.5   # RL_hip_joint, RL_thigh_joint, RL_calf_joint
        ])

        self.init_qpos[7:] = self.default_dof_pos



        # --- SET YOUR SAFETY MARGIN (AS A PERCENTAGE) ---
        # 5% (0.05) is a good start. This means a joint with 10 radians of
        # total motion will get a 0.5 rad offset on each end.
        self.joint_limit_margin_percentage = 0.3

        # --- 1. GET HARD (PHYSICAL) LIMITS ---
        # Get limits from the MuJoCo model
        # Slice from [1:13] to skip the base joint (at index 0)
        hard_limits = self.model.jnt_range[1:13].copy()
        hard_min = hard_limits[:, 0]
        hard_max = hard_limits[:, 1]
        
        # --- 2. CALCULATE DYNAMIC OFFSET (THE NEW LOGIC) ---
        # Calculate total amplitude (range) of each joint
        total_amplitude = hard_max - hard_min
        # Calculate the offset for each joint (e.g., 5% of its total amplitude)
        dynamic_offset = total_amplitude * self.joint_limit_margin_percentage

        # --- 3. APPLY OFFSET TO CREATE SOFT (SAFE) LIMITS ---
        self.soft_jnt_min = hard_min + dynamic_offset
        self.soft_jnt_max = hard_max - dynamic_offset

        # --- 4. PRE-CALCULATE SCALES (MORE EFFICIENT) ---
        # (self.default_dof_pos is defined in __init__)
        self.scale_to_max = self.soft_jnt_max - self.default_dof_pos
        self.scale_to_min = self.default_dof_pos - self.soft_jnt_min

        # --- 5. SANITY CHECK (UPDATED) ---
        if np.any(self.scale_to_max < 0) or np.any(self.scale_to_min < 0):
            print("\n" + "="*80)
            print("ERROR: YOUR 'default_dof_pos' IS OUTSIDE YOUR NEW 'soft_limits'.")
            print("This means your 'joint_limit_margin_percentage' is too large or 'default_dof_pos' is wrong.")
            print("\n--- DEBUG INFO ---")
            
            print(f"\nJoint Limit Margin Percentage: {self.joint_limit_margin_percentage}")
            
            # Use np.round for cleaner output
            print("\nTotal Joint Amplitude (hard_max - hard_min):")
            print(np.round(total_amplitude, 3))

            print("\nDynamic Offset (amplitude * percentage):")
            print(np.round(dynamic_offset, 3))

            print("\nSoft Min Limits (hard_min + offset):")
            print(np.round(self.soft_jnt_min, 3))
            
            print("\nDefault DOF Position:")
            print(np.round(self.default_dof_pos, 3))
            
            print("\nSoft Max Limits (hard_max - offset):")
            print(np.round(self.soft_jnt_max, 3))
            
            print("\n--- CALCULATED SCALES (MUST ALL BE >= 0) ---")
            print("\nScale to Max (soft_max - default):")
            print(np.round(self.scale_to_max, 3))
            
            print("\nScale to Min (default - soft_min):")
            print(np.round(self.scale_to_min, 3))

            print("\n" + "="*80)




        # Example PD Gains (Adapt from GO2RoughCfg.control)
        # Assuming stiffness=20, damping=0.5 for all joints
        self.p_gains = np.full(self.model.nu, 25.0)
        self.d_gains = np.full(self.model.nu, 0.5)


        self.obs_scales_lin_vel = 2.0
        self.obs_scales_ang_vel = 0.25     # <--- SCALE IT
        self.obs_scales_dof_pos = 1.0 # <--- SCALE IT
        self.obs_scales_dof_vel = 0.05           # <--- SCALE IT


        self.step_counter = 0
        self.max_episode_length = 6000 # Example: 1000 steps


        self.termination_geom_indices = []
        self.penalised_geom_indices = []

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        
        # Get IDs for penalized bodies (from GO2RoughCfg.asset.penalize_contacts_on)
        penalized_body_names = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", 
                                "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]
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

        print(f"Found {len(self.termination_geom_indices)} termination geoms (base).")
        print(f"Found {len(self.penalised_geom_indices)} penalized geoms (thighs/calves).")


        # Reward Scales (Adapt from LeggedRobotCfg.rewards.scales and GO2RoughCfg.rewards.scales)
        # Multiply by dt (assuming dt = frame_skip * model.opt.timestep)
        self.dt = frame_skip * self.model.opt.timestep # Example calculation
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
        # Filter out zero scales for efficiency
        self.active_reward_scales = {k: v for k, v in self.reward_scales.items() if v != 0.0}

        # Reward Function Configuration (Adapt from LeggedRobotCfg.rewards and GO2RoughCfg.rewards)
        self.tracking_sigma = 0.25
        self.base_height_target = 0.25 # From GO2RoughCfg
        # You'll need soft limit factors if using dof_pos/vel/torque limit rewards
        self.soft_dof_pos_limit = 0.9 # From GO2RoughCfg
        self.soft_dof_vel_limit = 1.0 # From LeggedRobotCfg
        self.soft_torque_limit = 1.0 # From LeggedRobotCfg
        # You'll need max contact force if using feet_contact_forces reward
        self.max_contact_force = 100.0 # From LeggedRobotCfg

        # --- Indices (CRITICAL: Find these in your MuJoCo model) ---
        self.feet_indices = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'FL_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'FR_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'RL_foot'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'RR_foot')
        ])
        # Find geom indices associated with feet bodies for contact checking
        
        # --- Observation/Action Space Definition ---
        # Original obs size = 48
        # New obs size = 48 + 4 (foot contacts) = 52
        obs_high = np.inf * np.ones(52)
        obs_low = -obs_high
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        action_high = np.ones(self.model.nu) # Action is offset, typically [-1, 1]
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        # --- Buffers for Rewards and State History ---
        self.commands = np.zeros(3) # x_vel, y_vel, yaw_vel_rate command
        # You might need observation scaling factors if using them
        # self.obs_scales_lin_vel = 2.0 # From LeggedRobotCfg.normalization.obs_scales
        # self.obs_scales_ang_vel = 0.25 # etc.
        self.last_actions = np.zeros(self.action_space.shape)
        self.last_dof_vel = np.zeros(self.model.nu)
        self.feet_air_time = np.zeros(len(self.feet_indices))
        self.last_contacts = np.zeros(len(self.feet_indices), dtype=bool)
        
        # Buffers for PD control (optional, could be calculated in step)
        self.dof_pos = None
        self.dof_vel = None
        self.torques = np.zeros(self.model.nu)

        # Buffers for state needed by rewards/observations
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.zeros(3)

        self.step_counter = 0

        # TODO: Add limits if needed for reward functions
        # self.dof_pos_limits = ... # Read from model or set manually
        # self.dof_vel_limits = ...
        # self.torque_limits = ... # Control limits from model

        # ... (rest of your __init__)

    def _map_actions_to_targets(self, action):
        """
        Maps the normalized action output [-1, 1] to the safe
        joint position range [soft_min, soft_max], relative to the
        default standing pose.
        
        Scales are pre-calculated in __init__ for efficiency.
        """
        
        # 1. Separate positive and negative actions
        positive_actions = np.clip(action, 0.0, 1.0) # Action's positive 'nudge' [0, 1]
        negative_actions = np.clip(action, -1.0, 0.0) # Action's negative 'nudge' [-1, 0]

        # 2. Apply pre-calculated scales:
        target_dof_pos = (self.default_dof_pos + 
                          positive_actions * self.scale_to_max + 
                          negative_actions * self.scale_to_min)
        
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
    def _get_obs(self):
        """ Computes and returns the observation vector. """
        # --- Base velocity and orientation ---
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

        # --- Foot Contacts ---
        # --- REVERT TO USING FOOT FORCE SENSORS ---
        contact_threshold = 1.0 # Adjust if needed

        # Get force sensor readings (assuming they output 3D force or similar)
        fl_force = self.data.sensor('FL_foot_force').data # Shape depends on sensor definition
        fr_force = self.data.sensor('FR_foot_force').data
        rl_force = self.data.sensor('RL_foot_force').data
        rr_force = self.data.sensor('RR_foot_force').data

        # Check the vertical component (usually index 2) against the threshold
        # Adapt index if your sensor output is different (e.g., just magnitude)
        fl_contact = float(np.abs(fl_force[2]) > contact_threshold) if len(fl_force) >=3 else float(np.abs(fl_force) > contact_threshold)
        fr_contact = float(np.abs(fr_force[2]) > contact_threshold) if len(fr_force) >=3 else float(np.abs(fr_force) > contact_threshold)
        rl_contact = float(np.abs(rl_force[2]) > contact_threshold) if len(rl_force) >=3 else float(np.abs(rl_force) > contact_threshold)
        rr_contact = float(np.abs(rr_force[2]) > contact_threshold) if len(rr_force) >=3 else float(np.abs(rr_force) > contact_threshold)

        foot_contacts_float = np.array([fl_contact, fr_contact, rl_contact, rr_contact], dtype=np.float32) # Shape (4,)
        self.current_foot_contacts = foot_contacts_float # Store for reward functions
        # -----------------------------------------------

        # --- Assemble Observation Buffer ---
        obs = np.concatenate((
            self.base_lin_vel * self.obs_scales_lin_vel,       # <--- SCALE IT
            self.base_ang_vel * self.obs_scales_ang_vel,     # <--- SCALE IT
            self.projected_gravity,                           # (Already normalized)
            self.commands,                                    # (Already normalized)
            (self.dof_pos - self.default_dof_pos) * self.obs_scales_dof_pos, # <--- SCALE IT
            self.dof_vel * self.obs_scales_dof_vel,           # <--- SCALE IT
            self.last_actions,
            foot_contacts_float 
        )).astype(np.float32)

        return obs

    def _sample_value(self, min_val, max_val, dead_zone=0.2):
        """
        Samples a value from a range, ensuring it's outside a 'dead zone'
        around zero (unless it's exactly zero).
        """
        # Sample a value
        val = self.np_random.uniform(min_val, max_val)
        
        # Apply the dead zone
        if 0 < abs(val) < dead_zone:
            # If it's in the dead zone (but not zero), force it to the minimum value
            val = dead_zone * np.sign(val)
            
        return val

    def step(self, action):
        """ Applies action, simulates, calculates rewards, and returns results. """
        # --- Apply PD Control Action ---
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # --- STORE CURRENT ACTION ---
        self._current_action_for_reward = clipped_action # Store for reward calc
        # ---------------------------

        #clipped_action = np.zeros_like(action) # Use a safe action

        target_dof_pos = self._map_actions_to_targets(clipped_action)
        
        # PD Controller: Calculate torques
        current_dof_pos = self.data.qpos[7:]
        current_dof_vel = self.data.qvel[6:]
        
        position_error = (target_dof_pos - current_dof_pos)
        velocity_error = -current_dof_vel 
        
        self.torques = self.p_gains * position_error + self.d_gains * velocity_error
        
        ctrl_limit = self.model.actuator_ctrlrange[:, 1]
        applied_torques = np.clip(self.torques, -ctrl_limit, ctrl_limit)
        
        # --- ADD THIS DEBUG BLOCK ---
        if self.render_mode == "human" and self.step_counter % 50 == 0: # Print every 50 steps
            print("\n--- PD Controller Debug (Step", self.step_counter, ") ---")
            # Print policy output (Front-Left leg)
            print(f"  Policy Action (FL leg):  {clipped_action[:3]}") 
            # Print target position (Front-Left leg)
            print(f"  Target Pos (FL leg):     {target_dof_pos[:3]}")
            # Print default position (Front-Left leg)
            print(f"  Default Pos (FL leg):    {self.default_dof_pos[:3]}")
            # Print final torque (Front-Left leg)
            print(f"  Applied Torque (FL leg): {applied_torques[:3]}")
        self.step_counter += 1
        # -------------------------------

        resampling_time_steps = int(3.0 / self.dt) 
        if self.step_counter % resampling_time_steps == 0:
            self._resample_commands()


        # --- Simulate ---
        self.data.ctrl[:] = applied_torques
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            #time.sleep(0.05)
        # --- Get Observations ---
        observation = self._get_obs()

        # --- Compute Reward ---
        # _compute_reward will now be able to access self._current_action_for_reward

        # --- Check Termination ---
        terminated, truncated = self._check_termination()

        reward, reward_info = self._compute_reward(terminated, truncated)
        #truncated = False # Add logic for time limits if needed (e.g., self.current_step >= self.max_steps)

        # --- Update State History ---
        current_dof_vel = self.data.qvel[6:].copy() # Get current velocity before updating last_dof_vel
        self.last_actions = clipped_action.copy()
        self.last_dof_vel = current_dof_vel # Update last velocity

        # --- Info dictionary ---
        info = {}
        info.update(reward_info)
        # Add any other relevant info

        # (Optional) Render if needed
        if self.render_mode == "human":
             self.render()

        return observation, reward, terminated, truncated, info

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


        return base_contact or orientation_violated or truncated, truncated# or body_too_low

# --- Helper to convert quat to RPY (adjust based on your quat convention) ---
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

    # --- Implement ALL reward functions from legged_robot.py below ---
    # Convert torch operations to numpy equivalents
    
    # Example conversions:
    # torch.sum(..., dim=1) -> np.sum(..., axis=1)
    # torch.square(...) -> np.square(...)
    # torch.clip(..., min=a, max=b) -> np.clip(..., a_min=a, a_max=b)
    # torch.exp(...) -> np.exp(...)
    # torch.norm(..., dim=1) -> np.linalg.norm(..., axis=1)
    # torch.logical_or(...) -> np.logical_or(...)
    # torch.any(..., dim=1) -> np.any(..., axis=1)
    
    # --- Make sure to use self.base_lin_vel, self.base_ang_vel etc.
    # --- which are updated in _get_obs()
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
        # Use actions (current) and last_actions (previous step)
        return np.sum(np.square(self.last_actions - self._current_action_for_reward)) # NEW LINE

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

    # Add reset logic if needed
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for Gymnasium compatibility

        # --- Reset simulation ---
        mujoco.mj_resetData(self.model, self.data)

        # --- Reset state ---
        # Example: Set to default pose + small noise
        noise_low = -0.02
        noise_high = 0.02
        self.step_counter = 0 # <-- Make sure to reset the counter!
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        #qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        #qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
        #qpos[2] = 0.30  # Manually set the height (e.g., 0.42 meters)
        # --- THIS LINE CAUSES THE ERROR ---
        # self.set_state(qpos, qvel) # REMOVE THIS LINE
        # ---------------------------------

        # --- REPLACE WITH THESE LINES ---
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
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
        """ Renders the environment. """
        #time.sleep(0.2)
        if self.render_mode == "human" and self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception as e:
                # Handle window close gracefully
                if "Window" in str(e) or "glfw" in str(e).lower():
                    print("Viewer window closed.")
                    self.viewer = None # Stop trying to render
                else:
                    raise

    def close(self):
        """ Closes the environment viewer. """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # Add _resample_commands if needed
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


# You will also need to update your training script (`multi_env.py`)
# Ensure the observation space matches the environment's new definition.
# The action space remains the same, but its interpretation changes in the step function.
