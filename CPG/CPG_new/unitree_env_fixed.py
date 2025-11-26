import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer # <-- ADD THIS
# Add torch if you want rewards calculated with it, otherwise use numpy
import torch 
import time
from CPG_Network_Enhanced import EnhancedCPGNetwork

class UnitreeEnv(gym.Env):
    def __init__(self, model_path, render_mode=None, frame_skip=4):
        super().__init__()
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # --- Identify Actuated Joints Correctly ---
        self.nu = self.model.nu
        # Get joint IDs for actuators (trnid stores [joint_id, type])
        self.actuator_joint_ids = self.model.actuator_trnid[:, 0].astype(int)
        self.actuator_joint_ranges = self.model.jnt_range[self.actuator_joint_ids]

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
        # Standard Go2 Standing Pose
        self.default_dof_pos = np.array([
            0.1, 0.8, -1.8,   # FL (Knee changed from -1.5 to -1.8)
            -0.1, 0.8, -1.8,  # FR
            0.1, 1.0, -1.8,   # RL
            -0.1, 1.0, -1.8   # RR
        ])
        self.init_qpos[7:] = self.default_dof_pos

        self.init_qpos[2] = 0.25
        # Control Parameters
        self.p_gains = np.full(self.nu, 100.0)
        self.d_gains = np.full(self.nu, 2.0)
        self.dt = self.model.opt.timestep * self.frame_skip

        # CPG
        self.cpg_network = EnhancedCPGNetwork(self.dt, self.default_dof_pos)

        # Observation Scales
        self.obs_scales = {
            "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05
        }

        # Action/Obs Spaces
        self.action_space = gym.spaces.Box(-1, 1, shape=(12,), dtype=np.float32)
        # 3 (lin) + 3 (ang) + 3 (grav) + 3 (cmd) + 12 (pos) + 12 (vel) + 12 (last_act) + 4 (contact) + 8 (cpg)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(60,), dtype=np.float32)

        # State Buffers
        self.commands = np.zeros(3)
        self.last_actions = np.zeros(12)
        self.cpg_states = np.zeros(8)
        self.last_dof_vel = np.zeros(12)
        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.zeros(4, dtype=bool)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)

        # Rewards
        self.reward_scales = {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "torques": -0.00002,
            "dof_acc": -2.5e-7,
            "action_rate": -0.1,
            "collision": -1.0,
            "dof_pos_limits": -10.0,
            "stand_still": -0.5,
            "feet_air_time": 0.5
        }

        # Mappings
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        self.feet_site_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.feet_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.feet_site_names]

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    

    def step(self, action):
        action = np.clip(action, -1, 1)
        
        amp_indices = [0, 3, 6, 9]
        cpg_action = action.copy()
        # Apply Mapping: [-1, 1] -> [0.0, 1.0]
        # Agent Output 0.0 (Idle) -> CPG Input 0.5 (Stepping)
        cpg_action[amp_indices] = (cpg_action[amp_indices] + 1.0) * 0.5


        # 1. Step CPG
        target_dof_pos, self.cpg_states = self.cpg_network.step(cpg_action)
        


        # 2. PD Control
        #self.dof_pos = self.data.qpos[7:]
        #self.dof_vel = self.data.qvel[6:]
        
        #torques = self.p_gains * (target_dof_pos - self.dof_pos) + \
        #          self.d_gains * (0 - self.dof_vel) # Damping term targets 0 velocity relative to CPG
        
        #self.data.ctrl[:] = np.clip(torques, -25, 25) # Clip to torque limits
        
        # 3. Physics Step
        for _ in range(self.frame_skip):
            # A. Get CURRENT state (updates every millisecond)
            current_pos = self.data.qpos[7:]
            current_vel = self.data.qvel[6:]
            
            # B. Calculate Torques based on FRESH state
            # This allows the 'D' term to dampen impacts immediately
            torques = self.p_gains * (target_dof_pos - current_pos) + \
                      self.d_gains * (0 - current_vel)
            
            # C. Apply and Step
            self.data.ctrl[:] = np.clip(torques, -25, 25)
            mujoco.mj_step(self.model, self.data)
            
            # Optional: Update viewer every substep if you want super smooth video, 
            # but usually we sync only once per control step to save speed.
        #mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        
        # 4. Update Visuals
        if self.render_mode == "human" and self.viewer.is_running():
            self.viewer.sync()

        # 5. Get Obs & Compute Reward
        self._update_state()
        obs = self._get_obs()
        
        terminated = self._check_termination()
        truncated = self.step_counter >= 1000
        reward = self._compute_reward(action)
        
        # 6. Update History
        self.last_actions = action.copy()
        self.last_dof_vel = self.dof_vel.copy()
        self.step_counter += 1

        return obs, reward, terminated, truncated, {}

    def _update_state(self):
        # Calculate Base Velocity in Body Frame
        q = self.data.sensor('imu_quat').data
        # Rotate world velocity into body frame
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, q)
        rot_mat = rot_mat.reshape(3, 3)
        
        vel_world = self.data.qvel[:3]
        self.base_lin_vel = rot_mat.T @ vel_world
        self.base_ang_vel = self.data.sensor('imu_gyro').data
        
        # Projected Gravity (Body Frame)
        self.projected_gravity = rot_mat.T @ np.array([0, 0, -1])

        # Foot Contacts (Simple force thresholding)
        # Note: Real Unitree go2 xmls often use site sensors for feet
        self.current_contacts = np.zeros(4)
        forces = [
            self.data.sensor('FL_foot_force').data[2],
            self.data.sensor('FR_foot_force').data[2],
            self.data.sensor('RL_foot_force').data[2],
            self.data.sensor('RR_foot_force').data[2]
        ]
        # 2. Hysteresis Settings
        contact_on_threshold = 6.0  # Force needed to ENTER stance
        contact_off_threshold = 3.0 # Force needed to EXIT stance (lift off)
        
        for i in range(4):
            f = np.abs(forces[i])
            
            # Logic:
            # If force is HUGE -> Contact is True
            # If force is TINY -> Contact is False
            # If force is MEDIUM -> Keep previous value (don't flicker)
            
            if f > contact_on_threshold:
                self.current_contacts[i] = 1.0
            elif f < contact_off_threshold:
                self.current_contacts[i] = 0.0
            else:
                # Keep the existing state (from the previous step)
                # We don't change anything
                pass


    def _get_obs(self):
        return np.concatenate([
            self.base_lin_vel * self.obs_scales['lin_vel'],
            self.base_ang_vel * self.obs_scales['ang_vel'],
            self.projected_gravity,
            self.commands * self.obs_scales['lin_vel'], # Scale commands same as vel
            (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
            self.dof_vel * self.obs_scales['dof_vel'],
            self.last_actions,
            self.current_contacts,
            self.cpg_states
        ]).astype(np.float32)

    def _compute_reward(self, action):
        dt = 0.01
        # 1. Linear velocity tracking (x, y)
        # Note: We removed the scalar multipliers (0.75, 0.5) here to treat them as probabilities
        r_lin_x = np.exp(-((self.commands[0] - self.base_lin_vel[0])**2) / 0.25)
        r_lin_y = np.exp(-((self.commands[1] - self.base_lin_vel[1])**2) / 0.25)
        r_ang_z = np.exp(-((self.commands[2] - self.base_ang_vel[2])**2) / 0.25)
        
        # MULTIPLICATIVE REWARD
        # If the robot doesn't track X, it gets NOTHING for tracking Y or Z.
        # This forces it to move to unlock the other points.
        reward_tracking = (r_lin_x * r_lin_y * r_ang_z) * 2.0  # Scale up total
        # 3. Linear velocity penalty (z)
        lin_z_penalty = - (self.base_lin_vel[2] ** 2) * 2 
        # 4. Angular velocity penalty (roll, pitch)
        ang_xy_penalty = - (np.sum(self.base_ang_vel[:2] ** 2)) * 0.05 
        # 5. Work penalty
        work = - np.abs(np.dot(self.data.actuator_force, self.dof_vel - self.last_dof_vel)) * 0.001 
        # Sum all terms
        reward = reward_tracking + lin_z_penalty + ang_xy_penalty + work
        return reward

    

    def _check_termination(self):
        # Terminate if falling
        if self.projected_gravity[2] > -0.5: # Tilted more than ~60 degrees
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        
        # Reset Physics
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)
        
        # Reset CPG & Buffers
        self.cpg_network.reset()
        self.last_actions[:] = 0
        self.last_dof_vel[:] = 0
        self.feet_air_time[:] = 0
        
        # Sample Command
        self._resample_commands()
        
        self._update_state()
        return self._get_obs(), {}

    def _resample_commands(self):
        """
        Resamples commands with NO mixing (standalone only).
        Reduces the chance of standing still to 10%.
        """
        # --- 1. Define Command Ranges (Slightly Reduced for Stability) ---
        # Format: (min_speed, max_speed) magnitude
        
        # Forward: [-0.5 to 1.5] approx
        lin_vel_x_range = [-0.5, 1.5] 
        
        # Sideways: [-0.4 to 0.4]
        lin_vel_y_range = [-0.4, 0.4]
        
        # Rotation: [-1.0 to 1.0]
        ang_vel_yaw_range = [-0.5, 0.5]
        
        # Minimum magnitude to ensure robot actually tries to move
        deadband = 0.01 
        # -----------------------------------------------------------

        # --- 2. Zero out all commands first ---
        self.commands[:] = 0.0
        
        # --- 3. Define Probabilities ---
        # 10% Stand, 40% Forward, 25% Side, 25% Rotate
        modes = ['stand', 'forward', 'sideways', 'rotate']
        probs = [0.05,    0.45,      0.25,       0.25]

        # --- 4. Select Mode ---
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        chosen_mode = rng.choice(modes, p=probs)

        if chosen_mode == 'stand':
            # Commands remain [0, 0, 0]
            pass

        elif chosen_mode == 'forward':
            # Sample uniform
            val = rng.uniform(*lin_vel_x_range)
            # Apply Deadband: If value is too close to 0, push it out
            if abs(val) < deadband:
                val = deadband if val > 0 else -deadband
            self.commands[0] = val

        elif chosen_mode == 'sideways':
            val = rng.uniform(*lin_vel_y_range)
            if abs(val) < deadband:
                val = deadband if val > 0 else -deadband
            self.commands[1] = val

        elif chosen_mode == 'rotate':
            val = rng.uniform(*ang_vel_yaw_range)
            if abs(val) < deadband:
                val = deadband if val > 0 else -deadband
            self.commands[2] = val

    def close(self):
        if self.viewer:
            self.viewer.close()


# ==============================================================================
# 6. ADD A MAIN BLOCK TO RUN THE SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- !! IMPORTANT !! ---
    # --- SET YOUR MODEL PATH HERE ---
    XML_MODEL_PATH = "../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml" # <--- UPDATE THIS
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
    