# --- START OF FILE cpg_controller.py ---

import numpy as np

class CPGController:
    """
    Manages a set of coupled oscillators to generate rhythmic leg movements
    for a quadruped robot, based on a trotting gait.
    """
    def __init__(self, dt):
        self.dt = dt
        
        # --- CPG Parameters ---
        # Phase offsets for a trotting gait:
        # FL and RR legs are in phase (0).
        # FR and RL legs are in phase (pi).
        self.phase_offsets = np.array([0, np.pi, np.pi, 0])
        
        # Current phase for each of the 4 legs
        self.phases = np.array([0.0, np.pi, np.pi, 0.0])
        
        # --- RL Action -> CPG Command Mapping ---
        # These are the base values for the CPG, which will be modulated by the RL agent
        self.base_frequency = 2.0  # Hz
        self.base_amplitude_swing = 0.2  # radians
        self.base_amplitude_lift = 0.2  # radians
        self.base_body_height = 0.3 # from your env
        
        # Neutral joint positions for the "standing" pose
        # (abduction, hip, knee) for FL, FR, RL, RR
        self.neutral_pose = np.array([
             0.0, 0.8, -1.6,  # FL
            -0.0, 0.8, -1.6,  # FR
             0.0, 0.8, -1.6,  # RL
            -0.0, 0.8, -1.6,  # RR
        ])

    def update(self, rl_action):
        """
        Updates the CPG phases and computes the target joint angles.
        
        Args:
            rl_action (np.ndarray): A 4-element array from the RL agent containing
                                    modulations for [freq, swing_amp, lift_amp, turn_rate].
        
        Returns:
            np.ndarray: A 12-element array of target joint positions.
        """
        # --- 1. Interpret RL Action ---
        # Scale the agent's [-1, 1] actions to meaningful CPG modulations
        freq_mod = 1.0 + 0.5 * rl_action[0]  # Scale frequency by +/- 50%
        swing_amp_mod = 1.0 + 0.5 * rl_action[1] # Scale swing amplitude by +/- 50%
        lift_amp_mod = 1.0 + 0.5 * rl_action[2] # Scale lift amplitude by +/- 50%
        turn_rate = rl_action[3] # Use this for turning, range [-1, 1]

        # --- 2. Update Oscillator Phases ---
        frequency = self.base_frequency * freq_mod
        self.phases += 2 * np.pi * frequency * self.dt
        self.phases %= (2 * np.pi) # Keep phase within [0, 2*pi]

        # --- 3. Calculate Joint Targets ---
        # Calculate the vertical (lift) and horizontal (swing) components
        vertical_offset = self.base_amplitude_lift * lift_amp_mod * np.sin(self.phases)
        horizontal_offset = self.base_amplitude_swing * swing_amp_mod * np.cos(self.phases)

        target_joint_pos = self.neutral_pose.copy()
        
        # Apply offsets to hip and knee joints for each leg
        # Hip joints control swing (forward/backward motion)
        # Knee joints control lift (up/down motion)
        
        # Front Left Leg (index 1 for hip, 2 for knee)
        target_joint_pos[1] += horizontal_offset[0]
        target_joint_pos[2] += vertical_offset[0]

        # Front Right Leg (index 4 for hip, 5 for knee)
        target_joint_pos[4] += horizontal_offset[1]
        target_joint_pos[5] += vertical_offset[1]

        # Rear Left Leg (index 7 for hip, 8 for knee)
        target_joint_pos[7] += horizontal_offset[2]
        target_joint_pos[8] += vertical_offset[2]
        
        # Rear Right Leg (index 10 for hip, 11 for knee)
        target_joint_pos[10] += horizontal_offset[3]
        target_joint_pos[11] += vertical_offset[3]

        # --- 4. Apply Turning ---
        # Add a differential to the swing of the front/rear legs to induce a turn
        # When turning right (turn_rate > 0), left legs swing more, right legs swing less.
        turn_swing_adjustment = 0.15 * turn_rate 
        target_joint_pos[[1, 7]] += turn_swing_adjustment  # Left legs (FL, RL)
        target_joint_pos[[4, 10]] -= turn_swing_adjustment # Right legs (FR, RR)
        
        return target_joint_pos

    def reset(self):
        """ Resets the CPG phases to their initial state. """
        self.phases = self.phase_offsets.copy()