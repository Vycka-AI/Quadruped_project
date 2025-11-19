import numpy as np

# ==============================================================================
# 1. Minimal Hopf Oscillator Class
# ==============================================================================
class HopfOscillator:
    def __init__(self, dt, alpha=5.0):
        self.dt = dt
        self.alpha = alpha
        self.x = 1.0
        self.y = 0.0
        self.mu = 1.0
        self.omega = 2 * np.pi * 1.0

    def set_parameters(self, amplitude, frequency):
        self.mu = amplitude**2
        self.omega = 2 * np.pi * frequency

    def step(self):
        r_sq = self.x**2 + self.y**2
        x_dot = self.alpha * (self.mu - r_sq) * self.x - self.omega * self.y
        y_dot = self.alpha * (self.mu - r_sq) * self.y + self.omega * self.x
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt
        return self.x, self.y
    def get_state(self):
        """Returns the current x, y state."""
        return np.array([self.x, self.y])

# ==============================================================================
# 2. CORRECTED CPGNetwork Class
# ==============================================================================
class CPGNetwork:
    def __init__(self, dt, base_positions=None):
        self.dt = dt
        self.oscillators = [HopfOscillator(dt) for _ in range(4)] # FR, FL, RR, RL
        
        self.joint_gains = [
            (1.5,  1.1, -0.8),  # FR leg (Abd, Hip, Knee)
            (1.5,  1.1, -0.8),  # FL leg
            (1.5,  1.1, -0.8),  # RR leg
            (1.5,  1.1, -0.8)   # RL leg
        ]
        #self.joint_gains = [
        #    (1.5,  1.1, -0.8),  # FR leg (Abd, Hip, Knee)
        #    (1.5,  1.1, -0.8),  # FL leg
        #    (1.5,  1.1, -0.8),  # RR leg
        #    (1.5,  1.1, -0.8)   # RL leg
        #]
        self.joint_gains = list(np.array(self.joint_gains)/2.0)
        
        if base_positions is None:
            self.base_positions = np.array([
                0.0, 0.9, -1.8,  # FR
                0.0, 0.9, -1.8,  # FL
                0.0, 0.9, -1.8,  # RR
                0.0, 0.9, -1.8   # RL
            ])
        else:
            self.base_positions = base_positions

    def step(self, rl_action=None):
        """
        Steps the CPG network.
        If rl_action is provided, it uses it to modulate the CPG.
        If rl_action is None, it runs with hard-coded "trot" parameters.
        """
        target_joint_positions = np.zeros(12)
        cpg_states = []

        if rl_action is not None:
            # --- 1. RL-DRIVEN MODE ---
            amplitudes_action = rl_action[0:4]
            frequencies_action = rl_action[4:8]
            
            # --- Amplitude Mapping ---
            base_amp = 1.5  # 0.5 rad
            amp_gain = 1.5  # +/- 0.5 rad
            scaled_amps = base_amp + (amplitudes_action * amp_gain)
            # This gives a range of [0.0, 1.0] rad

            # --- Frequency Mapping (with gain) ---
            base_freq = 3.5 # The "default" trot frequency in Hz
            freq_gain = 3.0  # How much the network can vary it (up or down)
            scaled_freqs = base_freq + (frequencies_action * freq_gain)
            # This gives a range of [1.0, 4.0] Hz
            
            # Ensure frequencies are non-negative (safety clip)
            scaled_freqs = np.clip(scaled_freqs, 0.5, 7.5)

        else:
            # --- 2. HARD-CODED (NON-RL) TEST MODE ---
            # This block runs ONLY if no rl_action is given 
            # (like in your __main__ test script)
            
            # Set physical parameters directly.
            scaled_amps = np.array([0.1, 0.1, 0.1, 0.1]) # 0.0 amp for stand-still test
            scaled_freqs = np.array([1.0, 1.0, 1.0, 1.0]) # 1.0 Hz
        

        for i in range(4):
            # 1. Set CPG parameters
            self.oscillators[i].set_parameters(scaled_amps[i], scaled_freqs[i])
            
            # 2. Step the oscillator
            x, y = self.oscillators[i].step()
            cpg_states.extend([x, y])
            
            # 3. Map CPG output to joint angles
            abd_gain, hip_gain, knee_gain = self.joint_gains[i] 
            
            # --- NOTE: You swapped your mapping. Is this intended? ---
            # This mapping seems to swap hip and abduction.
            target_joint_positions[i*3 + 0] = x * hip_gain   # Mapped to HIP gain
            target_joint_positions[i*3 + 1] = x * abd_gain   # Mapped to ABDUCTION gain
            target_joint_positions[i*3 + 2] = y * knee_gain  # Knee
            # ---------------------------------------------------------

        # 4. Add CPG output to a base stance
        final_target_positions = self.base_positions + target_joint_positions
        
        # --- RETURN TWO VALUES ---
        # This now matches what your Robot_env.py expects
        return final_target_positions, np.array(cpg_states)

    def reset(self):
        # Start the oscillator at 0 for this test
        A = 0.1
        
        # FR (Leg 0): Phase 0
        self.oscillators[0].x = A
        self.oscillators[0].y = 0.0
        
        # FL (Leg 1): Phase pi (180 degrees)
        self.oscillators[1].x = -A
        self.oscillators[1].y = 0.0
        
        # RR (Leg 2): Phase pi (180 degrees)
        self.oscillators[2].x = -A
        self.oscillators[2].y = 0.0
        
        # RL (Leg 3): Phase 0
        self.oscillators[3].x = A
        self.oscillators[3].y = 0.0

# ==============================================================================
# 3. YOUR REQUESTED TEST SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    print("--- Testing CPGNetwork Output ---")
    
    # 1. Initialize the CPG
    dt = 0.01 # Dummy dt
    cpg = CPGNetwork(dt=dt)
    cpg.reset()

    # 2. Get the base positions
    base_pos = cpg.base_positions
    print(f"Base Positions:\n{base_pos.reshape(4, 3)}\n")

    # 3. Run the 'step' function once. 
    # Because scaled_amps is 0.0 in the 'else' block, the oscillators
    # haven't converged to 0 yet (they are at A=0.1 from reset()).
    # We need to run it a few times to let them converge.
    
    print("Running CPG for 100 steps to let oscillators converge to 0...")
    target_pos = None
    for _ in range(1000):
        target_pos = cpg.step(rl_action=None) # Passing no action

    print(f"Target Positions after 100 steps (with amp=0.0):\n{target_pos.reshape(4, 3)}\n")

    # 4. Calculate and print the difference
    difference = target_pos - base_pos
    print(f"Difference (Target - Base):\n{difference.reshape(4, 3)}\n")

    print("--- Test Result ---")
    if np.allclose(target_pos, base_pos):
        print("SUCCESS: Target Positions correctly converged to Base Positions when Amplitude is 0.0")
    else:
        print("FAILURE: Target Positions are NOT equal to Base Positions.")