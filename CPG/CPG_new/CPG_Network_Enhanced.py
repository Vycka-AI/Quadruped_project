import numpy as np

class EnhancedHopfOscillator:
    def __init__(self, dt, alpha=50.0): # Increased alpha for faster convergence to limit cycle
        self.dt = dt
        self.alpha = alpha
        self.x, self.y = 1.0, 0.0
        self.mu = 1.0
        self.omega = 2 * np.pi * 1.0
        self.phase_offset = 0.0
        
    def set_parameters(self, amplitude, frequency, phase_offset):
        self.mu = amplitude**2
        self.omega = 2 * np.pi * frequency
        self.phase_offset = phase_offset

    def _dynamics(self, x, y):
        r_sq = x**2 + y**2
        x_dot = self.alpha * (self.mu - r_sq) * x - self.omega * y
        y_dot = self.alpha * (self.mu - r_sq) * y + self.omega * x
        return x_dot, y_dot

    def step(self):
        # RK4 Integration for stability at lower control frequencies
        k1x, k1y = self._dynamics(self.x, self.y)
        k2x, k2y = self._dynamics(self.x + 0.5 * self.dt * k1x, self.y + 0.5 * self.dt * k1y)
        k3x, k3y = self._dynamics(self.x + 0.5 * self.dt * k2x, self.y + 0.5 * self.dt * k2y)
        k4x, k4y = self._dynamics(self.x + self.dt * k3x, self.y + self.dt * k3y)

        self.x += (self.dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        self.y += (self.dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

        # Apply phase modulation to output only (don't disrupt the limit cycle state)
        # Rotation matrix application
        px = self.x * np.cos(self.phase_offset) - self.y * np.sin(self.phase_offset)
        py = self.x * np.sin(self.phase_offset) + self.y * np.cos(self.phase_offset)
        return px, py

    def get_state(self):
        return np.array([self.x, self.y])

class EnhancedCPGNetwork:
    def __init__(self, dt, base_positions):
        self.dt = dt
        self.oscillators = [EnhancedHopfOscillator(dt) for _ in range(4)]
        
        # Gains: [Abduction, Hip, Knee]
        self.joint_gains = np.array([0.1, -0.4, 0.4]) 
        self.base_positions = base_positions
        
        self.param_ranges = {
            'amplitude': (0.0, 1.5),
            'frequency': (1.0, 5.0),
            'phase': (-np.pi, np.pi)
        }

    def step(self, rl_action):
        # Action shape: (12,) -> (4 oscillators, 3 params)
        action_matrix = rl_action.reshape(4, 3)
        
        # --- FIX IS HERE ---
        # 1. Amplitude: Use DIRECTLY (The controller already outputs 0.0 to 1.0)
        amps = action_matrix[:, 0]
        
        # 2. Frequency: Keep scaling (Controller outputs -1 to 1, map to 1Hz-4Hz)
        freqs = self.param_ranges['frequency'][0] + (action_matrix[:, 1] + 1)/2 * (self.param_ranges['frequency'][1] - self.param_ranges['frequency'][0])
        
        # 3. Phase: Keep scaling (Controller outputs -1 to 1 (or 0/1), map to -pi to pi)
        phases = action_matrix[:, 2] * np.pi 

        joint_commands = []
        cpg_states = []

        for i, osc in enumerate(self.oscillators):
            osc.set_parameters(amps[i], freqs[i], phases[i])
            x, y = osc.step()
            
            # Mapping strategy:
            base_idx = i * 3
            
            # Abduction
            abd = self.base_positions[base_idx] 
            
            # Hip (Thigh) - Driven by X
            hip = self.base_positions[base_idx+1] + self.joint_gains[1] * x
            
            # Knee (Calf) - Driven by Y
            # Note: If amp is 0, y is 0, so Knee stays at base_position (Standard Standing)
            knee = self.base_positions[base_idx+2] + self.joint_gains[2] * y
            
            joint_commands.extend([abd, hip, knee])
            cpg_states.extend([x, y])

        return np.array(joint_commands), np.array(cpg_states)
    
    def reset(self):
        for osc in self.oscillators:
            osc.x, osc.y = 1.0, 0.0

