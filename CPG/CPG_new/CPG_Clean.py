from CPG_Network_Enhanced import EnhancedHopfOscillator
import numpy as np
import time
from unitree_env_fixed import UnitreeEnv
from pynput import keyboard
import matplotlib.pyplot as plt

class GaitAnalyzer:
    def __init__(self):
        self.contact_history = []
        self.time_history = []
        self.feet_names = ['FL', 'FR', 'RL', 'RR']
        
    def log(self, time_sec, contacts):
        """
        time_sec: Current simulation time
        contacts: Boolean array of shape (4,) where True = touching ground
        """
        self.time_history.append(time_sec)
        self.contact_history.append(contacts.copy())

    def plot(self):
        print("\nGenerating Gait Plot...")
        
        # Convert list of arrays to a 2D array (Time x 4)
        data = np.array(self.contact_history).T 
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # We plot horizontal bars for each foot
        # 0=FL, 1=FR, 2=RL, 3=RR
        colors = ['red', 'blue', 'orange', 'green']
        
        for i in range(4):
            # Find time segments where contact is active (True)
            # We construct a collection of horizontal bars
            active_times = []
            start_t = None
            
            for t_idx, is_contact in enumerate(data[i]):
                current_time = self.time_history[t_idx]
                
                if is_contact and start_t is None:
                    start_t = current_time
                elif not is_contact and start_t is not None:
                    # Contact ended
                    ax.hlines(y=i, xmin=start_t, xmax=current_time, 
                              linewidth=20, color=colors[i], label=self.feet_names[i] if t_idx == 0 else "")
                    start_t = None
            
            # Catch the last segment if it ends while still in contact
            if start_t is not None:
                ax.hlines(y=i, xmin=start_t, xmax=self.time_history[-1], 
                          linewidth=20, color=colors[i])

        # Formatting
        ax.set_yticks(range(4))
        ax.set_yticklabels(self.feet_names)
        ax.set_xlabel("Time (s)")
        ax.set_title("Gait Phase Diagram (Solid Bar = Stance Phase)")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.5, 3.5)
        
        plt.tight_layout()
        plt.show()

class ActionAnalyzer:
    """Independent class for logging and plotting CPG actions."""
    def __init__(self):
        self.time_history = []
        self.action_history = [] # Store actions
        self.feet_names = ['FL', 'FR', 'RL', 'RR']

    def log(self, time_sec, action):
        """
        time_sec: Current simulation time
        action: The control action array (12,) [Amp, Freq, Phase, ...] per leg
        """
        self.time_history.append(time_sec)
        self.action_history.append(action.copy())

    def plot(self):
        print("\nGenerating Action Plots...")
        if len(self.action_history) == 0:
            print("No action data to plot.")
            return

        actions = np.array(self.action_history) # Shape: (Time, 12)
        time_axis = np.array(self.time_history)
        
        colors = ['red', 'blue', 'orange', 'green']

        fig2, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig2.suptitle("Control Actions per Leg")
        
        # Indices in the flat action array:
        # Leg i: Amp=(i*3), Freq=(i*3)+1, Phase=(i*3)+2
        
        # Subplot 0: Amplitude
        for i in range(4):
            idx = i * 3
            axes[0].plot(time_axis, actions[:, idx], label=f'{self.feet_names[i]}', color=colors[i])
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc='upper right', fontsize='small', ncol=4)
        axes[0].grid(True)
        
        # Subplot 1: Frequency
        for i in range(4):
            idx = i * 3 + 1
            axes[1].plot(time_axis, actions[:, idx], label=f'{self.feet_names[i]}', color=colors[i], linestyle='--')
        axes[1].set_ylabel("Frequency Cmd")
        axes[1].grid(True)

        # Subplot 2: Phase
        for i in range(4):
            idx = i * 3 + 2
            axes[2].plot(time_axis, actions[:, idx], label=f'{self.feet_names[i]}', color=colors[i])
        axes[2].set_ylabel("Phase Offset")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True)
        
        plt.tight_layout()

class GlobalKeyboardControl:
    def __init__(self):
        self.vel_cmd = 0.0   # Maps to Frequency/Speed
        self.turn_cmd = 0.0  # Maps to Steering
        self.running = True
        self.pressed_keys = set()
        
        print("\n=== KEYBOARD CONTROL (Game-Style) ===")
        print(" [UP]    Forward")
        print(" [DOWN]  Backward")
        print(" [LEFT]  Rotate Left")
        print(" [RIGHT] Rotate Right")
        print(" [ESC]   Quit")
        print("=====================================")

        # Start the listener to capture both press and release events
        self.listener = keyboard.Listener(
            on_press=self.on_press, 
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            return False
        
        # Add key to the set of pressed keys
        self.pressed_keys.add(key)
        self.update_commands()

    def on_release(self, key):
        # Remove key from the set
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        self.update_commands()

    def update_commands(self):
        """Calculate commands based on currently pressed keys"""
        # Default values (Stop)
        target_vel = 0.0
        target_turn = 0.0
        
        # --- Velocity Control (Forward/Backward) ---
        if keyboard.Key.up in self.pressed_keys:
            target_vel += 1.0   # Forward target
        if keyboard.Key.down in self.pressed_keys:
            target_vel -= 1.0   # Backward target

        # --- Turning Control (Left/Right) ---
        if keyboard.Key.left in self.pressed_keys:
            target_turn += 1.5  # Turn Left intensity
        if keyboard.Key.right in self.pressed_keys:
            target_turn -= 1.5  # Turn Right intensity (negative)

        # Apply to commands
        # Note: You can adjust these multipliers to change max speed/turn rate
        self.vel_cmd = np.clip(target_vel, -1.0, 1.0)
        self.turn_cmd = np.clip(target_turn, -3.0, 3.0)

        # Special Case: Rotate in place
        # If the robot needs a non-zero frequency to step/turn, 
        # but you aren't pressing Forward/Back, we might need a small vel_cmd.
        # Uncomment the lines below if the robot refuses to turn while standing still:
        # if self.vel_cmd == 0 and self.turn_cmd != 0:
        #     self.vel_cmd = 0.4  # Small stepping frequency to allow turning

    def stop(self):
        self.listener.stop()


class TrotController:
    def __init__(self):
        # Maximum step height when running at full speed (1.0)
        self.max_amplitude = 0.6 
        
        self.phase_map = [0.0, 1.0, 1.0, 0.0] 
        self.kp_yaw = 0.4

    def get_action(self, current_yaw_rate, user_target_freq, user_target_yaw):
        """
        user_target_freq: Forward speed command (-1.0 to 1.0)
        user_target_yaw:  Turning command (-1.0 to 1.0)
        """
        
        # 1. FREQUENCY (Speed)
        freq_action = user_target_freq

        # 2. CALCULATE BASE AMPLITUDE (Proportional Mapping)
        # We want amplitude to scale with how much we are moving.
        # We look at both forward speed AND turning speed.
        # If we are just turning in place, we still need amplitude!
        motion_magnitude = max(abs(user_target_freq), abs(user_target_yaw))
        
        # Map magnitude (0 to 1) to Amplitude (0 to max_amplitude)
        target_amp = motion_magnitude * self.max_amplitude
        
        # Deadband: If inputs are tiny, just force 0 to stop jitter
        if motion_magnitude < 0.05:
            target_amp = 0.0

        # 3. STEERING CONTROL
        # Error calculation
        yaw_error = user_target_yaw - current_yaw_rate
        steering_correction = self.kp_yaw * yaw_error
        
        # Apply Differential Amplitude
        # Note: We scale correction by target_amp so we don't turn hard when barely moving
        left_amp = target_amp - (steering_correction * 0.3)
        right_amp = target_amp + (steering_correction * 0.3)
        
        # Clip to ensure we don't get negative amplitudes or crazy high ones
        left_amp = np.clip(left_amp, 0.0, 1.0)
        right_amp = np.clip(right_amp, 0.0, 1.0)

        # 4. CONSTRUCT ACTION
        amps = [left_amp, right_amp, left_amp, right_amp]
        
        action = []
        for i in range(4):
            action.extend([amps[i], freq_action, self.phase_map[i]])
            
        return np.array(action)


class WalkController:
    def __init__(self):
        # 1. AMPLITUDE: High base amplitude (0.85)
        # We need this because of the "Duty Factor Hack" (y - 0.5).
        # If amp is too low, the feet won't touch the ground.
        self.amp_action = 0.85 
        
        # 2. PHASE MAP: 4-Beat Walk Pattern
        # Sequence: RL -> FL -> RR -> FR
        self.phase_map = [-0.5, 0.5, -1.0, 0.0] 
        
        # 3. GAINS
        # Slightly higher P-Gain than Trot because walking has more 
        # friction (3 feet on ground) to overcome when turning.
        self.kp_yaw = 0.4 

    def get_action(self, current_yaw_rate, user_target_freq, user_target_yaw):
        """
        current_yaw_rate: From IMU (Gyro Z)
        user_target_freq: From Up/Down Keys (For Walk, usually negative values work best)
        user_target_yaw:  From Left/Right Keys
        """
        
        # --- SPEED CONTROL ---
        freq_action = user_target_freq

        # --- STEERING CONTROL ---
        # Error calculation
        yaw_error = user_target_yaw - current_yaw_rate
        
        # P-Controller for correction
        steering_correction = self.kp_yaw * yaw_error
        
        # Differential Amplitude (Tank Steering)
        # We apply the correction to the heavy 0.85 amplitude
        left_amp = self.amp_action - (steering_correction * 0.2)
        right_amp = self.amp_action + (steering_correction * 0.2)
        
        # Clip to ensure we don't go negative or impossibly high
        left_amp = np.clip(left_amp, 0.5, 1.2)
        right_amp = np.clip(right_amp, 0.5, 1.2)
        
        # Stop Logic: If freq is basically 0, stop lifting legs
        if abs(freq_action) < 0.1:
            left_amp = 0.0
            right_amp = 0.0

        amps = [left_amp, right_amp, left_amp, right_amp]
        
        action = []
        for i in range(4):
            leg_params = [
                amps[i],            # Steering-adjusted Amplitude
                freq_action,        # Speed
                self.phase_map[i]   # Fixed Walk Phase
            ]
            action.extend(leg_params)
            
        return np.array(action)


# --- 5. EXECUTION ---
if __name__ == "__main__":
    # SET YOUR MODEL PATH
    XML_MODEL_PATH = "../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml" # Ensure this file exists relative to this script
    
    # Initialize Environment and Controller
    
    try:
        env = UnitreeEnv(model_path=XML_MODEL_PATH, render_mode="human")
        # 2. Get User Input
            # Map -1..1 input to useful frequency range (e.g., 0.0 to 1.5)
            # We add 0.5 so '0' input is a slow walk, not a stand-still
        key_input = GlobalKeyboardControl() # <--- The new class
        target_freq = key_input.vel_cmd 
        
        # Map steering directly
        target_steer = key_input.turn_cmd
        

        controller = TrotController() # <--- NEW CONTROLLER
        
                # 3. Update CPG Controller
        
        analyzer = GaitAnalyzer()
        action_analyzer = ActionAnalyzer() # <--- NEW: Independent Action Analyzer
        obs, info = env.reset(seed=42)

        TARGET_FPS = 60            # Set this to 30 or 60 for real-time
        SLOW_MO = 1.0
        render_dt = 1.0 / TARGET_FPS
        
        print("Walking with CPG (Trot)... Press Ctrl+C to stop.")
        
        # --- CAMERA SETUP (Optional Initial View) ---
        if env.viewer:
            env.viewer.cam.distance = 3.0  # Zoom out a bit
            env.viewer.cam.azimuth = 90    # Look from the side
            env.viewer.cam.elevation = -20 # Look slightly down

        sim_time = 0.0

        while True:
            # 1. Get action
            target_freq = key_input.vel_cmd 
            
            # Map steering directly
            target_steer = key_input.turn_cmd
            
            loop_start = time.time()

            yaw_rate = obs[5] / env.obs_scales['ang_vel'] # Unscale it to get real rad/s

            action = controller.get_action(
                current_yaw_rate=yaw_rate,
                user_target_freq=target_freq,
                user_target_yaw=target_steer
            )
            
            # 2. Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- 3. CAMERA SNAP LOGIC ---
            if env.viewer and env.viewer.is_running():
                # Get the robot's Base X, Y, Z position from the simulation data
                # qpos[:3] contains the global [x, y, z] of the free joint
                robot_pos = env.data.qpos[:3]
                
                # Update the camera's focus point to match the robot
                env.viewer.cam.lookat[:] = robot_pos
                
                # (Optional) Keep the camera following behind/beside by updating azimuth
                # env.viewer.cam.azimuth += 0.1 # Uncomment to spin around the robot
                
                # Sync the viewer
                env.viewer.sync()
            # ---------------------------
            
            # 3. Calculate how long the physics calculation took
            loop_end = time.time()
            elapsed = loop_end - loop_start
            
            # 4. Sleep to match Target FPS
            # If calculation was fast (0.001s) and target is 60fps (0.016s), 
            # we sleep for the remaining 0.015s.
            sim_time += env.dt
            # Apply Slow Motion factor to the wait time
            wait_time = (render_dt / SLOW_MO) - elapsed
            contacts = env.current_contacts
            print(contacts)
            if sim_time > 5.0 and sim_time < 10.0:
                analyzer.log(sim_time, contacts)
                action_analyzer.log(sim_time, action) # <--- Log actions
            if wait_time > 0:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")

    finally:
        env.close()
        # --- SHOW PLOT ---
        analyzer.plot()
        action_analyzer.plot() # Generate action plots
        plt.show() # Show all open figures