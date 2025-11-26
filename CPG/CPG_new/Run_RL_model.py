import time
import argparse
import os
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pynput import keyboard  # Using pynput as requested

# IMPORT YOUR ENV CLASS
from unitree_env_fixed import UnitreeEnv 

# --- Global flags for key presses ---
reset_request = False
resample_request = False

class ActionAnalyzer:
    """Independent class for logging and plotting Actions."""
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
        fig2.suptitle("RL Control Actions per Leg")
        
        # Indices in the flat action array:
        # Leg i: Amp=(i*3), Freq=(i*3)+1, Phase=(i*3)+2
        
        # Subplot 0: Amplitude
        for i in range(4):
            idx = i * 3
            if idx < actions.shape[1]:
                axes[0].plot(time_axis, actions[:, idx], label=f'{self.feet_names[i]}', color=colors[i])
        axes[0].set_ylabel("Amplitude / Param 1")
        axes[0].legend(loc='upper right', fontsize='small', ncol=4)
        axes[0].grid(True)
        
        # Subplot 1: Frequency
        for i in range(4):
            idx = i * 3 + 1
            if idx < actions.shape[1]:
                axes[1].plot(time_axis, actions[:, idx], label=f'{self.feet_names[i]}', color=colors[i], linestyle='--')
        axes[1].set_ylabel("Frequency / Param 2")
        axes[1].grid(True)

        # Subplot 2: Phase
        for i in range(4):
            idx = i * 3 + 2
            if idx < actions.shape[1]:
                axes[2].plot(time_axis, actions[:, idx], label=f'{self.feet_names[i]}', color=colors[i])
        axes[2].set_ylabel("Phase / Param 3")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True)
        
        plt.tight_layout()

def on_press(key):
    """
    Callback for pynput listener. 
    Runs in a separate thread, so we just toggle flags here.
    """
    global reset_request, resample_request
    try:
        # Check for character keys
        if hasattr(key, 'char'):
            if key.char == 'q':
                reset_request = True
            elif key.char == 'v':
                resample_request = True
    except AttributeError:
        pass

# Start the non-blocking listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated paths based on user input, ensure these are correct for your local setup
    model_name = "./models/Backup/New_New/rl_model_v2__2840448_steps.zip"
    vec_norm_name = "./models/Backup/New_New/rl_model_v2__2840448_steps_vecnormalize.pkl"
    xml_path = "../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml" 

    # 1. Create the Environment
    env = DummyVecEnv([lambda: UnitreeEnv(model_path=xml_path, frame_skip=10)])

    # 2. Load Normalization Stats
    if os.path.exists(vec_norm_name):
        print(f"Loading VecNormalize stats from {vec_norm_name}")
        env = VecNormalize.load(vec_norm_name, env)
        env.training = False     # Do not update stats during playback
        env.norm_reward = False  # See real rewards
    else:
        print("WARNING: No normalization stats found! If you trained with VecNormalize, this will fail.")

    # 3. Load Agent
    print(f"Loading Model from {model_name}")
    model = PPO.load(model_name, env=env)

    # 4. Access the underlying MuJoCo objects
    real_env = env.envs[0] 

    # Initialize Action Analyzer
    action_analyzer = ActionAnalyzer()

    obs = env.reset()

    print("\n" + "="*50)
    print("Simulation is ready. Press SPACE in the window to pause/play.")
    print("Press 'q' to Reset, 'v' to Resample commands.")
    print("="*50 + "\n")

    try:
        # --- Manually Launch Viewer ---
        with mujoco.viewer.launch_passive(real_env.model, real_env.data) as viewer:

            # --- VISUALIZATION SETTINGS ---
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            real_env.model.vis.scale.forcewidth = 0.05
            real_env.model.vis.map.force = 0.1 

            # --- Main Simulation Loop ---
            print(f"New Commands: {real_env.commands}")
            
            sim_time = 0.0
            
            while viewer.is_running():
                start_time = time.time()
                
                # Predict Action
                # action is shape (n_envs, action_dim), here (1, 12)
                action, _states = model.predict(obs, deterministic=True)
                
                # Log Action for plotting
                # We take action[0] because the model returns a batch of actions
                action_analyzer.log(sim_time, action[0])

                # Step Environment
                obs, reward, done, info = env.step(action)
                
                # Sync Viewer
                viewer.sync()

                # Debug Print: Check commands if a reset happened
                if done[0]:
                     print(f"Reset! New Commands: {real_env.commands}")

                if reset_request:
                    print("User requested Reset (Q)...")
                    obs = env.reset() # env.reset() returns just obs in SB3 DummyVecEnv usually, or (obs, info) depending on version. SB3 usually just obs.
                    reset_request = False
        
                if resample_request:
                    print("User requested Resample (V)...")
                    # We need to access the inner env to force a command resample without a full reset if desired,
                    # or just reset. The original script called _resample_commands() on the inner env.
                    env.envs[0]._resample_commands()
                    print(f"New Commands: {real_env.commands}")
                    resample_request = False

                # Time Sync (Keep it real-time)
                step_duration = time.time() - start_time
                time_to_wait = real_env.dt - step_duration
                
                sim_time += real_env.dt # Increment local sim time tracking

                if time_to_wait > 0:
                    time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("\nSimulation Stopped by User.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        # PLOT AT THE END
        action_analyzer.plot()
        plt.show()