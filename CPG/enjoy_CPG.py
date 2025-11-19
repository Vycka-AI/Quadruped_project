import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from Robot_env import UnitreeEnv

# --- NEW: Import pynput for reliable key detection ---
from pynput import keyboard

# --- Paths ---
XML_PATH = '../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
MODEL_PATH = "models/Backup/FIRRST_CPG/rl_model_v2__300000_steps.zip"

# --- Global Reset Flag ---
reset_trigger = False

def on_press(key):
    """Callback for keyboard listener."""
    global reset_trigger
    try:
        # Check if the key pressed is 'j'
        if key.char == 'q':
            reset_trigger = True
    except AttributeError:
        # Handles special keys (ctrl, alt, etc.) that don't have a char representation
        pass

# Start the keyboard listener in a non-blocking way
listener = keyboard.Listener(on_press=on_press)
listener.start()

# --- Environment Setup ---
env = UnitreeEnv(
    model_path=XML_PATH,
    render_mode="human",
    frame_skip=10
)

# --- Load the trained PPO agent ---
print(f"Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, env=env)

obs, info = env.reset()

# --- Apply Visualization Settings ---
if env.viewer:
    env.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    env.model.vis.scale.forcewidth = 0.1
    env.model.vis.map.force = 0.5

print("\n" + "="*50)
print("Simulation is ready.")
print("Press SPACE (in viewer) to pause.")
print("Press 'J' (anywhere) to RESET the agent.")
print("="*50 + "\n")

try:
    # --- Main Simulation Loop ---
    while env.viewer and env.viewer.is_running():
        
        # --- 1. Check for 'J' Key Reset ---
        if reset_trigger:
            print("\n>>> Manual Reset Requested (J pressed) <<<")
            obs, info = env.reset()

            print(f"Reset complete. New Command: {env.commands}")
            reset_trigger = False  # Reset the flag so we don't reset forever
        
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for environment resets (falls/timeouts)
        if terminated or truncated:
            print("Episode finished. Auto-resetting.")
            obs, info = env.reset()
            print(f"New Command: {env.commands}")

        # Sleep to match timestep
        time.sleep(env.dt)

except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    listener.stop() # Stop the key listener
    env.close()
