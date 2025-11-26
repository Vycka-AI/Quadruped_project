import time
import threading
from pynput import keyboard  # Using pynput as requested
from stable_baselines3 import PPO
from unitree_env_minimal import UnitreeEnv
import mujoco
import mujoco.viewer
import numpy as np

# --- Global flags for key presses ---
# These are toggled by the background thread listener
reset_request = False
resample_request = False

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

# --- Environment Setup ---
env = UnitreeEnv(
    model_path='../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
)

# Load the trained agent
#model = PPO.load("../models/Backup/Large_Student_A_bit_Newer/rl_model_large_student_1000000_steps.zip", env=env)
#model = PPO.load("../models/Backup/A_bit_Newer/rl_model_minimal_15286336_steps.zip", env=env)
model = PPO.load("../models/Backup/Stabler/rl_model_minimal_1500000_steps.zip", env=env)

obs, info = env.reset()

print("\n" + "="*50)
print("Simulation is ready.")
print("Controls (pynput enabled):")
print("  [Q]     : Reset Robot/Episode")
print("  [V]     : Resample Commands (Velocity targets)")
print("  [SPACE] : Pause/Resume (Focus window first)")
print("="*50 + "\n")

# --- Manually Launch Viewer ---
# Note: We removed the key_callback argument since we are using pynput
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    # --- CAMERA TRACKING SETUP ---
    # We wrap this in a function so we can re-call it after resets
    track_body_name = "base_link" 
    robot_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)

    def attach_camera():
        """Refreshes the camera tracking settings."""
        if robot_body_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = robot_body_id
            viewer.cam.distance = 3.0 
            viewer.cam.elevation = -20
            # We don't set lookat here so it tracks naturally
        else:
            print(f"WARNING: Body '{track_body_name}' not found. Camera will not track.")

    # Initial attachment
    attach_camera()

    # --- VISUALIZATION SETUP ---
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    env.model.vis.scale.forcewidth = 0.1
    env.model.vis.map.force = 0.5

    # --- FPS Control Setup ---
    TARGET_FPS = 30.0
    FRAME_DURATION = 1.0 / TARGET_FPS 

    # --- Main Simulation Loop ---
    while viewer.is_running():
        loop_start_time = time.time()

        # 1. Handle User Inputs (from pynput flags)
        if reset_request:
            print("User requested Reset (Q)...")
            obs, info = env.reset()
            attach_camera() # Re-attach immediately after reset
            reset_request = False
        
        if resample_request:
            print("User requested Resample (Z)...")
            if hasattr(env, 'resample_commands'):
                env.resample_commands()
            elif hasattr(env, '_resample_commands'):
                env._resample_commands()
            else:
                print("Error: Could not find a 'resample_commands' method in UnitreeEnv.")
            
            if hasattr(env, 'commands'):
                print(f"New Commands: {env.commands}")
            
            attach_camera() # Re-attach immediately after resampling
            resample_request = False

        # 2. Step the Physics
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
            attach_camera() # Re-attach immediately after auto-reset

        # 3. Render
        viewer.sync()

        # 4. FPS Lock
        loop_end_time = time.time()
        elapsed = loop_end_time - loop_start_time
        
        if elapsed < FRAME_DURATION:
            time.sleep(FRAME_DURATION - elapsed)

# Stop the listener when the loop exits
listener.stop()