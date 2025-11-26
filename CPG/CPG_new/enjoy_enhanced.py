import time
import argparse
import os
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pynput import keyboard  # Using pynput as requested


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


# IMPORT YOUR ENV CLASS
# Assuming your environment file is named 'unitree_cpg_env.py'
from unitree_env_fixed import UnitreeEnv 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_name = "./models/Backup/New_New/rl_model_v2__965184_steps.zip"
    vec_norm_name = "./models/Backup/New_New/rl_model_v2__965184_steps_vecnormalize.pkl"

    # Paths

    xml_path = "../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml" # Check your path

    # 1. Create the Environment
    # We must wrap it in DummyVecEnv because VecNormalize expects it
    env = DummyVecEnv([lambda: UnitreeEnv(model_path=xml_path, frame_skip=10)])

    # 2. Load Normalization Stats (CRITICAL)
    # Without this, the robot will fail immediately
    if os.path.exists(vec_norm_name):
        env = VecNormalize.load(vec_norm_name, env)
        env.training = False     # Do not update stats during playback
        env.norm_reward = False  # See real rewards
    else:
        print("WARNING: No normalization stats found! If you trained with VecNormalize, this will fail.")

    # 3. Load Agent
    model = PPO.load(model_name, env=env)

    # 4. Access the underlying MuJoCo objects
    # env -> VecNormalize -> DummyVecEnv -> UnitreeEnv
    # We need to drill down to get the model/data for the viewer
    real_env = env.envs[0] 

    obs = env.reset()

    print("\n" + "="*50)
    print("Simulation is ready. Press SPACE in the window to pause/play.")
    print("="*50 + "\n")

    # --- Manually Launch Viewer ---
    with mujoco.viewer.launch_passive(real_env.model, real_env.data) as viewer:

        # --- VISUALIZATION SETTINGS ---
        # Enable Contact Forces
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # Scale the arrows
        real_env.model.vis.scale.forcewidth = 0.05
        real_env.model.vis.map.force = 0.1 

        # --- Main Simulation Loop ---
        print(f"New Commands: {real_env.commands}")
        while viewer.is_running():
            start_time = time.time()
            
            # Predict Action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step Environment
            # Note: DummyVecEnv automatically resets the sub-env if done=True
            obs, reward, done, info = env.step(action)
            
            # Sync Viewer
            viewer.sync()

            # Debug Print: Check commands if a reset happened
            # done is an array of booleans (because of VecEnv)
            if done[0]:
                 print(f"Reset! New Commands: {real_env.commands}")


            if reset_request:
                print("User requested Reset (Q)...")
                obs, info = env.reset()
                #attach_camera() # Re-attach immediately after reset
                reset_request = False
        
            if resample_request:
                print("User requested Resample (V)...")
                env.envs[0]._resample_commands()
                print(f"New Commands: {real_env.commands}")
                #_resample_commands()
                
                #attach_camera() # Re-attach immediately after resampling
                resample_request = False

            # Time Sync (Keep it real-time)
            # We use the dt from the real environment
            step_duration = time.time() - start_time
            time_to_wait = real_env.dt - step_duration
            
            if time_to_wait > 0:
                time.sleep(time_to_wait)
