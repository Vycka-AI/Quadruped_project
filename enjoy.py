import time
from stable_baselines3 import PPO
from unitree_env import UnitreeEnv
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

# --- Environment Setup ---
env = UnitreeEnv(
    model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
)

# Load the trained agent
model = PPO.load("Neww_with_feet", env=env)

obs, info = env.reset()

print("\n" + "="*50)
print("Simulation is ready. Press SPACE in the window to pause/play.")
print("="*50 + "\n")

# --- Manually Launch Viewer ---
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    # --- THE CORRECT WAY TO ENABLE AND SCALE CONTACT FORCES ---
    
    # 1. Enable the visualization flag in viewer.opt
    # This part tells the viewer to *draw* the forces.
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    # 2. Set the scale for the force vectors in the MjModel's visual properties
    # This part tells the viewer *how big* to draw the forces.
    # It controls the width of the force vector arrows.
    env.model.vis.scale.forcewidth = 0.1
    # This scales the length of the force vector arrows based on magnitude.
    env.model.vis.map.force = 0.5

    # --- Main Simulation Loop ---
    while viewer.is_running():
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
        
        # viewer.sync() is handled by the context manager, but calling it
        # ensures updates if you add manual steps or pauses.
        viewer.sync()

        # Optional: add a small delay to watch it in slow-motion
        time.sleep(1/60)