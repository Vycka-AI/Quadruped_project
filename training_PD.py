# Updated training.py

import os # Import os to check for existing model file
from stable_baselines3 import PPO
from PD_env import UnitreeEnv # Assuming updated env is in 'unitree_env.py'

# --- Configuration ---
MODEL_NAME = "ppo_unitree_pd" # New name for the PD-controlled model
SAVE_PATH = f"{MODEL_NAME}.zip"

# Check if an old model exists and warn the user
OLD_MODEL_NAME = "ppo_unitree.zip"
if os.path.exists(OLD_MODEL_NAME):
    print(f"WARNING: Found old model '{OLD_MODEL_NAME}'.")
    print("The environment's action space has changed (now uses PD control).")
    print(f"This script will save the new model as '{SAVE_PATH}'.")
    print(f"Consider deleting or renaming '{OLD_MODEL_NAME}' if you don't need it.")
    # input("Press Enter to continue...") # Optional: Pause to ensure user sees warning


# 1. Instantiate the environment
# Make sure this path points to your go2.xml or scene.xml with the robot
# Using scene_ground.xml as it was used in multi_env.py
try:
    env = UnitreeEnv(
        model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
        render_mode='human' # Keep render_mode=None for faster training
    )
except Exception as e:
    print(f"Error creating environment: {e}")
    exit()

# 2. Instantiate the PPO agent
# 'MlpPolicy' is suitable. verbose=1 shows training progress.
# If you want to load and continue training, use PPO.load() instead.
print(f"--- Starting new training for {MODEL_NAME} ---")
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_go2_pd_tensorboard/")

# 3. Start the training!
# Adjust total_timesteps as needed. 1 million is a starting point.
try:
    model.learn(total_timesteps=1_000_000, log_interval=10) # Log every 10 updates
except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    # 4. Save the trained model
    print(f"Saving model to {SAVE_PATH}...")
    model.save(SAVE_PATH)
    print("Model saved.")

    # 5. Clean up
    env.close()
    print("Environment closed.")