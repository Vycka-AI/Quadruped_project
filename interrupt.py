import os
import argparse  # Import the argparse library
from stable_baselines3 import PPO
from unitree_env import UnitreeEnv

# 1. --- Set up Argument Parser ---
parser = argparse.ArgumentParser(description="Train a Unitree Go2 robot.")
parser.add_argument(
    '--gui',
    action='store_true',  # Makes it a flag: if present, sets value to True
    help="Enable GUI rendering for the simulation."
)
args = parser.parse_args()

# 2. --- Use the Argument to Set Render Mode ---
# If --gui is used, render_mode is 'human', otherwise it is None (headless)
render_mode = 'human' if args.gui else None
if render_mode == 'human':
    print("GUI mode enabled. Training will be VERY slow.")
else:
    print("Headless mode enabled. Training will be fast.")


# --- The rest of your script ---
model_save_path = "New.zip"

env = UnitreeEnv(
    model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
    render_mode=render_mode  # Use the variable here
)

# Check if a model file already exists
if os.path.exists(model_save_path):
    # Load the existing model
    print(f"--- Loading existing model from {model_save_path} ---")
    model = PPO.load(model_save_path, env=env)
    # Create the new model with a higher entropy coefficient

    print("Model loaded. Continuing training...")
else:
    # Create a new model if one doesn't exist
    print("--- No model found, starting new training ---")
    model = PPO("MlpPolicy", env, verbose=1)

try:
    model.learn(
        total_timesteps=10_000_000,
        reset_num_timesteps=False
    )
except KeyboardInterrupt:
    print("\n--- Training interrupted by user ---")
finally:
    # Save the model
    print(f"Saving model to {model_save_path}")
    model.save(model_save_path.replace(".zip", ""))
    print("Model saved. Exiting.")

env.close()