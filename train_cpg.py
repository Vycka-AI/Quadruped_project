# --- START OF FILE train_cpg.py (with GUI argument) ---

import os
import argparse # 1. Import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from unitree_cpg_env import UnitreeCPGEnv
from torch.nn import Tanh

# 2. --- Set up Argument Parser ---
parser = argparse.ArgumentParser(description="Train a Unitree Go2 robot with a CPG controller.")
parser.add_argument(
    '--gui',
    action='store_true', # Makes it a flag: if present, sets value to True
    help="Enable GUI rendering for debugging (forces n_envs=1)."
)
args = parser.parse_args()

# --- Environment creation and paths ---
TENSORBOARD_LOG_DIR = "./ppo_go2_cpg_tensorboard/"
model_save_path = "Go2_CPG_Model.zip"
checkpoint_dir = model_save_path.replace(".zip", "")
os.makedirs(checkpoint_dir, exist_ok=True)

# 3. --- Conditionally Create Environment ---
if args.gui:
    # If --gui is used, create a single, renderable environment
    print("--- GUI mode enabled. Training will be slow (n_envs=1). ---")
    num_cpu = 1
    env = UnitreeCPGEnv(
        model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
        render_mode="human"
    )
else:
    # If --gui is NOT used, create a vectorized environment for fast training
    print("--- Headless mode enabled. Training with multiple environments. ---")
    num_cpu = 16 # Or your desired number of parallel environments
    env_id = lambda: UnitreeCPGEnv(
        model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
    )
    env = make_vec_env(env_id, n_envs=num_cpu)

# --- Load or create model (this part remains the same) ---
if os.path.exists(model_save_path):
    print(f"--- Loading CPG model and continuing training ---")
    model = PPO.load(model_save_path, env=env, tensorboard_log=TENSORBOARD_LOG_DIR)
else:
    print(f"--- Starting new CPG training ---")
    policy_kwargs = dict(
        activation_fn=Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=policy_kwargs
    )

# --- Checkpoint callback ---
# The save frequency is adjusted for the number of environments
checkpoint_callback = CheckpointCallback(
    save_freq=max(1, 100_000 // num_cpu), # Use max(1, ...) to prevent division by zero
    save_path=checkpoint_dir,
    name_prefix="cpg_model"
)

try:
    model.learn(
        total_timesteps=50_000_000,
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )
except KeyboardInterrupt:
    print("\n--- Training interrupted by user ---")
finally:
    print(f"Saving final CPG model to {model_save_path}")
    model.save(model_save_path.replace(".zip", ""))
    print("Model saved. Exiting.")

env.close()