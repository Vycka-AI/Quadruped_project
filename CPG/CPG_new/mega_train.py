import os
import argparse
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Replace with your actual file name if different
from unitree_env_fixed import UnitreeEnv 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Unitree Go2 robot.")
    parser.add_argument('--gui', action='store_true', help="Enable GUI rendering.")
    args = parser.parse_args()

    # --- Configuration ---
    TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/"
    model_path = '../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
    
    # Training Settings
    num_cpu = 16  # Increased for better throughput
    total_timesteps = 100_000_000_000
    
    # Save Paths
    model_name = "CPG_12"
    folder = "models/Current/"
    checkpoint_dir = f"models/Backup/{model_name}/"
    
    model_save_path = os.path.join(folder, f"{model_name}.zip")
    stats_save_path = os.path.join(folder, "vec_normalize.pkl") # CRITICAL for resuming

    os.makedirs(folder, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 1. Create Environment ---
    # We use SubprocVecEnv for parallel training
    env = make_vec_env(
        UnitreeEnv,
        n_envs=num_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"model_path": model_path, "frame_skip": 10}
    )

    # --- 2. Wrap in Normalization ---
    # This scales inputs to mean 0, std 1. Essential for PPO.
    # We initialize it here, but might overwrite it if loading.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- 3. Check for Existing Model & Stats (Resume Logic) ---
    if os.path.exists(model_save_path):
        print(f"\n--- Found existing model at {model_save_path} ---")
        
        # A. Load Normalization Statistics first
        if os.path.exists(stats_save_path):
            print(f"--- Loading normalization stats from {stats_save_path} ---")
            env = VecNormalize.load(stats_save_path, env.venv) # Load into the existing venv
            env.training = True # Important: keep updating stats during training
            env.norm_reward = True
        else:
            print("!!! WARNING: Model found but no normalization stats found. Training might be unstable !!!")

        # B. Load the PPO Agent
        # We pass the env so the agent knows the current normalization state
        model = PPO.load(model_save_path, env=env, tensorboard_log=TENSORBOARD_LOG_DIR)
        print(f"--- Model loaded. Continuing training... ---")
        
    else:
        print(f"\n--- No existing model found. Creating new model '{model_name}' ---")
        
        policy_kwargs = dict(
            activation_fn=nn.ELU,
            net_arch=dict(pi=[256, 128], qf=[256, 128]),
            log_std_init=-2.0 # Start with low exploration noise for CPG tasks
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            ent_coef=0.005,
            clip_range=0.2,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=TENSORBOARD_LOG_DIR,
        )

    # --- 4. Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_cpu,
        save_path=checkpoint_dir,
        name_prefix="rl_model_v2_"
    )

    # --- 5. Training Loop ---
    try:
        print("Starting training loop...")
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False, # CRITICAL: Keeps the Tensorboard graph continuous
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    finally:
        # --- 6. Safe Saving on Exit ---
        print(f"Saving final model to {model_save_path}...")
        model.save(model_save_path.replace(".zip", ""))
        
        print(f"Saving normalization stats to {stats_save_path}...")
        env.save(stats_save_path)
        
        print("Save complete. Exiting.")
        env.close()