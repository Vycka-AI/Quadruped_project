import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from Robot_env import UnitreeEnv
from torch.nn import ELU

# Move execution logic inside this block
if __name__ == "__main__":
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Train a Unitree Go2 robot.")
    parser.add_argument('--gui', action='store_true', help="Enable GUI rendering.")
    args = parser.parse_args()

    # --- Configuration ---
    TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/"
    num_cpu = 2  # Change this to match your machine
    model_path = '../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'

    # --- Environment creation ---
    # Note: make_vec_env handles the 'make_env' logic internally when you pass the class
    vec_env = make_vec_env(
        UnitreeEnv,
        n_envs=num_cpu,
        seed=0, # It's good practice to set a seed
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"model_path": model_path, "frame_skip": 10}
    )

    # --- Model save path ---
    model_name = "FIRRST_CPG"
    folder = "models/Current/"
    model_save_path = folder + model_name + ".zip"
    checkpoint_dir = "models/Backup/" + model_name + "/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(model_save_path):
        print(f"--- Loading model and continuing training on {num_cpu} environments ---")
        model = PPO.load(model_save_path, env=vec_env, tensorboard_log=TENSORBOARD_LOG_DIR)
    else:

        # --- Policy Setup ---
        policy_kwargs = dict(
            activation_fn=ELU,
            net_arch=dict(pi=[256, 128], qf=[256, 128]),
            log_std_init=0.7
        )

        # --- Model Initialization ---
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cpu",
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

    # --- Checkpoint callback ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_cpu,
        save_path=checkpoint_dir,
        name_prefix="rl_model_v2_"
    )

    # --- Training Loop ---
    try:
        print("Starting training...")
        model.learn(
            total_timesteps=100_000_000_000,
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    finally:
        print(f"Saving final model to {model_save_path}")
        model.save(model_save_path.replace(".zip", ""))
        print("Model saved. Exiting.")
        vec_env.close()