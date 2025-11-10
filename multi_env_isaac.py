import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from copied_isaac import UnitreeEnv
from torch.nn import ELU

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Train a Unitree Go2 robot.")
parser.add_argument('--gui', action='store_true', help="Enable GUI rendering.")
args = parser.parse_args()

# --- Environment creation ---
env_id = lambda: UnitreeEnv(
    model_path='../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
    #render_mode="human", 
    test_mode=False,
    frame_skip=1
)

TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/"
num_cpu = 16  # Change this to match your machine
env = make_vec_env(env_id, n_envs=num_cpu)

# --- Model save path ---

model_name = "Newest_PPOOO"

folder = "models/Current/"
model_save_path = folder + model_name + ".zip"
checkpoint_dir = "models/Backup/" + model_name + "/"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Load or create model ---
if os.path.exists(model_save_path):
    print(f"--- Loading model and continuing training on {num_cpu} environments ---")
    model = PPO.load(model_save_path, env=env, tensorboard_log=TENSORBOARD_LOG_DIR)
else:
    print(f"--- Starting new training on {num_cpu} environments ---")
    policy_kwargs = dict(
        activation_fn=ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        log_std_init=2.0
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        n_steps=1024,  # Number of steps to run for each environment per update
        #ent_coef=0.005,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=policy_kwargs
    )

# --- Checkpoint callback (save every 100k steps) ---
checkpoint_callback = CheckpointCallback(
    save_freq=100_000 // num_cpu,  # adjusted for vectorized envs
    save_path=checkpoint_dir,
    name_prefix="rl_model_v2_"
)

try:
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

env.close()
