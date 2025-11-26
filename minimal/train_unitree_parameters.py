import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from torch.nn import ELU
from unitree_env_minimal import UnitreeEnv

env_id = lambda: UnitreeEnv(
    model_path='../../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
    test_mode=False,
    frame_skip=5
)

TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/Stabler/"
num_cpu = 16
#model_name = "PPO_Minimal_New"
model_name = "Stabler"
folder = "../models/Current/"
model_save_path = folder + model_name + ".zip"
model_load_path = "../models/Best/Stabeler.zip"
normalize_path = folder + model_name + "_vecnormalize.pkl"
checkpoint_dir = "../models/Backup/" + model_name + "/"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- VecNormalize: load if exists, else create new ---
env = make_vec_env(env_id, n_envs=num_cpu)
#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

policy_kwargs = dict(
    activation_fn=ELU,
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    log_std_init=0.0
)

# --- PPO: load if exists, else create new ---
if os.path.exists(model_load_path):
    print(f"Loading model from {model_load_path}")
    model_load = PPO.load(model_load_path, env=env)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        ent_coef=0.02,
        n_steps = 4096,
        batch_size = 2048,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4
    )
    model.set_parameters(model_load.get_parameters())
    #model.n_steps = 4096  # Update n_steps if needed
    model.policy.optimizer.load_state_dict(model_load.policy.optimizer.state_dict())
else:
    print("Creating new PPO model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        n_steps=2048,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4
    )

checkpoint_callback = CheckpointCallback(
    save_freq=100_000 // num_cpu,
    save_path=checkpoint_dir,
    name_prefix="rl_model_minimal"
)

try:
    model.learn(
        total_timesteps=100_000_000,
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )
except KeyboardInterrupt:
    print("\n--- Training interrupted by user ---")
finally:
    print(f"Saving final model to {model_save_path}")
    model.save(model_save_path.replace(".zip", ""))
    print(f"Saving VecNormalize statistics to {normalize_path}")
    #env.save(normalize_path)
    print("Model and normalization stats saved. Exiting.")

env.close()
