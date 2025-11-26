import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from torch.nn import ELU
from unitree_env_minimal import UnitreeEnv

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on device: {device}")

env_id = lambda: UnitreeEnv(
    model_path='../../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
    test_mode=False,
    frame_skip=5
)

TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/Stabler_New/"
num_cpu = 16
model_name = "Stabler_New"
folder = "../models/Current/"
model_load_path = "../models/Best/Stabler.zip" 
model_save_path = folder + model_name + ".zip"
checkpoint_dir = "../models/Backup/" + model_name + "/"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Hyperparameters ---
train_n_steps = 4096
train_batch_size = 2048 
learning_rate = 5e-4 # You might want to bump this slightly to help the critic learn fast initially

env = make_vec_env(env_id, n_envs=num_cpu)

# --- Network Architecture ---
policy_kwargs = dict(
    activation_fn=ELU,
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    log_std_init=0.0
)

# --- HELPER: Function to re-init weights ---
def init_weights_orthogonal(module):
    """
    Applies orthogonal initialization to linear layers (standard for PPO).
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

# --- PPO Loading & Critic Surgery ---
if os.path.exists(model_load_path):
    print(f"Loading ACTOR from {model_load_path}")
    model = PPO.load(model_load_path, env=env, device=device, tensorboard_log=TENSORBOARD_LOG_DIR)
    
    # --- 1. FORCE HYPERPARAMETER UPDATES ---
    print(f"Updating model parameters: n_steps={train_n_steps}, batch_size={train_batch_size}")
    model.n_steps = train_n_steps
    model.batch_size = train_batch_size
    model.learning_rate = learning_rate 
    model.rollout_buffer.buffer_size = train_n_steps
    model.rollout_buffer.reset()

    # --- 2. CRITIC LOBOTOMY (Reset Weights) ---
    print("-------------------------------------------------")
    print("SURGERY: Resetting Critic (Value Net) weights...")
    print("-------------------------------------------------")
    
    # In SB3, the Actor and Critic are in model.policy
    # The 'mlp_extractor' contains the separate networks for pi (actor) and vf (critic)
    # The 'value_net' is the final projection head.
    
    # A. Reset the hidden layers of the Value Function
    # 
    model.policy.mlp_extractor.value_net.apply(init_weights_orthogonal)
    
    # B. Reset the final output head of the Value Function
    model.policy.value_net.apply(init_weights_orthogonal)
    
    # C. Reset the Optimizer
    # This is CRITICAL. The optimizer holds "momentum" from previous training.
    # If we don't clear this, the first update will apply old momentum to new random weights,
    # causing an explosion.
    model.policy.optimizer.state.clear()

    print(">>> Critic weights randomized. Actor weights preserved.")
    print(">>> Optimizer state cleared.")

else:
    print("Creating new PPO model [512, 256, 128]")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        n_steps=train_n_steps,
        batch_size=train_batch_size,
        n_epochs=10,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate
    )

checkpoint_callback = CheckpointCallback(
    save_freq=100_000 // num_cpu,
    save_path=checkpoint_dir,
    name_prefix="rl_model_gpu"
)

print("Starting training...")
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

env.close()