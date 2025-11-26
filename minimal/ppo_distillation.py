import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from torch.nn import ELU

# Import the Distillation Algorithm
# Ensure the sb3_distill folder is in your python path or installed
try:
    from sb3_distill import ProximalPolicyDistillation
except ImportError:
    import sys
    # Assuming the uploaded folder structure matches
    sys.path.append('spiglerg/sb3_distill/sb3_distill-e82ff6b10259aad6ccc12eba83bbbdba62fb31aa') 
    from sb3_distill import ProximalPolicyDistillation

# --- Your Environment Setup ---
from unitree_env_minimal import UnitreeEnv 

env_id = lambda: UnitreeEnv(
    model_path='../../unitree_mujoco/unitree_robots/go2/scene_ground.xml',
    test_mode=False,
    frame_skip=5
)


num_cpu = 16
folder = "../models/Current/"
teacher_load_path = "../models/Best/Forward_big.zip" # The existing small model
student_model_name = "Mix_big" # The new large model

TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/" + student_model_name + "/"

student_save_path = folder + student_model_name + ".zip"
checkpoint_dir = "../models/Backup/" + student_model_name + "/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Create Env
env = make_vec_env(env_id, n_envs=num_cpu)

# --- 1. Load the TEACHER (Small Model) ---
print(f"Loading Teacher (Small) model from {teacher_load_path}")
# We load it as a standard PPO model. Force CPU to avoid device mismatch errors.
teacher_model = PPO.load(teacher_load_path, env=env, device='cpu')

# --- 2. Define the STUDENT (Large Model) Configuration ---
# This matches the target architecture you wanted
large_policy_kwargs = dict(
    activation_fn=ELU,
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    log_std_init=0.0
)

# --- 3. Initialize the Student using ProximalPolicyDistillation ---
print("Creating Student (Large) model with ProximalPolicyDistillation")
student_model = ProximalPolicyDistillation(
    "MlpPolicy",
    env,
    verbose=1,
    device='cpu', # Keeping CPU as per your successful setup
    n_steps=2048,
    tensorboard_log=TENSORBOARD_LOG_DIR,
    policy_kwargs=large_policy_kwargs,
    # CHANGE 1: Lower learning rate to prevent policy explosion
    learning_rate=5e-5 
)

# --- 4. Link Teacher to Student ---
# CHANGE 2: Lower distill_lambda to 0.5. 
# This acts as a "soft constraint": it keeps the robot upright (like the teacher)
# but isn't so strict that it prevents sideways/backward learning.
student_model.set_teacher(teacher_model, distill_lambda=0.3) 

# --- 5. Train (Distill) ---
checkpoint_callback = CheckpointCallback(
    save_freq=100_000 // num_cpu,
    save_path=checkpoint_dir,
    name_prefix="rl_model_large_student"
)

print("Starting Distillation...")
try:
    student_model.learn(
        total_timesteps=100_000_000,
        reset_num_timesteps=True, # Reset steps since this is a "new" model
        tb_log_name=student_model_name,
        callback=checkpoint_callback
    )
except KeyboardInterrupt:
    print("\n--- Training interrupted by user ---")
finally:
    print(f"Saving final student model to {student_save_path}")
    student_model.save(student_save_path.replace(".zip", ""))
    env.close()