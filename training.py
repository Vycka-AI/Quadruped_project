from stable_baselines3 import PPO
from unitree_env import UnitreeEnv # Assuming your env is in 'unitree_env.py'

# 1. Instantiate the environment
#env = UnitreeEnv(model_path='../unitree_mujoco/unitree_robots/go2/go2.xml')

env = UnitreeEnv(
    model_path='../unitree_mujoco/unitree_robots/go2/scene.xml',#,
    render_mode='human' # This will open a window during training
)

# 2. Instantiate the PPO agent
# 'MlpPolicy' is a standard multi-layer perceptron (a simple neural network)
model = PPO('MlpPolicy', env, verbose=1)

# 3. Start the training!
# This will run for 1 million timesteps. The agent will learn by trial and error.
model.learn(total_timesteps=1_000_000)

# 4. Save the trained model for later use
model.save("ppo_unitree")

# Clean up
env.close()