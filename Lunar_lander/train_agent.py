import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Initialize the environment
lander_env = gym.make("LunarLander-v3")
lander_env = Monitor(lander_env)
lander_env = DummyVecEnv([lambda: lander_env])

# Define a policy model with minor adjustments
policy_config = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[256, 256]  # Two-layer MLP architecture
)

# Custom learning rate function
def adaptive_lr(start_value):
    return lambda progress: start_value * (1 - progress)

# Attempt to load a pre-trained model; otherwise, create a fresh one
try:
    agent_model = PPO.load("ppo_trained_lander", env=lander_env)
    print("Pretrained model loaded successfully!")
except:
    print("No existing model found, creating a new PPO agent.")
    agent_model = PPO(
        "MlpPolicy", lander_env, policy_kwargs=policy_config, verbose=1,
        learning_rate=adaptive_lr(3e-4),
        gamma=0.99,
        batch_size=256,
        n_steps=4096,
        ent_coef=0.005,
        normalize_advantage=True,
        tensorboard_log="./lander_ppo_logs/"
    )

# Training parameters
TOTAL_TRAINING_STEPS = 5_500_000
BATCH_TRAIN_SIZE = 550_000

for iteration in range(TOTAL_TRAINING_STEPS // BATCH_TRAIN_SIZE):
    print(f"Training phase {iteration + 1}/{TOTAL_TRAINING_STEPS // BATCH_TRAIN_SIZE}...")
    agent_model.learn(total_timesteps=BATCH_TRAIN_SIZE, tb_log_name="Lander_PPO")
    agent_model.save("ppo_trained_lander")

# Save trained policy parameters without pickling
trained_policy = agent_model.policy.state_dict()
numpy_representation = {key: value.cpu().numpy() for key, value in trained_policy.items()}

# Create a scalar (0-d) structured array to store the dictionary fields without extra dimensions.
dtype = [(key, np.float32, val.shape) for key, val in numpy_representation.items()]
policy_struct = np.empty((), dtype=dtype)  # 0-d structured array
for key, val in numpy_representation.items():
    policy_struct[key] = val

np.save("best_policy.npy", policy_struct)
print("Training complete! Policy saved as best_policy.npy")

lander_env.close()
