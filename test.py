import gymnasium as gym
from stable_baselines3 import DQN
from envs.dino_env import DinoEnv
import time

def test():
    # Load the environment in human render mode
    env = DinoEnv(render_mode="human")

    # Load the trained model
    try:
        model = DQN.load("dino_ddqn_model")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Please train the model first using train.py")
        return

    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    print("Starting validation run...")
    
    while not terminated:
        # Predict action using the trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # env.render() is called inside step() for human mode
    
    print(f"Game Over! Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test()
