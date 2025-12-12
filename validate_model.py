import gymnasium as gym
from stable_baselines3 import DQN
from envs.dino_env import DinoEnv
import config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import imageio
from datetime import datetime

def validate():
    validation_steps = int(config.TOTAL_TIMESTEPS * config.VALIDATION_FRACTION)
    print(f"Starting validation for {validation_steps} steps (Greedy Policy)...")

    #Load environment with RGB array capture
    env = DinoEnv(render_mode="rgb_array") 

    #Load model
    try:
        model = DQN.load("dino_ddqn_model")
    except FileNotFoundError:
        print("Model not found. Train first.")
        return

    rewards = []
    durations = []
    
    total_steps = 0
    episode_count = 0
    
    best_reward = -float('inf')
    best_frames = []

    while total_steps < validation_steps:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        
        current_frames = []
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            #Capture frame
            frame = env.render()
            current_frames.append(frame)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if total_steps >= validation_steps:
                break
        
        rewards.append(episode_reward)
        durations.append(episode_steps)
        episode_count += 1
        print(f"Episode {episode_count}: Reward={episode_reward:.2f}, Steps={episode_steps} (Total Progress: {total_steps}/{validation_steps})")
        
        #Save best run
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = current_frames
            print(f"New High Score! (Reward: {best_reward:.2f})")

    env.close()

    #Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"validation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    #Save video
    if best_frames:
        video_path = os.path.join(results_dir, "best_run.mp4")
        print(f"Saving best run video ({len(best_frames)} frames) to {video_path}...")
        imageio.mimsave(video_path, best_frames, fps=30)
        print("Video saved.")

    #Plotting
    print("Generating validation plots...")
    plot_validation_results(rewards, durations, results_dir)

def plot_validation_results(rewards, durations, results_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    #Rewards
    axs[0].plot(rewards, label='Episode Reward')
    axs[0].axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    axs[0].set_title('Validation: Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    
    #Durations
    axs[1].plot(durations, label='Episode Duration')
    axs[1].axhline(y=np.mean(durations), color='r', linestyle='--', label=f'Mean: {np.mean(durations):.2f}')
    axs[1].set_title('Validation: Episode Duration (Steps)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')
    axs[1].legend()

    plt.tight_layout()
    output_file = os.path.join(results_dir, "validation_results.png")
    plt.savefig(output_file)
    print(f"Validation plots saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    validate()
