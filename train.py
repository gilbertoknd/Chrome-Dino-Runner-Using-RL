import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.dino_env import DinoEnv
import pandas as pd
import matplotlib.pyplot as plt

import config

def train(total_timesteps=config.TOTAL_TIMESTEPS):
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = DinoEnv()
    env = Monitor(env, log_dir)

    #Initializing agent
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        learning_starts=config.LEARNING_STARTS,
        batch_size=config.BATCH_SIZE,
        #Epsilon Decay is controlled by exploration_fraction (duration) and exploration_final_eps (target)
        #Decay happens linearly from 1.0 to final_eps over the first (fraction * total_timesteps) steps.
        exploration_fraction=config.EXPLORATION_FRACTION,
        exploration_final_eps=config.EXPLORATION_FINAL_EPS,

        target_update_interval=config.TARGET_UPDATE_INTERVAL,
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,

    )

    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save("dino_ddqn_model")
    print("Model saved.")

    plot_results(log_dir)

def plot_results(log_dir):
    results_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(results_path):
        print("No monitor.csv found. Skipping plotting.")
        return

    #Read the CSV, skipping the first line (metadata)
    try:
        df = pd.read_csv(results_path, skiprows=1)
    except Exception as e:
        print(f"Error reading monitor file: {e}")
        return

    #Create plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    #Reward
    window = 50
    if len(df) > window:
        rolling_mean = df['r'].rolling(window=window).mean()
        axs[0].plot(df['r'], alpha=0.3, label='Reward')
        axs[0].plot(rolling_mean, label=f'Rolling Mean ({window})')
    else:
        axs[0].plot(df['r'], label='Reward')
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    #Duration (l = length)
    axs[1].plot(df['l'])
    axs[1].set_title('Episode Duration (Steps)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')

    #Time/Wall clock
    axs[2].plot(df['t'] / 60, df['r']) #Time in minutes vs reward
    axs[2].set_title('Reward over Time')
    axs[2].set_xlabel('Time (min)')
    axs[2].set_ylabel('Reward')

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Plots saved to training_results.png")
    plt.show(block=False) #Non-blocking show
    plt.pause(3)
    plt.close()

if __name__ == "__main__":
    train() #Uses default from config 
