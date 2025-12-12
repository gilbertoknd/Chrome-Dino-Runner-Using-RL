import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from envs.dino_env import DinoEnv
import pandas as pd
import matplotlib.pyplot as plt
import config
import plotting


class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.losses = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if 'train/loss' in self.logger.name_to_value:
            self.losses.append(self.logger.name_to_value['train/loss'])
            self.timesteps.append(self.num_timesteps)
        return True

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
    loss_callback = LossCallback()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=loss_callback)

    # Save loss data
    loss_df = pd.DataFrame({'step': loss_callback.timesteps, 'loss': loss_callback.losses})
    loss_df.to_csv(os.path.join(log_dir, 'loss_log.csv'), index=False)

    model.save("dino_ddqn_model")
    print("Model saved.")

    plotting.plot_training_results(log_dir)

    plotting.plot_training_results(log_dir)

if __name__ == "__main__":
    train() #Uses default from config 
