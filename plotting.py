import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def plot_training_results(log_dir="logs/"):
    results_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(results_path):
        print(f"No monitor.csv found in {log_dir}. Skipping plotting.")
        return

    #Read the CSV, skipping the first line (metadata)
    try:
        df = pd.read_csv(results_path, skiprows=1)
    except Exception as e:
        print(f"Error reading monitor file: {e}")
        return

    #Create plots
    plt.style.use('ggplot')
    # Change to 2x1 grid, or 2x2 if we want to keep it spacious, but we only have 3 graphs now.
    # User asked to remove "reward over time".
    # Remaining: Reward, Duration, Loss.
    # Let's do 3 rows, 1 column for clarity, or 2x2 with one empty/shared.
    # Let's stick to the 2x2 grid from previous step but remove the 3rd one and move Loss to 3rd position?
    # Or just 3 rows. 3 rows is cleaner for vertical scanning.
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    #Reward
    window = 50
    if len(df) > window:
        rolling_mean = df['r'].rolling(window=window).mean()
        axs[0].plot(df['r'], alpha=0.3, label='Reward', color='tab:orange')
        axs[0].plot(rolling_mean, label=f'Rolling Mean ({window})', color='tab:red')
    else:
        axs[0].plot(df['r'], label='Reward', color='tab:orange')
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    #Duration (l = length)
    axs[1].plot(df['l'], color='tab:blue')
    axs[1].set_title('Episode Duration (Steps)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')

    # Training Loss
    loss_path = os.path.join(log_dir, "loss_log.csv")
    if os.path.exists(loss_path):
        loss_df = pd.read_csv(loss_path)
        # Apply rolling mean to smooth loss if many points
        if len(loss_df) > 100:
             loss_smooth = loss_df['loss'].rolling(window=50).mean()
             axs[2].plot(loss_df['step'], loss_df['loss'], alpha=0.3, label='Loss', color='tab:purple')
             axs[2].plot(loss_df['step'], loss_smooth, label='Rolling Mean (50)', color='indigo')
        else:
             axs[2].plot(loss_df['step'], loss_df['loss'], label='Loss', color='tab:purple')
        
        axs[2].set_title('Training Loss')
        axs[2].set_xlabel('Timesteps')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
    else:
        axs[2].text(0.5, 0.5, 'No Loss Data Available', horizontalalignment='center', verticalalignment='center')
        axs[2].set_title('Training Loss')

    plt.tight_layout()
    output_path = "training_results.png" # Saving to root as before
    plt.savefig(output_path)
    print(f"Plots saved to {output_path}")
    # plt.show() # Don't show by default to avoid blocking if run on server/headless, user can open the png

def plot_validation_results(rewards, durations, results_dir):
    #Create plots
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    #Rewards
    axs[0].plot(rewards, label='Episode Reward', color='tab:orange')
    axs[0].axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    axs[0].set_title('Validation: Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    
    #Durations
    axs[1].plot(durations, label='Episode Duration', color='tab:blue')
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
    parser = argparse.ArgumentParser(description="Plot training results.")
    parser.add_argument("--log_dir", type=str, default="logs/", help="Path to the logs directory.")
    args = parser.parse_args()
    
    plot_training_results(args.log_dir)
