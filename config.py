#Training Hyperparameters
TOTAL_TIMESTEPS = 500000  #Total steps for training
VALIDATION_FRACTION = 0.10  #Fraction of total timesteps to run for validation (e.g., 10%)

#Environment Config
RENDER_FPS = 30

# DDQN Hyperparameters
LEARNING_RATE = 1e-4
BUFFER_SIZE = 50000
LEARNING_STARTS = 1000
BATCH_SIZE = 32
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.02
TARGET_UPDATE_INTERVAL = 1000
TRAIN_FREQ = 4
GRADIENT_STEPS = 1

