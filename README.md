# Chrome Dino Runner with Reinforcement Learning (DDQN)

This project implements a Reinforcement Learning agent (using Double DQN from Stable Baselines3) to play the Chrome Dino Run game. It includes a custom Gymnasium environment, training script, and visual validation script.

## Project Structure

- **`game/`**: Contains the refactored game logic (`dino_game.py`) allowing external control.
- **`envs/`**: Contains the Gymnasium Environment wrapper (`dino_env.py`).
- **`train.py`**: Script to train the DDQN model.
- **`test.py`**: Script to visually validate the trained model.
- **`chromedino.py`**: The original manual game (playable with keyboard).

## Installation

1.  **Prerequisites**: Python 3.8+ (Tested on Python 3.14).

2.  **Create a Virtual Environment**:

    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install stable-baselines3 gymnasium matplotlib pandas shimmy tqdm rich
    ```

## Usage

### 1. Manual Play

To play the game yourself (Space to jump, Down to duck):

```bash
python chromedino.py
```

### 2. Training the Model

To start training the agent:

```bash
python train.py
```

- This will create a `logs/` directory for Tensorboard and CSV logs.
- The model will saved as `dino_ddqn_model.zip`.
- A plot of the results (`training_results.png`) will be generated at the end.
- Hyperparameters (learning rate, epsilon decay, etc.) can be adjusted in `train.py`.

### 3. Testing / Validation

To watch the trained agent play:

```bash
python test.py
```

- This runs the game at 30 FPS.

## Environment Details

- **Action Space**: 3 Actions (0: Do Nothing, 1: Jump, 2: Duck).
- **Observation Space**: Array of 5 features:
  1.  Distance to next obstacle
  2.  Y position of next obstacle
  3.  Type of obstacle (-1 to 2)
  4.  Game speed
  5.  Dino Y position
