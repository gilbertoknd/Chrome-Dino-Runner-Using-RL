import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
import time
from game.dino_game import DinoGame

class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = DinoGame()
        
        #Action Space: 0: Do Nothing, 1: Jump, 2: Duck
        self.action_space = spaces.Discrete(3)
        
        #Observation Space:
        #[distance_to_obstacle, obstacle_y, obstacle_type, game_speed, dino_y]

        #Distance: 0 to 1200
        #Obstacle Y: 0 to 600
        #Type: -1 to 2
        #Speed: 0 to 100
        #Dino Y: 0 to 600
        
        low = np.array([0, 0, -1, 0, 0], dtype=np.float32)
        high = np.array([1200, 600, 2, 100, 600], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, terminated, info = self.game.step(action)
        truncated = False #Endless game
        
        if self.render_mode == "human":
            self.render()
            
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        self.game.render()
        #Cap FPS if needed, but training usually wants max speed.
        #Using clock in human mode
        self.game.clock.tick(30) if hasattr(self.game, 'clock') else None 
        pygame.display.update() 

    def close(self):
        self.game.close()
