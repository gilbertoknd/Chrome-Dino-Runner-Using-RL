#Training Hyperparameters
TOTAL_TIMESTEPS = 1000000  #Total steps for training
VALIDATION_FRACTION = 0.10  #Fraction of total timesteps

#DDQN Hyperparameters
LEARNING_RATE = 2.5e-4        # (Aumentado) O padrão ouro para DQN em imagens
BUFFER_SIZE = 100000          # (Fixo) 10% do total pode ser pouco se o treino for curto
LEARNING_STARTS = 10000       # (Aumentado) O agente precisa ver muitos obstáculos antes de começar
BATCH_SIZE = 128              # (Aumentado) Estabiliza o aprendizado com imagens
EXPLORATION_FRACTION = 0.15   # Explora por um pouco mais de tempo
EXPLORATION_FINAL_EPS = 0.01  # (Reduzido) No final, queremos quase 0 aleatoriedade
TARGET_UPDATE_INTERVAL = 2000 # (Aumentado) Rede alvo mais estável
TRAIN_FREQ = 4
GRADIENT_STEPS = 1           

#Environment Config
RENDER_FPS = 30
