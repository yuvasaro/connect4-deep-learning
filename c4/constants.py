# Matrix dimensions (m x n)
M = 6
N = 7

# Players and game states
P1 = 1
P2 = 2
DRAW = 3
ONGOING = 0
PLACEHOLDER = -1

# Empty space on board
SPACE = 0

# Game framerate
FPS = 60

# Deep reinforcement learning
MEMORY_SIZE = 100_000     # size of memory buffer
BATCH_SIZE = 64           # batch size
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
TAU = 1e-3                # soft update parameter
E_DECAY = 0.995           # ε decay rate for ε-greedy policy
E_MIN = 0.01              # minimum ε value for ε-greedy policy

# Dense-only model settings
INPUT_SHAPE = (1, M * N)
BATCH_SHAPE = (BATCH_SIZE, M * N)

# Conv2D model settings
# INPUT_SHAPE = (1, M, N, 1)
# BATCH_SHAPE = (BATCH_SIZE, M, N, 1)

# Model name
MODEL_P1 = "connect4_model_p1.keras"
MODEL_P2 = "connect4_model_p2.keras"
