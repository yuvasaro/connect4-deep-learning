# Matrix dimensions (m x n)
M = 6
N = 7

# Players and game states
P1 = 1
P2 = 2
DRAW = 3
ONGOING = 0

# Empty space on board
SPACE = 0

# Game framerate
FPS = 60

# Deep reinforcement learning
MEMORY_SIZE = 100_000     # size of memory buffer
MINIBATCH_SIZE = 20       # mini-batch size
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
TAU = 1e-3                # soft update parameter
E_DECAY = 0.995           # ε decay rate for ε-greedy policy
E_MIN = 0.01              # minimum ε value for ε-greedy policy

# Model name
MODEL = "connect4_model.keras"
