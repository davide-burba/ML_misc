
# set game
#GAME = 'CartPole-v1'
#GAME = 'LunarLander-v2'
GAME = 'Pong-ram-v0'
#GAME = 'Tennis-ram-v0'

# flag: use delta states?
USE_DELTA_STATES = False

# model path
MODEL_PATH = "weights_Q_network.pickle"
# start from existing weights
MODEL_START = False
MODEL_START_PATH = "weights_Q_network.pickle"
# render pause
PAUSE = 0.02

### Training parameters
# episodes before training
OBSERVATION_TIME = 0
# training episodes
EXPLORATION_TIME = 5000
# simulation episodes (total)
SIMULATION_END = OBSERVATION_TIME + EXPLORATION_TIME

# size of replay memory
REPLAY_MEMORY = int(1e5)
# size hidden layers
SIZE_HIDDEN_LAYER = 32
# batch size
BATCH_SIZE = 256
# "future horizon" length
GAMMA = 0.99
# starting value of epsilon
INITIAL_EPSILON = 0.8
# final value of epsilon
FINAL_EPSILON = 0


# learning rate
LEARNING_RATE = 1e-3
# target net update
TARGET_NET_UPDATE = 500
# saving frequency
SAVE_TIME = 20
