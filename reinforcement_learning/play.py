import gym
import time
import numpy as np
import torch
import random

import sys
sys.path.insert(0,'./utils')

from params import *
from model import ModelNet


# build environment
environment = gym.envs.make(GAME)
# length of input features, number of possible actions
input_features = environment.observation_space.shape[0]
actions = environment.action_space.n

#actions_meaning = environment.unwrapped.get_action_meanings()

# load Q network
Q_net = ModelNet(input_features,actions,SIZE_HIDDEN_LAYER)
Q_net.load_state_dict(torch.load(MODEL_PATH))

# start game :)
state = environment.reset()
environment.render()
# initialise previous state
previous_state = state

episode_result = 0
is_done = False
while not is_done:
    # compute Q
    if USE_DELTA_STATES:
        Q = Q_net(torch.from_numpy(state-previous_state).float()).detach().numpy()
    else:
        Q = Q_net(torch.from_numpy(state).float()).detach().numpy()
    # pick best action
    action = np.argmax(Q)

    # store previous state
    previous_state = state
    # next step
    state, reward, is_done, _ = environment.step(action)
    #print(actions_meaning[action], reward)
    print(action,reward)
    episode_result += reward

    # Render
    environment.render()
    time.sleep(PAUSE)

environment.close()
print('*********************************************')
print('Game Finished. Result:',episode_result)
print('*********************************************')
