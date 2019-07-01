import gym
import time
import numpy as np
import torch
import random

from params import *



# build environment
environment = gym.envs.make(GAME)

# start game :)
state = environment.reset()
environment.render()
# initialise previous state
previous_state = state

episode_result = 0
is_done = False
while not is_done:

    # play randomly
    action = environment.action_space.sample()
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
