
import numpy as np
import gym
import random
import torch
import logging

import sys
sys.path.insert(0,'./utils')

from params import *
from dqn import DQN
from transition import Transition
from replay_memory import ReplayMemory


# build environment
environment = gym.envs.make(GAME)

#environment.unwrapped.get_action_meanings()

# length of input features
input_features = environment.observation_space.shape[0]
# number of possible actions
actions = environment.action_space.n

print('length input:',input_features)
print('number of possible actions:',actions)

# DQN agent
agent = DQN(input_features, actions,INITIAL_EPSILON)
# replay memory
replay_memory = ReplayMemory(REPLAY_MEMORY)

def main():
    # episode count
    episode = 0
    # episode result
    episode_result = 0
    # reset environment
    state = environment.reset()
    # define step loss
    loss = 0
    # initialise previous_state
    previous_state = state

    # simulation loop
    while episode < SIMULATION_END:

        # evaluate actions
        if USE_DELTA_STATES:
            Q = agent.evaluate(state-previous_state,episode)
        else:
            Q = agent.evaluate(state,episode)
        # pick best action
        action = np.argmax(Q)
        # next step
        observation, reward, terminal_state, _ = environment.step(action)

        # move state, previous = current, current = next
        previous_state = state
        state = observation

        # episode result
        episode_result += reward

        # optimize agent, if observation time is over
        if episode > OBSERVATION_TIME and episode < EXPLORATION_TIME + OBSERVATION_TIME:
            loss = agent.optimize_regular(replay_memory.sample(BATCH_SIZE),episode)

        # add experience to replay memory
        replay_memory.push(Transition(previous_state, action, reward, state, terminal_state))

        # if terminal state, reset environment
        if terminal_state:
            episode += 1
            print("Percentage = %f %%, Episode=%d, Result=%.5lf" % \
                   (100*episode/SIMULATION_END, episode, episode_result))
            episode_result = 0
            terminal_state = False
            environment.reset()

    # close environment
    environment.close()


if __name__ == "__main__":
    main()
