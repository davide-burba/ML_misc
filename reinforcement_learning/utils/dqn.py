import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as util

import sys
sys.path.insert(0,'./utils')

from params import *
from model import ModelNet



class DQN(object):

    def __init__(self, input_features,actions,initial_epsilon):
        self.net = ModelNet(input_features,actions, SIZE_HIDDEN_LAYER)
        self.target = ModelNet(input_features,actions,SIZE_HIDDEN_LAYER)
        if MODEL_START:
            self.net.load_state_dict(torch.load(MODEL_START_PATH))
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=LEARNING_RATE)

        self.input_features = input_features
        self.actions = actions
        self.epsilon = initial_epsilon


    def evaluate(self, state, episode):

        # Q values array
        Q = None

        # calculate epsilon
        if episode > OBSERVATION_TIME and episode <= EXPLORATION_TIME + OBSERVATION_TIME:
            self.epsilon += (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORATION_TIME

        if random.random() > self.epsilon and OBSERVATION_TIME < episode:
            # fill Q values array using network
            Q = self.net(torch.from_numpy(state).float()).detach().numpy()
        else:
            # fill Q values array with zeroes
            Q = np.zeros(shape=self.actions)
            # select random action
            random_action = int(random.random() * self.actions)
            Q[random_action] = 1

        return Q


    def get_epsilon(self):
        return self.epsilon


    def calculate_current_Qs(self, transitions):
        # array of input states for all transitions
        input_states = np.zeros(shape=(len(transitions), self.input_features))

        # save states to calculate current Qs for all transitions
        for i in range(len(transitions)):
            # save input state for  evaluation
            input_states[i] = transitions[i].state

        # calculate current Qs
        current_Qs = self.net(torch.from_numpy(input_states).float())

        return current_Qs


    def calculate_target_Qs(self, transitions):
        # target Q arrays for all transitions
        target_Qs = np.zeros(shape=(len(transitions), self.actions))

        # calculate target Qs for all transitions
        for i in range(len(transitions)):
            state = transitions[i].state
            action = transitions[i].action
            reward = transitions[i].reward
            next_state = transitions[i].next_state
            terminal = transitions[i].terminal

            # calculate Qs for current state
            target_Qs[i] = self.net(torch.from_numpy(state).float()).detach().numpy()
            # calculate Qs for next state
            next_Q = self.target(torch.from_numpy(next_state).float()).detach().numpy()

            # calculate target Qs
            if terminal == True:
                # if state is terminal, target Q[action] is only reward
                target_Qs[i][action] = reward
            else:
                # if state is not terminal
                # target Qs for action is reward and discounted Q of next state
                target_Q = reward + GAMMA * np.max(next_Q)
                # calculate error
                error_Q = target_Q - target_Qs[i][action]
                # set target and clip error
                target_Qs[i][action] += error_Q if np.abs(error_Q) <= 1 else np.sign(error_Q)

        return target_Qs


    def optimize_regular(self, transitions,episode):
        # calculate target Q values
        target_Qs = self.calculate_target_Qs(transitions)

        # calculate current Q values
        current_Qs = self.calculate_current_Qs(transitions)

        # update network
        # zero gradients
        self.optimizer.zero_grad()
        # calculate loss
        loss = F.mse_loss(current_Qs, torch.from_numpy(target_Qs).float())
        # calculate gradients
        loss.backward()
        # clip gradients
        util.clip_grad_norm_(self.net.parameters(), 1.0)
        # change weights
        self.optimizer.step()

        # update target network
        if episode % TARGET_NET_UPDATE == 0:
            self.target.load_state_dict(self.net.state_dict())

        # save model weights
        if episode % SAVE_TIME == 0:
            torch.save(self.net.state_dict(), MODEL_PATH)

        return loss
