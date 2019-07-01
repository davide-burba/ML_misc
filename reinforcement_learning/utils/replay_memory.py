import random

"""
Replay memory buffer
"""

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # add new transition to replay buffer
    def push(self, transition):
        # if buffer is not full extend it
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # add new transition
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    # sample transitions from replay buffer
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
