import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import random



# Constants and Helper Classes

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# End Constants


def Main():
    # init environment
    env = gym.make("Taxi-v2")
    env.reset()
    env.render()

    # init FCN
    # device = torch.device("cuda:0")
    # model = torch.nn






if __name__ == 'main':
    Main()