import os
from time import strftime, localtime
import random
import numpy as np
from collections import namedtuple

from SumTree import SumTree

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))



def prepare_model_dir(WorkDir):
    # Create results directory
    result_path = os.getcwd() + WorkDir + '/' + strftime('%b_%d_%H_%M_%S', localtime())
    os.mkdir(result_path)

    return result_path

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
# using for priority replay memory
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        if self.tree[right] > 0:
            return self._retrieve(right, s - self.tree[left])
        else:  # Bug - parent size is not sum of kids
            return self._retrieve(left, self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.sample_enable = False
        self.position = 0


    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def push(self, sample, error):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.position = (self.position + 1) % self.capacity

    def min_batch_load(self, batch_size):
        if self.position >= batch_size:
            self.sample_enable = True
        return self.sample_enable

    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)

            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            states.append(data[0])
            actions.append(data[1])
            next_states.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return states, actions, rewards, next_states, dones, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

