import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

env = gym.make("Taxi-v2")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.do1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(in_features=HIDDEN_SIZE, out_features=out_channels)
        self.do2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.do1(self.relu(self.fc1(x)))
        x = self.do2(self.fc2(x))
        return x

    def Save(self, optimizer, episode, episode_durations, accumulate_reward, dir_path=None):
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_durations': episode_durations,
                    'accumulate_reward': accumulate_reward
                   },'Taxi-v2_model_DQN.pt')
        print('Save Model episode', episode)

    def Load(self, model_path):
        optimizer = optim.RMSprop(self.parameters())
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode_durations = checkpoint['episode_durations']
        last_episode = checkpoint["episode"] + 1
        accumulate_reward = checkpoint["accumulate_reward"]
        self.eval()
        return optimizer, episode_durations, last_episode, accumulate_reward

class ReplayMemory(object):
    """
    Memory , save tarnsitions.
    """
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

def select_action(state):
    """
    draw number [0,1] if it smaller than eps_threshold do random action , else use net to decide action.
    :param state:
    :return: next state
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state_encoded = encoded_state(encode_method, state)
            policy_action = policy_net(state_encoded).max(1)[1].view(1, 1)
            return torch.tensor([policy_action.item()])
    else:
        return torch.tensor([random.randrange(6)])


def plot_rewards(reward_arr, avg_reward_arr, stdev_reward_arr, save=True):
    """
    plot 2 plots:
    a. reward VS avg. reward
    b. avg. rewards VS std. rewards
    """
    fig1 = plt.figure(1)
    # rewards + average rewards
    plt.plot(reward_arr, color='b', alpha=0.3)
    plt.plot(avg_reward_arr, color='b')
    plt.xlabel('# episodes')
    plt.ylabel('Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes')
    plt.legend(['Acc. episodic reward', 'Avg. acc. episodic reward'])
    plt.tight_layout()

    # average rewards + stdevs
    fig2 = plt.figure(2)
    plt.plot(avg_reward_arr, color='b')
    plt.fill_between(range(len(avg_reward_arr)), np.array(avg_reward_arr) - np.array(stdev_reward_arr),
                     np.array(avg_reward_arr) + np.array(stdev_reward_arr), color='b', alpha=0.2)
    plt.xlabel('# episodes')
    plt.ylabel('Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes')
    plt.legend(['Avg. acc. episodic reward', 'Stdev envelope of acc. episodic reward'])
    plt.tight_layout()

    plt.pause(0.01)
    if save:
        fig1.savefig("DQN_fig_reward_and_avg")
        fig2.savefig("DQN_fig_avg_and_std")
    fig1.clf()
    fig2.clf()


# out = [target , pass-on-taxi , taxi-row , taxi-col]
def decode_16_taxi_v2(state_batch):
    batch_size = len(state_batch)
    out = []
    target = []
    target_drop = state_batch % 4
    state_batch = state_batch // 4
    target_pick = state_batch % 5
    pass_in_taxi = target_pick // 4
    for i in range(batch_size):
        if pass_in_taxi[i] == 1:
            target.append(target_drop[i].item())
        else:
            target.append(target_pick[i].item())
    target = torch.tensor(target).float()
    out.append(target)
    out.append(pass_in_taxi)
    state_batch = state_batch // 5
    out.append(state_batch % 5)
    state_batch = state_batch // 5
    out.append(state_batch)
    return out

# out = [target , pass , taxi-row , taxi-col]
def decode_19_taxi_v2(state_batch):
    out = []
    out.append(state_batch % 4)
    state_batch = state_batch // 4
    out.append(state_batch % 5)
    state_batch = state_batch // 5
    out.append(state_batch % 5)
    state_batch = state_batch // 5
    out.append(state_batch)
    return out


def encoded_state(encode_method,state_batch):
    """
    get state in number and change it to 0 and 1 vector
    :param encode_method: the method we chose to encode
    :param state_batch: state's transitions from the reaply memory
    :return: the states which encodded
    """
    if encode_method is 'one_hot':
        batch_size = len(state_batch)
        states_encoded = np.zeros((batch_size, in_channels))
        states_encoded[np.arange(batch_size), state_batch] = 1
        return torch.tensor(states_encoded).float()

    if encode_method is '16_hot':
        batch_size = len(state_batch)
        dest_encoded = np.zeros((batch_size, dest_encoded_size))
        pass_encoded = np.zeros((batch_size, pass_encoded_size))
        taxirow_encoded = np.zeros((batch_size, taxirow_encoded_size))
        taxicol_encoded = np.zeros((batch_size, taxicol_encoded_size))
        state_batch_decoded = decode_16_taxi_v2(state_batch)
        dest_encoded[np.arange(batch_size), state_batch_decoded[0].int()] = 1
        pass_encoded[np.arange(batch_size), state_batch_decoded[1].int()] = 1
        taxirow_encoded[np.arange(batch_size), state_batch_decoded[2].int()] = 1
        taxicol_encoded[np.arange(batch_size), state_batch_decoded[3].int()] = 1
        state_batch_encoded = np.append(np.append(dest_encoded, pass_encoded, axis=1), np.append(taxirow_encoded, taxicol_encoded, axis=1), axis=1)
        return torch.tensor(state_batch_encoded).float()

    if encode_method is '19_hot':
        batch_size = len(state_batch)
        dest_encoded = np.zeros((batch_size, dest_encoded_size))
        pass_encoded = np.zeros((batch_size, pass_encoded_size))
        taxirow_encoded = np.zeros((batch_size, taxirow_encoded_size))
        taxicol_encoded = np.zeros((batch_size, taxicol_encoded_size))
        state_batch_decoded = decode_19_taxi_v2(state_batch)
        dest_encoded[np.arange(batch_size), state_batch_decoded[0].int()] = 1
        pass_encoded[np.arange(batch_size), state_batch_decoded[1].int()] = 1
        taxirow_encoded[np.arange(batch_size), state_batch_decoded[2].int()] = 1
        taxicol_encoded[np.arange(batch_size), state_batch_decoded[3].int()] = 1
        state_batch_encoded = np.append(np.append(dest_encoded, pass_encoded, axis=1), np.append(taxirow_encoded, taxicol_encoded, axis=1), axis=1)
        return torch.tensor(state_batch_encoded).float()


def optimize_model():
    """
    optimized the model by backpropogation
    :return: None
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)

    state_batch_encoded = encoded_state(encode_method, state_batch)
    state_action_values = policy_net(state_batch_encoded).gather(1, action_batch.unsqueeze(1)) # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    next_state_values = torch.zeros(BATCH_SIZE) # Compute V(s_{t+1}) for all next states.
    non_final_next_states = encoded_state(encode_method, non_final_next_states)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() # if it's the last state there is no t+1 time stamp
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()# Compute the expected Q values
    huber_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # Compute Huber loss
    optimizer.zero_grad() # Clear previous gradients before backward pass
    all_layer1_params = torch.cat([x.view(-1) for x in policy_net.fc1.parameters()])
    all_layer2_params = torch.cat([x.view(-1) for x in policy_net.fc2.parameters()])
    l1_layer1 = lambda1 * torch.norm(all_layer1_params, 1) # calculate L1 regularization first layer
    l1_layer2 = lambda1 * torch.norm(all_layer2_params, 1) # calculate L1 regularization first layer
    loss = huber_loss + l1_layer1 + l1_layer2
    loss.backward() # Run backward pass
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1) # Clip the Huber loss between [-1 , 1] - prevent big gradients
    optimizer.step() # update


def EncodedPram(encode_method):
    """
    define parameters which are suitable to the encode method (in channels is the input size of the net)
    :param encode_method:
    :return: in_channels, dest_encoded_size, pass_encoded_size, taxirow_encoded_size, taxicol_encoded_size
    """
    dest_encoded_size = 4
    taxirow_encoded_size = 5
    taxicol_encoded_size = 5
    if encode_method is 'one_hot':
        in_channels = 500
    elif encode_method is '16_hot':
        pass_encoded_size = 2
        in_channels = dest_encoded_size + pass_encoded_size + taxirow_encoded_size + taxicol_encoded_size
    elif encode_method is '19_hot':
        pass_encoded_size = 5
        in_channels = dest_encoded_size + pass_encoded_size + taxirow_encoded_size + taxicol_encoded_size
    return in_channels, dest_encoded_size, pass_encoded_size, taxirow_encoded_size, taxicol_encoded_size




def InitMemory():
    """
    initializetion the memory
    """
    state = torch.tensor([int(env.reset())])
    for i in range(memory.capacity):
        action = torch.tensor([random.randrange(6)])
        next_state, reward, finish, _ = env.step(action.item())
        next_state = torch.tensor([next_state])
        reward = torch.tensor([reward])
        memory.push(state, action, next_state, reward)
        if finish:
            state = torch.tensor([int(env.reset())])
        else:
            state = next_state


def DefineOptimizer(choose_optimizer):
    """
    define optimizer to the net
    :param choose_optimizer:
    :return: optimizer
    """
    if choose_optimizer is 'RMSProp':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=LR_RATE)
    elif choose_optimizer is 'Adam':
        optimizer = optim.Adam(policy_net.parameters(), lr=LR_RATE)
    elif choose_optimizer is 'SGD':
        optimizer = optim.SGD(policy_net.parameters(), lr=LR_RATE)
    return optimizer


def main():
    """
    run num_episodes every episode run untill it get "finish" flag from the env
    :return: None
    :SAVE: model , figures and rewards
    """
    for i_episode in range(num_episodes):
        state = torch.tensor([int(env.reset())])      # Initialize the environment and state
        cumulative_reward = 0
        for t in count():
            action = select_action(state)              # Select and perform an action
            next_state, reward, finish, _ = env.step(action.item())
            cumulative_reward += reward
            next_state = torch.tensor([next_state])
            reward = torch.tensor([reward])
            memory.push(state, action, next_state, reward) # Store the transition in memory
            optimize_model()
            if finish:
                break
            state = next_state
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        reward_arr.append(cumulative_reward)
        if i_episode > 9:
            avg_reward_arr.append(np.mean(reward_arr[i_episode-9:]))
            stdev_reward_arr.append(np.std(reward_arr[i_episode-9:]))
    np.save("reward_arr", reward_arr)
    np.save("avg_reward_arr", avg_reward_arr)
    np.save("stdev_reward_arr", stdev_reward_arr)
    plot_rewards(reward_arr, avg_reward_arr, stdev_reward_arr)
    policy_net.Save(optimizer, num_episodes, episode_durations=0, accumulate_reward=0)


if __name__ == '__main__':

    MEM_SIZE = 20000  # size of reaply memory
    HIDDEN_SIZE = 64  # size of hidden layer
    BATCH_SIZE = 512  # how many transition the net see every step
    GAMMA = 0.999  # discount coefficient for rewards
    EPS_START = 1  # start value of epsilon
    EPS_END = 0.05  # end value of epsilon
    EPS_DECAY = 200000  # decay coefficient for epsilon
    TARGET_UPDATE = 500  # indicate for how many steps before we update the weight of the target net from the policy net
    LR_RATE = 0.0005  # learmimg rate
    num_episodes = 70000  # number of episodes for the traim
    encode_method = 'one_hot'  # encodded option , can be one_hot,19_hot,16_hot
    num_of_actions = 6
    dropout1 = 0  # dropout on the hidden layer
    dropout2 = 0  # dropout on the output layer
    lambda1 = 0  # factor for L1 regularization
    choose_optimizer = 'RMSProp'  # Choose optimizer , can be RMSProp,Adam,SGD

    in_channels, dest_encoded_size, pass_encoded_size, taxirow_encoded_size, taxicol_encoded_size = EncodedPram(
        encode_method)
    policy_net = DQN(in_channels, num_of_actions)
    target_net = DQN(in_channels, num_of_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = DefineOptimizer(choose_optimizer)

    memory = ReplayMemory(MEM_SIZE)
    InitMemory()
    epsilon = EPS_START
    reward_arr = []
    avg_reward_arr = []
    stdev_reward_arr = []
    steps_done = 0
    cumulative_reward = 0

    main()

    print('Complete')
    env.render()
    env.close()

