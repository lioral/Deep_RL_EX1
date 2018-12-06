import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            # skip positive reward
            while int(self.memory[self.position].reward) == 1:
                self.position = (self.position + 1) % self.capacity

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(128, 2)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def Save(self, optimizer, episode, episode_durations, plt_duration, plt_reward, accumulate_reward, dir_path):
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_durations': episode_durations,
                    'accumulate_reward': accumulate_reward
                   }, dir_path + '/' + str(episode) + '.pt')
        plt_duration.savefig(dir_path + '/Duration.png')
        plt_reward.savefig(dir_path + '/Reward.png')
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


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    # screen = screen[:, 160:320]
    # view_width = 320

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def epsilon_decay_schedule(episode, Cons):
    EPISODE_DECAY = 500
    return Cons.EPS_DECAY_MIN + (Cons.EPS_DECAY_MAX - Cons.EPS_DECAY_MIN) * \
                    math.exp(-1. * episode / EPISODE_DECAY)



def select_action(state, Cons, policy_net, steps_done, episode, epsilon_history):
    sample = random.random()
    epsilon_decay = epsilon_decay_schedule(episode, Cons)
    eps_threshold = Cons.EPS_END + (Cons.EPS_START - Cons.EPS_END) * \
        math.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1
    epsilon_history.append(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def plot_durations(episode_durations, episode_reward):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.figure(3)
    plt.clf()
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    plt.title('Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def Plot_Epsilon_Loss(epsilon_history,loss_history):
    plt.figure(4)
    plt.clf()
    epsilon_t = torch.tensor(epsilon_history, dtype=torch.float)
    plt.title('Epsilon in episode...')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.semilogy(epsilon_t.numpy())

    plt.figure(5)
    plt.clf()
    loss_history_t = torch.tensor(loss_history, dtype=torch.float)
    plt.title('Loss in episode...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.semilogy(loss_history_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



def optimize_model(memory, Cons, policy_net, target_net, optimizer, loss_history):
    if len(memory) < Cons.BATCH_SIZE:
        return
    transitions = memory.sample(Cons.BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(Cons.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * Cons.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    loss_history.append(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

class Constants:

    BATCH_SIZE = 256
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY_MAX = 2000
    EPS_DECAY_MIN = 20
    TARGET_UPDATE = 10
    MAX_DURATION = 2000

def Main():
    load_models = True
    model_dir = "/Acrobot_Model"
    Model_to_Load = '/Dec_05_17_15_02'
    Model_num = '/150.pt'
    Cons = Constants()
    env = gym.make('Acrobot-v1').unwrapped

    # env.reset()
    # plt.figure()
    # plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #            interpolation='none')
    # plt.title('Example extracted screen')
    # plt.show()

    episode_durations = []
    accumulate_reward = []
    last_episode = 0

    policy_net = DQN().to(device)
    target_net = DQN().to(device)

    if load_models:
        output_dir_path = os.getcwd() + model_dir + Model_to_Load
        optimizer, episode_durations, last_episode, accumulate_reward = policy_net.Load(output_dir_path + Model_num)
    else:
        output_dir_path = prepare_model_dir(model_dir)
        optimizer = optim.RMSprop(policy_net.parameters())

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(10000)

    minimum_duration = float('inf')
    num_episodes = 3000


    for i_episode in range(last_episode, num_episodes):
        epsilon_history = []
        loss_history = []
        episode_reward = 0
        steps_done = 0
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, Cons, policy_net, steps_done, i_episode, epsilon_history)
            steps_done += 1
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            episode_reward += float(reward)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(memory, Cons, policy_net, target_net, optimizer, loss_history)

            if done or t == Cons.MAX_DURATION:
                episode_durations.append(t + 1)
                accumulate_reward.append(episode_reward)
                plot_durations(episode_durations, accumulate_reward)
                policy_net.Save(optimizer, i_episode, episode_durations,
                                plt.figure(2), plt.figure(3), accumulate_reward, output_dir_path)
                break
            if t % 500 == 0:
                Plot_Epsilon_Loss(epsilon_history,loss_history)
        # Update the target network
        if i_episode % Cons.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.render()
    env.close()


if __name__ == '__main__':
    Main()
    plt.ioff()
    plt.show()
    print('Complete')

