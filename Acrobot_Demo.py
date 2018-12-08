import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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

    def Save(self, optimizer, episode, episode_durations, plt_duration, dir_path):
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_durations': episode_durations
                   }, dir_path + '/' + str(episode) + '.pt')
        plt_duration.savefig(dir_path + '/Duration.png')
        print('Save Model episode', episode)


    def Load(self, model_path):
        optimizer = optim.RMSprop(self.parameters())

        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode_durations = checkpoint['episode_durations']

        self.eval()

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


def select_action(state, Cons, policy_net):

    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


class Constants:

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10


def Main():
    model_dir = "/Acrobot_Model"
    Model_to_Load = '/Dec_05_17_15_02/692.pt'
    Cons = Constants()
    env = gym.make('Acrobot-v1').unwrapped

    episode_durations = []

    policy_net = DQN().to(device)

    policy_net.Load(os.getcwd() + model_dir + Model_to_Load)

    accumulate_reward = 0

    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state, Cons, policy_net)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Move to the next state
        state = next_state
        accumulate_reward += reward

        if done:
            episode_durations.append(t + 1)
            break
    print("Reward: ",accumulate_reward)
    env.render()
    env.close()


if __name__ == '__main__':
    Main()
    plt.ioff()
    plt.show()
    print('Complete')



