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

import cv2

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

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(64)

        self.head = nn.Linear(576, num_actions)


    def forward(self, x):
        x = F.relu(self.bn1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.bn2(self.maxpool2(self.conv2(x))))
        x = F.relu(self.bn3(self.maxpool3(self.conv3(x))))
        return self.head(x.view(x.size(0), -1))

    def Save(self, optimizer, episode, plt_reward,
             accumulate_reward, avg_accumulate_reward, STD_accumulate_reward, dir_path):
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_accumulate_reward': avg_accumulate_reward,
                    'STD_accumulate_reward': STD_accumulate_reward,
                    'accumulate_reward': accumulate_reward,
                   }, dir_path + '/' + str(episode) + '.pt')
        plt_reward.savefig(dir_path + '/Reward.png')
        print('Save Model episode', episode)

    def Load(self, model_path):
        optimizer = optim.RMSprop(self.parameters())

        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_episode = checkpoint["episode"] + 1
        accumulate_reward = checkpoint["accumulate_reward"]

        self.eval()

        return optimizer, last_episode, accumulate_reward

def get_screen(env, frame_stack):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
    frame_stack.append(screen.transpose((1, 2, 0)))# transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    # screen = screen[:, 160:320]
    # view_width = 320

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def select_action(state, policy_net):

    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


def VideoClip(frame_stack):
    video_name = 'Acrobot.avi'

    height, width, ch = frame_stack[0].shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 80, (width, height))

    for ii, image in enumerate(frame_stack):
        video.write(image)
        if ii > 800:
            break

    video.release()


def Main():
    Model_to_Load = '/Model_Wieghts.pt'
    env = gym.make('Acrobot-v1')
    GenVideoClip = False

    frame_stack = []
    episode_reward = 0
    num_actions = env.action_space.n

    policy_net = DQN(num_actions).to(device)

    policy_net.Load(os.getcwd() + Model_to_Load)

    accumulate_reward = []

    # Initialize the environment and state

    for episode in range(10):
        env.reset()
        current_screen = get_screen(env, frame_stack)
        state = current_screen
        episode_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(state, policy_net)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            episode_reward += int(reward)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, frame_stack)
            next_state = current_screen - last_screen

            # Move to the next state
            state = next_state

            if done:
                accumulate_reward.append(episode_reward)
                break
        print("Reward: ",episode_reward)

    print('Mean {:.2f}\t STD:{:.2f}'.format(np.mean(accumulate_reward), np.std(accumulate_reward)))
    env.render()
    env.close()
    if GenVideoClip:
        VideoClip(frame_stack)


if __name__ == '__main__':
    Main()
    plt.ioff()
    plt.show()
    print('Complete')



