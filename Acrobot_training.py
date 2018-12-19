import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def select_action(state, Cons, policy_net, steps_done, num_actions):
    # Check if learning stage began
    eps_threshold = 1.0 # init epsilon
    if steps_done <= Cons.PURE_EXPLORATION_STEPS:
        return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long), eps_threshold

    # Epsilon greedy
    sample = random.random()
    decay_factor = min(1.0, steps_done / Cons.STOP_EXPLORATION_STEPS)
    eps_threshold = Cons.EPS_START + decay_factor * (Cons.EPS_END - Cons.EPS_START)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long), eps_threshold


def plot_durations(accumulate_reward, avg_accumulate_reward, STD_accumulate_reward):
    fig = plt.figure(2)
    fig.clf()
    durations_t = np.arange(len(accumulate_reward))
    plt.title('Training - Accumulated reward and AVG reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t, accumulate_reward, color='r')
    plt.plot(durations_t, avg_accumulate_reward, color='b')
    plt.fill_between(durations_t, np.array(avg_accumulate_reward) - np.array(STD_accumulate_reward),
                     np.array(avg_accumulate_reward) + np.array(STD_accumulate_reward), color='b', alpha=0.2)
    plt.tight_layout()
    plt.legend(['Reward', 'Mean reward', 'STD'])


    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())




def optimize_model(memory, Cons, policy_net, target_net, optimizer, steps_done):
    if not memory.min_batch_load(Cons.BATCH_SIZE) or steps_done <= Cons.PURE_EXPLORATION_STEPS:
        return 0
    states_batch, actions_batch, rewards_batch, next_states_batch, not_done_mask, idx_batch, IS_weight = memory.sample(Cons.BATCH_SIZE)

    # Compute a mask of non-final states and concatenate the batch elements
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=device, dtype=torch.uint8)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
    states_batch = torch.cat(states_batch).cuda()
    actions_batch = torch.cat(actions_batch).cuda()
    rewards_batch = torch.cat(rewards_batch).cuda()
    next_states_batch = torch.cat(next_states_batch).cuda()
    not_done_mask = (1 - torch.tensor(not_done_mask, device=device).type(torch.cuda.FloatTensor))
    IS_weight = torch.tensor(IS_weight).unsqueeze(1).type(torch.cuda.FloatTensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(states_batch).gather(1, actions_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = target_net(next_states_batch).max(1)[0].detach() * not_done_mask
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * Cons.GAMMA) + rewards_batch

    # Compute Huber loss
    loss = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2) * IS_weight
    prios = loss + Cons.PRIOR_REG #update prios
    loss = loss.mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    for i in range(Cons.BATCH_SIZE):
        idx = idx_batch[i]
        memory.update(idx, prios[i].data.cpu().numpy())
    # update parameters
    optimizer.step()

    return 1

class Constants:

    BATCH_SIZE = 32
    GAMMA = 0.999
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY_MAX = 2000
    EPS_DECAY_MIN = 20
    TARGET_UPDATE = 500
    MAX_DURATION = 2000
    REPLAY_BUFFER = 100000
    LEARNING_RATE = 0.00025
    EPS_RMSPROP = 0.01
    ALPHA_RMSPROP = 0.95
    PURE_EXPLORATION_STEPS = 50000
    STOP_EXPLORATION_STEPS = 250000
    PRIOR_REG = 1e-5
    EPISODES_MEAN_REWARD = 10
    STOP_CONDITION = -100

def Main():
    save_model = True
    model_dir = "/Acrobot_Model"
    Cons = Constants()
    env = gym.make('Acrobot-v1')
    num_actions = env.action_space.n

    policy_net = DQN(num_actions).to(device)
    target_net = DQN(num_actions).to(device)


    output_dir_path = prepare_model_dir(model_dir)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=Cons.LEARNING_RATE,
                              eps=Cons.EPS_RMSPROP, alpha=Cons.ALPHA_RMSPROP)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = PrioritizedReplayMemory(Cons.REPLAY_BUFFER)

    best_avg_score = -np.inf
    i_episode = 0
    parameters_update_counter = 0

    accumulate_reward = []
    avg_accumulate_reward = []
    STD_accumulate_reward = [0]

    episode_reward = 0

    # Initialize the environment and state
    env.reset()
    # last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen

    for t in count():
        # stop criterion average reward above under 200
        if len(avg_accumulate_reward):
            if avg_accumulate_reward[-1] > Cons.STOP_CONDITION:
                policy_net.Save(optimizer, 'Model_Wieghts', plt.figure(2),
                                accumulate_reward, avg_accumulate_reward, STD_accumulate_reward,
                                output_dir_path)
                break
        # Select and perform an action
        action, eps_threshold = select_action(state, Cons, policy_net, t, num_actions)

        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_reward += float(reward)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen

        # Store the transition in memory with priority for the new sample
        current_Q_value = policy_net(state)[0][action]
        next_Q_value = target_net(next_state).detach().max(1)[0]
        target_value = reward + Cons.GAMMA * next_Q_value
        # Loss function of bellman eq to priority assessment
        loss_eq = np.abs(target_value.data - current_Q_value.squeeze().data)

        transition = Transition(state=state.cpu(), action=action.cpu(), next_state=next_state.cpu(),
                                reward=reward.cpu(), done=float(done))

        memory.push(transition, loss_eq)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        parameters_update_counter +=\
            optimize_model(memory, Cons, policy_net, target_net, optimizer, t)

        # Update the target network
        if parameters_update_counter % Cons.TARGET_UPDATE == 0 and not parameters_update_counter == 0:
            target_net.load_state_dict(policy_net.state_dict())


        if done:
            # restart the environment to new episode
            _ = env.reset()
            current_screen = get_screen(env)
            state = current_screen

            # statistics:
            accumulate_reward.append(episode_reward)
            i_episode += 1
            episode_reward = 0

            # Average reward and variance (standard deviation)
            if len(accumulate_reward) <= Cons.EPISODES_MEAN_REWARD:
                avg_accumulate_reward.append(np.mean(np.array(accumulate_reward)))
                if len(accumulate_reward) >= 2:
                    STD_accumulate_reward.append(np.std(np.array(accumulate_reward)))
            else:
                avg_accumulate_reward.append(np.mean(np.array(accumulate_reward[-10:])))
                STD_accumulate_reward.append(np.std(np.array(accumulate_reward[-10:])))

            # Check if average acc. reward has improved
            if avg_accumulate_reward[-1] > best_avg_score:
                best_avg_score = avg_accumulate_reward[-1]
                if save_model:
                    policy_net.Save(optimizer, i_episode, plt.figure(2),
                                    accumulate_reward, avg_accumulate_reward, STD_accumulate_reward,
                                    output_dir_path)
            # Update plot of acc. rewards every 20 episodes and print
            # training details
            if i_episode % 20 == 0:
                plot_durations(accumulate_reward, avg_accumulate_reward, STD_accumulate_reward)
                print('Episode {}\tAverage Reward: {:.2f}\tEpsilon: {:.4f}\t'.format(
                    i_episode, avg_accumulate_reward[-1], eps_threshold))

    env.render()
    env.close()


if __name__ == '__main__':
    Main()
    plt.ioff()
    plt.show()
    print('Complete')

