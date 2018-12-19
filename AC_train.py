import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

class ActorCritic(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(in_features=in_channels, out_features=HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc_action = nn.Linear(in_features=HIDDEN_SIZE, out_features=out_channels)
        self.fc_value = nn.Linear(in_features=HIDDEN_SIZE, out_features=1)

        self.saved_actions = []
        self.rewards = []
        self.avg_20_rewards = []
        self.std_20_rewards = []

    def forward(self, x):
        x = self.relu(self.fc(x))
        action_scores = self.fc_action(x)
        state_values = self.fc_value(x)
        return action_scores, state_values

    def Save(self, optimizer, episode, dir_path=None):
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'Taxi-v2_model_AC.pt')

    def Load(self, model_path):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.eval()
        return optimizer

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
        fig1.savefig("AC_fig_reward_and_avg")
        fig2.savefig("AC_fig_avg_and_std")

    fig1.clf()
    fig2.clf()

def select_action(state):
    """
    get probabilities for any action and rdaw action from the probabilites.
    :param state:
    :return: action
    """
    state_encoded = np.zeros((1, num_of_states))
    state_encoded[0, int(state)] = 1

    state_encoded = torch.from_numpy(state_encoded).float()
    logits, state_value = model(state_encoded)
    probs = F.softmax(logits, dim=-1)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), probs, state_value))
    return action.item()

def finish_episode(steps_done):
    """
    calculate the value for any state in the episode, normalized it, compute the loss and update the net.
    the update is on the batch size depend on the episode.
    :param steps_done:
    :return:
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:                                                   # calculate the discount reward
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward.detach())                           # Compute actor loss
        value_losses.append(F.smooth_l1_loss(value.squeeze(1), torch.tensor([r])))  # Compute critic loss
    optimizer.zero_grad()                                                           # Clear previous gradients before backward pass
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()                                                                 # Run backward pass
    optimizer.step()                                                                # update
    del model.rewards[:]
    del model.saved_actions[:]

def main():
    """
    run num of episoddes, stop each episode after it finish or 30000 steps.
    :return:
    """
    for i_episode in range(num_episodes):
        cumulative_reward = 0
        state = env.reset()

        for t in count():  # Don't infinite loop while learning

            action = select_action(state)
            next_state, reward, finish, _ = env.step(action)
            model.rewards.append(reward)
            cumulative_reward += reward
            state = next_state
            if t > 30000:
                break
            if reward == 20:
                break
        episode_reward.append(cumulative_reward)
        acc_rewards.append(cumulative_reward)
        if i_episode % UPDATE == 0:
            avg_reward = np.mean(episode_reward[-20:])
            std_reward = np.std(episode_reward[-20:])
            model.avg_20_rewards.append(avg_reward)
            model.std_20_rewards.append(std_reward)
            if avg_reward > 5:
                model.Save(optimizer, i_episode)
                save_episode_to_print.append(i_episode)
                save_avg_rewars_to_print.append(avg_reward)
        finish_episode(i_episode)

    np.save("reward_arr", acc_rewards)
    for i in range(len(acc_rewards)):
        avg_reward_arr.append(np.mean(acc_rewards[i:i+20]))
        stdev_reward_arr.append(np.std(acc_rewards[i:i+20]))
    plot_rewards(acc_rewards, avg_reward_arr, stdev_reward_arr)

if __name__ == '__main__':
    env = gym.make("Taxi-v2")

    SavedAction = namedtuple('SavedAction', ['log_prob', 'prob', 'value'])
    HIDDEN_SIZE = 64                                                           # size of hidden layer
    num_of_states = 500
    in_channels = num_of_states
    num_of_actions = 6
    gamma = 1                                                                  # discount coefficient for rewards
    model = ActorCritic(in_channels, num_of_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()
    num_episodes = 400000
    UPDATE = 20

    save_episode_to_print = []
    save_avg_rewars_to_print = []
    avg_reward_arr = []
    stdev_reward_arr = []
    acc_rewards = []
    episode_reward = []

    main()




