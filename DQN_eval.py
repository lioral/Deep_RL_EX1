import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim


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


def select_action(state):
    state_encoded = encoded_state(encode_method, state)
    policy_action = test_net(state_encoded).max(1)[1].view(1, 1)
    return torch.tensor([policy_action.item()])


def encoded_state(encode_method,state_batch):
    if encode_method is 'one_hot':
        states_encoded = np.zeros((1, num_of_states))
        states_encoded[0, state_batch] = 1
        return torch.tensor(states_encoded).float()

def main():
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        cumulative_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(state)
            next_state, reward, finish, _ = env.step(action.item())
            cumulative_reward += reward
            state = next_state
            if finish:
                break
        acc_reward.append(cumulative_reward)


if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    dropout1 = 0
    dropout2 = 0
    HIDDEN_SIZE = 64
    num_of_states = 500
    in_channels = num_of_states
    num_of_actions = 6
    test_net = DQN(in_channels, num_of_actions)
    test_net.Load('Taxi-v2_model_DQN.pt')
    encode_method = 'one_hot'
    acc_reward = []
    num_episodes = 1000

    main()

    print("#####################################")
    print("run " + str(num_episodes) + " episods")
    print("average reward: " + str(np.mean(acc_reward)))
    print("standard diveation:" + str(np.std(acc_reward)))
    print("#####################################")