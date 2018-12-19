import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


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

    def Save(self, optimizer, episode):
        torch.save({'episode': episode,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                   },'Taxi-v2_model_AC.pt')
        print('Save Model episode', episode)

    def Load(self, model_path):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eval()
        return optimizer


def select_action(state):
    state_encoded = encoded_state(encode_method, state)
    logits, state_value = test_net(state_encoded)
    probs = F.softmax(logits, dim=-1)
    m = Categorical(probs)
    action = m.sample()
    return torch.tensor([action.item()])



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
    HIDDEN_SIZE = 64
    num_of_states = 500
    in_channels = num_of_states
    num_of_actions = 6
    test_net = ActorCritic(in_channels, num_of_actions)
    test_net.Load('Taxi-v2_model_AC.pt')
    rewards = test_net.rewards
    encode_method = 'one_hot'
    num_episodes = 1000
    acc_reward = []

    main()

    print("#####################################")
    print("run " + str(num_episodes) + " episods")
    print("average reward: " + str(np.mean(acc_reward)))
    print("standard diveation:" + str(np.std(acc_reward)))
    print("#####################################")