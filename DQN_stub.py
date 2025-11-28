import torch
import numpy as np
import torch.nn as nn
import random

# Pick top 50 best performing players from 2023
# 5 input features
# 5 x 50 -> 250 inputs
# outputs: softmax to get probabilities of picking each player, highest probability is picked

# TO DISCUSS: typically softmax is not used, lets just output raw Q values for each player and 
# pick the max instead (same outputs and better structure)

# same player twice?: large negative reward to incentivize not picking the same player twice
# Deep Q-Learning, DQN

# Make a deep Q learning model for predicting home runs, using 5 input features for the top 50 players overall.\
# using torch modules.

# general structure for deep Q learning model (note this was moved from explore_weekly_data.ipynb to DQN_stub.py)

class DeepQLearningModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: tuple[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))

        self.model = nn.Sequential(*layers)

    # can modify these forward pass to see how the model performs with different layers
    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward: int, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_layers, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.model = DeepQLearningModel(state_size, action_size, hidden_layers)
        self.target_model = DeepQLearningModel(state_size, action_size, hidden_layers)
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Potentially use Huber loss here
        self.replay_buffer = ReplayBuffer(10000)

    def update_target_model(self):
        self.target_model.model.load_state_dict(self.model.model.state_dict())

    def train_step(self):
        if len(self.replay_buffer) < 32:
            return

        batch = self.replay_buffer.sample(32)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states = torch.LongTensor(states).unsqueeze(1)
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.FloatTensor(s) for s in states]).squeeze(1)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        # next_states = torch.tensor(next_states, dtype=torch.float32)
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.FloatTensor(s) for s in next_states]).squeeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.model.forward(states).gather(1, actions)
        next_q_values = self.target_model.forward(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


'''
Environment for player selection using DeepQLearning
'''


class PlayerSelection:

    def __init__(self, player_features):

        assert player_features.shape == (50, 5)

        self.state_size = 250  # 5 features * 50 players
        self.action_size = 50  # 50 players to choose from
        self.max_pics = 10
        self.player_features = player_features
        self.agent = DQNAgent(self.state_size, self.action_size, self.hidden_layers)

        self.reset()

    def reset(self):
        '''
        Resets the environment
        '''
        self.picked = set()
        return self.get_state()  # should this always get a random bit of data?

    def step(self, action):
        '''
        Picks a player, if the player was already picked then there is a 
        large negative reward.
        '''
        reward = 0
        if action in self.picked:
            reward = -1000  # subject to change based on results
        else:
            reward += 1  # need to modify to get reward based on HR's that week
            self.picked.add(action)

        done = len(self.picked) >= self.max_picks

        next_state = self.get_state()
        return next_state, reward


class Env:
    def __init__(self, dataloader, num_weeks: int = 10):
        self.state_space = 250
        self.action_space = range(50)
        self.rewards_for_players = None
        self.data = list(dataloader)
        self.chosen_players = set()
        self.num_weeks = num_weeks
        self.week = 0
        self.initial_state = None

    def reset(self):
        self.week = 0
        self.chosen_players = set()
        self.initial_state, self.rewards_for_players = self.data[self.week]
        return self.initial_state, 0.0, False

    def step(self, player):  # action represents a chosen player here
        """
        player == action
        """
        reward = 0.0
        if player in self.chosen_players:
            # changed to -1 to equally fight against a good reward, (-1)
            # our reward system may need to be redone...
            reward -= 1.0  # also fixed the reward -= -5 bug causing the agent to favor the same players
        self.chosen_players.add(player)

        new_state = np.zeros_like(self.initial_state)
        reward += self.rewards_for_players[0][player]
        self.week += 1
        done = self.week >= len(self.data) or self.week >= self.num_weeks  # upper bounded by len(dataloader), e.g. 33 > 10

        if not done:
            new_state, self.rewards_for_players = self.data[self.week]

        return new_state, reward, done

