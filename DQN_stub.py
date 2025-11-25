import torch
import numpy
import torch.nn as nn
import random

# Pick top 50 best performing players from 2023
# 5 input features
# 5 x 50 -> 250 inputs
# outputs: softmax to get probabilities of picking each player, highest probability is picked
# same player twice?: large negative reward to incentivize not picking the same player twice
# Deep Q-Learning, DQN

# Make a deep Q learning model for predicting home runs, using 5 input features for the top 50 players overall.\
# using torch modules.

# general structure for deep Q learning model (note this was moved from explore_weekly_data.ipynb to DQN_stub.py)
class DeepQLearningModel:
    def __init__(self, input_size, output_size, hidden_layers):

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

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
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
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(10000)

    def update_target_model(self):
        self.target_model.model.load_state_dict(self.model.model.state_dict())

    def train_step(self):
        if len(self.replay_buffer) < 32:
            return

        batch = self.replay_buffer.sample(32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.model.forward(states).gather(1, actions)
        next_q_values = self.target_model.forward(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

