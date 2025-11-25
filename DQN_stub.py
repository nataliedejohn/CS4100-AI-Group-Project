import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import random
import pybaseball as pb
from datetime import datetime, timedelta
from pybaseball import batting_stats_range

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
        actions = torch.LongTensor(actions).unsqueeze(1)
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
        return self.get_state()
    
    def step(self, action):
        '''
        Picks a player, if the player was already picked then there is a 
        large negative reward.
        '''
        if action in self.picked:
            reward = -1000 # subject to change based on results
        else:
            reward += 1 #need to modify to get reward based on HR's that week
            self.picked.add(action)


        done = len(self.picked) >= self.max_picks

        next_state = self.get_state()
        return next_state, reward 


'''
Collects the HR data for each week in the 2023 season
'''
# Still need to implement this into the player selection reward portion for accurate player rewards. 
weeks = []
start = datetime(2023, 3, 30)  
end = datetime(2023, 10, 1) 

week_start = start

while week_start < end:
    week_end = week_start + timedelta(days=6)
    df = batting_stats_range(str(week_start.date()), str(week_end.date()))
    df['week_start'] = week_start.date()
    df['week_end'] = week_end.date()
    weeks.append(df[['Name', 'playerid', 'HR', 'week_start', 'week_end']])
    week_start += timedelta(days=7)

full_weekly_hr = pd.concat(weeks, ignore_index=True)
full_weekly_hr.to_csv('weekly_hr_2023.csv', index=False)