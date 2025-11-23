import torch
import numpy

# Pick top 50 best performing players from 2023
# 5 input features
# 5 x 50 -> 250 inputs
# outputs: softmax to get probabilities of picking each player, highest probability is picked
# same player twice?: large negative reward to incentivize not picking the same player twice
# Deep Q-Learning, DQN


