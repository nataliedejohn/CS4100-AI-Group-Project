import random

import numpy as np
import torch
from DQN_stub import Env, DQNAgent
def train_dqn(dataloader, num_episodes: int, decay: float, epsilon: float = 1.0,
              input_size: int = 250, output_size: int = 50,
              hidden_layer_sizes: tuple[int] = (150, 150)):
    env = Env(dataloader=dataloader, num_weeks=len(dataloader))
    agent = DQNAgent(input_size, output_size, hidden_layer_sizes)

    update_frequency = int(num_episodes / 10)
    dt = 0

    for episode in range(num_episodes):
        curr_state, reward, done = env.reset()
        while not done:
            prob = random.random()
            if prob < epsilon:
                action: int = random.choice(env.action_space)
            else:
                with torch.no_grad(): # the aha method -_-
                    q_values = agent.model(curr_state)
                    action = torch.argmax(q_values).item()
            next_state, reward, done = env.step(action)
            # state, action, reward, next_state, done
            agent.replay_buffer.push(curr_state, action, reward, next_state, done)
            agent.train_step()
            curr_state = next_state
        epsilon = epsilon * decay
        dt += 1
        if dt >= update_frequency:
            agent.update_target_model()
            dt = 0
    torch.save(agent.model.state_dict(), f'dqn_{num_episodes}_{decay}.pth')
