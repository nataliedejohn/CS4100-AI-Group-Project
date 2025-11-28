import random

import numpy as np
import torch
from DQN_stub import Env, DQNAgent

'''
This is a Relatively small MDP...

    -> action space = 50
    -> state space = 250 (PA2 had something like ~2b)
    -> network is small, 250x150x150x50 -> (250x150+150) + (150x150+150) + (150x50+50) = 67,850 parameters
    
    -> network prefers lower decay rates to pick actions from its optimal policy, it leatns optimal quick
    -> rewards CURRENTLY are super simple, 0 or 1, then -5 for repeated player choice
    
    -> we can either constrain the model size and training time to give it a harder challenge
    -> or we can explode the state space into the millions to get the same effect
    
    -> we also need to take our dataset into consideration
    -> fairly small
    -> used for both training and testing
    
'''
print_metrics = True
episode_print_interval = 100
test_episodes = 10000

# model
input_size = 250
output_size = 50
hidden_layers = (150, 150)  # largely subject to change, play around


def train_dqn(trainloader, num_episodes: int, decay: float, epsilon: float = 1.0, ):
    env = Env(dataloader=trainloader, num_weeks=len(trainloader))
    agent = DQNAgent(input_size, output_size, hidden_layers)

    update_frequency = int(100)
    dt = 0
    print("Training...")

    reward_counts = []
    action_distribution = [0 for _ in env.action_space]

    for episode in range(num_episodes + 1):
        episode_reward = 0
        curr_state, reward, done = env.reset()
        while not done:
            prob = random.random()
            if prob < epsilon:
                action: int = random.choice(env.action_space)
            else:
                q_values = agent.model(curr_state)
                action = torch.argmax(q_values).item()
            next_state, reward, done = env.step(action)
            # state, action, reward, next_state, done
            agent.replay_buffer.push(curr_state, action, reward, next_state, done)
            agent.train_step()
            curr_state = next_state
            # update metrics tracking

            action_distribution[action] = action_distribution[action] + 1
            episode_reward += reward

            dt += 1
            if dt >= update_frequency:
                agent.update_target_model()
                dt = 0
        reward_counts.append(episode_reward)
        epsilon = epsilon * decay

        # debug print
        if (episode / episode_print_interval).is_integer() and episode > 1:
            last_slice = reward_counts[-episode_print_interval:]
            if print_metrics:
                print(
                    f"avg reward from {episode - episode_print_interval}-{episode}: {sum(last_slice) / len(last_slice)}")
    print(f"action distribution: {action_distribution}") if print_metrics else None
    torch.save(agent.model.state_dict(), f'dqn_{num_episodes}_{decay}.pth')


def test_dqn(dataloader, num_episodes: int, decay: float):
    print(f"Testing DQN num_episodes={num_episodes}, decay={decay}...")
    env = Env(dataloader, len(dataloader))
    agent = DQNAgent(input_size, output_size, hidden_layers)
    agent.model.load_state_dict(torch.load(f'dqn_{num_episodes}_{decay}.pth'))

    total_rewards = []
    action_distribution = [0 for _ in env.action_space]

    for episode in range(test_episodes):  # some value to run testing by
        curr_state, reward, done = env.reset()
        reward_for_episode = 0

        while not done:
            with torch.no_grad():
                q_values = agent.model(curr_state)
                action = torch.argmax(q_values).item()
            next_state, reward, done = env.step(action)
            reward_for_episode += reward

            action_distribution[action] = action_distribution[action] + 1

            curr_state = next_state
        total_rewards.append(reward_for_episode)
    print(f"action distribution: {action_distribution}") if print_metrics else None
    print(f"total reward over {test_episodes} episodes: {sum(total_rewards)}")
    print(f"Average reward over {test_episodes} episodes: {sum(total_rewards) / len(total_rewards)}")
