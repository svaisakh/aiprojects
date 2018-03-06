import gym
import numpy as np

get_action = lambda obs: 0 if obs.dot(params[1:].T) < params[0] else 1

def sample(episodes=1, observe=True):
    epoch_reward = 0
    for episode in range(episodes):
        env.reset()
        done = False
        obs = env.observation_space.sample()
        episode_reward = 0
        while not done:
            if observe:  env.render()
            obs, r, done, _ = env.step(get_action(obs)) # take a random action
            episode_reward += r
                
        epoch_reward += episode_reward
    return epoch_reward

env = gym.make('CartPole-v1')

params = np.random.randn(5)
best_params = params
best_reward = -np.inf

while best_reward / 100 < 500:
    params = np.random.randn(5)
    epoch_reward = sample(100, False)

    if epoch_reward > best_reward:
        best_reward = epoch_reward
        best_params = params
        print('Average reward:', int(best_reward / 100), end='\r', flush=True)
        sample()