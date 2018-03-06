import gym
import numpy as np
import os
from time import time

DIR_CHECKPOINTS = os.path.join(os.path.split(os.getcwd())[0], 'Checkpoints')
def relu(x):
    x[x < 0] = 0
    return x

get_params = lambda: np.random.randn(5 * h + 1) * 100

def get_action(obs):
    Wxh = params[:3 * h].reshape((3, h))
    bh = params[3 * h: 4 * h]
    x = np.tanh(obs.dot(Wxh) + bh)
    Why = params[4 * h: 5 * h]
    by = params[-1]
    return np.array([x.dot(Why.T) + by])

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

env = gym.make('Pendulum-v0')
h = 10
episodes = 10

params = get_params()
best_reward = -np.inf
if os.path.exists(os.path.join(DIR_CHECKPOINTS, 'pendulum-params.npy')):
    params = np.load(os.path.join(DIR_CHECKPOINTS, 'pendulum-params.npy'))
    best_reward = np.load(os.path.join(DIR_CHECKPOINTS, 'pendulum-best.npy'))
    print('Previous best')
    sample()
    print('Now training')

best_params = params
std = 100
start_time = time()
for _ in range(100):#while best_reward / episodes < 0:
    print('std:', np.round(std, 2), end='\r', flush=True)
    params = params + get_params() / std
    epoch_reward = sample(episodes, False)
    std *= (1 - 1e-2)

    if epoch_reward > best_reward:
        best_reward = epoch_reward
        best_params = params
        print('Average reward:', int(best_reward / episodes))
        np.save(os.path.join(DIR_CHECKPOINTS, 'pendulum-params'), params)
        np.save(os.path.join(DIR_CHECKPOINTS, 'pendulum-best'), best_reward)
        #sample()
        std = 100

print(time() - start_time)