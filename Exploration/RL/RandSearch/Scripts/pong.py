import gym
import numpy as np
import os

DIR_CHECKPOINTS = os.path.join(os.path.split(os.getcwd())[0], 'Checkpoints')
def softmax(x):
    x -= x.max(-1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(-1, keepdims=True)

def relu(x):
    x[x < 0] = 0
    return x

get_params = lambda: np.random.randn((n + 1 + a) * h + a) * 100

def get_action(obs):
    Wxh = params[:n * h].reshape((n, h))
    bh = params[n * h: (n + 1) * h]
    x = np.tanh(obs.dot(Wxh) + bh)
    Why = params[(n + 1) * h: (n + 1 + a) * h].reshape((h, a))
    by = params[(n + 1 + a) * h:]
    x = softmax(x.dot(Why) + by) 
    return np.random.choice(a, p=x)

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

env = gym.make('Pong-ram-v0')
n = env.observation_space.shape[0]
a = env.action_space.n
h = 10
episodes = 1

params = get_params()
best_reward = -np.inf
if os.path.exists(os.path.join(DIR_CHECKPOINTS, 'pong-params.npy')):
    params = np.load(os.path.join(DIR_CHECKPOINTS, 'pong-params.npy'))
    best_reward = np.load(os.path.join(DIR_CHECKPOINTS, 'pong-best.npy'))
    print('Previous best')
    sample()
    print('Now training')

best_params = params
std = 100
while best_reward / episodes < 0:
    print('std:', np.round(std, 2), end='\r', flush=True)
    params = params + get_params() / std
    epoch_reward = sample(episodes, False)
    std *= (1 - 1e-2)

    if epoch_reward > best_reward:
        best_reward = epoch_reward
        best_params = params
        print('Average reward:', int(best_reward / episodes))
        np.save(os.path.join(DIR_CHECKPOINTS, 'pong-params'), params)
        np.save(os.path.join(DIR_CHECKPOINTS, 'pong-best'), best_reward)
        #sample()
        std = 100