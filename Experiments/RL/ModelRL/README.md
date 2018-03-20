# Model Based RL

A small experiment.

First, we train a predictor network which predicts the next state.

Then, we train a reward network which predicts the reward for a given state.

Finally, we prepare a policy roll-out for T timesteps by maximizing the total reward for the next T steps conditioned on the action probabilities.

[Follow my Trello Board](https://trello.com/c/Uy5gj07b/2-prednet-reinforcement-learning-through-predictive-analysis)