import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gym
import numpy as np

# model parameters
n = 4  # input dimension
h1 = 24
h2 = 36
k = 2


class CartTransform():
    def get_state(self, observation):
        return observation.astype(np.float).ravel()


class Policy(nn.Module):
    def __init__(self, n, h1, h2, k):
        super(Policy, self).__init__()
        self.w1 = nn.Linear(n, h1)
        self.w2 = nn.Linear(h1, h2)
        self.w3 = nn.Linear(h2, k)
        self.saved_logp = []
        self.saved_rewards = []
        self.saved_penalty = []

    def forward(self, state):
        h = self.w2(F.relu(self.w1(state)))
        logp = self.w3(F.relu(h))
        prob = F.sigmoid(logp)
        return prob, h


def get_model():
    env = gym.make("CartPole-v0")
    transform = CartTransform()

    pi = Policy(n, h1, h2, k)
    init.xavier_uniform_(pi.w1.weight, gain=np.sqrt(2))
    init.xavier_uniform_(pi.w2.weight, gain=np.sqrt(2))
    init.xavier_uniform_(pi.w3.weight)
    return env, transform, pi

