import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gym
import numpy as np
from torch.distributions import multivariate_normal


def get_action(self, parameters):
    m = Categorical(prob)
    action = m.sample()
    return action, m.log_prob(action)

class Transform():
    def get_state(self, observation):
        return observation.astype(np.float).ravel()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        h = 200
        self.w1 = nn.Linear(8, h)
        self.w2 = nn.Linear(h, 2)
        self.w3 = nn.Linear(h, 1)
        self.w4 = nn.Linear(h, 2)

    def forward(self, state):
        H = F.relu(self.w1(state))
        mu = self.w2(H)
        value = self.w3(H)
        sigma = self.w4(H)[0]
        covar = torch.diag(torch.exp(sigma))
        m = multivariate_normal.MultivariateNormal(mu, covar)
        return m, value

def get_model():
    env = gym.make("Swimmer-v2")
    transform = Transform()

    pi = Policy()
    init.xavier_uniform_(pi.w1.weight, gain=np.sqrt(2))
    init.xavier_uniform_(pi.w2.weight)
    init.xavier_uniform_(pi.w3.weight)
    init.xavier_uniform_(pi.w4.weight)
    action_bias = -0.5
    return env, transform, pi, action_bias


