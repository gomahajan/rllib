import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gym
import numpy as np


class Transform():
    def get_state(self, observation):
        return observation.astype(np.float).ravel()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        h = 200
        self.w1 = nn.Linear(4, h)
        self.w2 = nn.Linear(h, 2)
        self.w3 = nn.Linear(h, 1)

    def forward(self, state):
        H = F.relu(self.w1(state))
        logp = self.w2(H)
        value = self.w3(H)
        prob = F.softmax(logp, dim=-1)
        return prob, value

def get_model():
    env = gym.make("Swimmer-v2")
    transform = Transform()

    pi = Policy()
    init.xavier_uniform_(pi.w1.weight, gain=np.sqrt(2))
    init.xavier_uniform_(pi.w2.weight)
    init.xavier_uniform_(pi.w3.weight)
    action_bias = 0
    return env, transform, pi, action_bias


