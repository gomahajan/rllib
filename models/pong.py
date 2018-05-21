import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gym
import numpy as np

# model parameters
n = 80 * 80  # input dimension
h1 = 200  # size of hidden layer
k = 2
action_bias = 2


class PongTransform():
    def __init__(self, n):
        self.prev_o = np.zeros(n)

    def get_state(self, observation):
        curr_o = self._transform(observation)
        state = curr_o - self.prev_o
        self.prev_o = curr_o
        return state

    def _transform(self, observation):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        observation = observation[35:195]  # crop
        observation = observation[::2, ::2, 0]  # downsample by factor of 2
        observation[observation == 144] = 0  # erase background (background type 1)
        observation[observation == 109] = 0  # erase background (background type 2)
        observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
        return observation.astype(np.float).ravel()


class Policy(nn.Module):
    def __init__(self, n, h1, k):
        super(Policy, self).__init__()
        self.w1 = nn.Linear(n, h1)
        self.w2 = nn.Linear(h1, k)
        self.saved_logp = []
        self.saved_rewards = []
        self.saved_penalty = []

    def forward(self, state):
        h = self.w1(state)
        logp = self.w2(F.relu(h))
        prob = F.softmax(logp, dim=1)
        return prob, h


class Value(nn.Module):
    def __init__(self, n, h1, k):
        super(Value, self).__init__()
        self.w1 = nn.Linear(n, h1)
        self.w2 = nn.Linear(h1, k)
        self.w3 = nn.Linear(k,1)

    def forward(self, state):
        h = self.w1(state)
        out = self.w3(F.relu(self.w2(F.relu(h))))
        return out

def get_model():
    env = gym.make("Pong-v0")
    transform = PongTransform(n)

    pi = Policy(n, h1, k)
    init.xavier_uniform_(pi.w1.weight, gain=np.sqrt(2))
    init.xavier_uniform_(pi.w2.weight, gain=np.sqrt(2))
    V = Value(n, h1, k)
    return env, transform, pi, action_bias, V


