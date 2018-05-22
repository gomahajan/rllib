import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gym
import numpy as np


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
    def __init__(self):
        super(Policy, self).__init__()
        self.w1 = nn.Linear(6400, 200)
        self.w2 = nn.Linear(200, 2)
        self.w3 = nn.Linear(200, 1)

    def forward(self, state):
        H = F.relu(self.w1(state))
        logp = self.w2(H)
        value = self.w3(H)
        prob = F.softmax(logp, dim=-1)
        return prob, value


def get_model():
    env = gym.make("Pong-v0")
    n = 80 * 80  # input dimension
    action_bias = 2
    transform = PongTransform(n)

    pi = Policy()
    init.xavier_uniform_(pi.w1.weight, gain=np.sqrt(2))
    init.xavier_uniform_(pi.w2.weight, gain=np.sqrt(2))
    return env, transform, pi, action_bias
