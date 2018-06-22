import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
debug = False


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


ploss = []
def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    ploss.append(policy_loss.item())
    policy_loss.backward()
    optimizer.step()

    if debug:
        plt.figure(5)
        plt.clf()
        plt.title('Log Probs')
        plt.plot(policy.saved_log_probs)
        plt.pause(0.001)

        plt.figure(4)
        plt.clf()
        plt.title('Policy Loss')
        plt.plot(ploss)
        plt.pause(0.001)
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    all_rewards = []
    for i_episode in count(1):
        state = env.reset()
        a = []
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            a.append(action)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        if debug:
            plt.figure(1)
            plt.clf()
            plt.title('actions')
            plt.plot(a)
            plt.pause(0.001)

        all_rewards.append(t)
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        plt.figure(2)
        plt.clf()
        plt.plot(all_rewards)
        plt.pause(0.001)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()