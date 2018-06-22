import argparse
import gym
import numpy as np
from itertools import count
import collections
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--num-episodes', type=int, default=1, metavar='N',
                    help='number of episodes per policy (default: 20)')
parser.add_argument('--num-epochs-policy', type=int, default=1, metavar='N',
                    help='number of epochs while learning the policy (default: 5)')
parser.add_argument('--num-epochs-q', type=int, default=20, metavar='N',
                    help='number of epochs while learning the q-value (default: 5)')
parser.add_argument('--batch-size-q', type=int, default=200, metavar='N',
                    help='batch size for learning q function (default: 20)')
parser.add_argument('--batch-size-policy', type=int, default=200, metavar='N',
                    help='batch size for learning policy (default: 200)')
parser.add_argument('--buffer-size', type=int, default=1, metavar='N',
                    help='buffer size for previous experiences (default: 1)')
parser.add_argument('--min-q-loss', type=int, default=10, metavar='N',
                    help='min loss for q function (default: 1)')
parser.add_argument('--debug', action='store_true',
                    help='shows debug graphs')
parser.add_argument('--approx', action='store_true',
                    help='use q function approximation')

args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

Experience = collections.namedtuple('Experience', 'interactions value')
Interaction = collections.namedtuple('Interaction', 'state action reward_to_go')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class QFunction(nn.Module):
    def __init__(self):
        super(QFunction, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return action_scores

class ExperienceDataset(Dataset):

    def __init__(self, states, actions, reward_to_gos):
        self.states = states
        self.actions = actions
        self.reward_to_gos = reward_to_gos

    def __getitem__(self, index):
        return self.states[index, :], self.actions[index, :], self.reward_to_gos[index, :]

    def __len__(self):
        return list(self.states.size())[0]

exp = Experience([], 0)
exps = [exp]* args.buffer_size
policy = Policy()
qfunction = QFunction()
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

def log_prob_action(state, action):
    state = state.unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    return m.log_prob(action)

# run the policy to get experiences
def sample():
    total_rewards = []
    interactions = []
    for i_episode in count(1):
        total_reward = 0
        states = []
        actions = []
        rewards = []

        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            states.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            total_reward = total_reward + reward
            rewards.append(reward)
            if done:
                break
        R = 0
        reward_to_gos = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            reward_to_gos.insert(0, R)

        for state, action, reward_to_go in zip(states, actions, reward_to_gos):
            interactions.append(Interaction(state, action, reward_to_go))

        total_rewards.append(total_reward)
        #if i_episode % args.log_interval == 0:
            #print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            #    i_episode, t, np.mean(total_rewards)))

        if i_episode % args.num_episodes == 0:
            break

    return Experience(interactions, np.mean(total_rewards))

# replace experience with least value with new experience
def update_exp(new_exp):
    min = exps[0].value
    min_idx = 0
    for index, exp in enumerate(exps):
        if exp.value <= min:
            min_idx = index
            min = exp.value

    exps[min_idx] = new_exp

qloss = []
# learn q value
def learn_q(dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_q)
    optimizer = optim.Adam(qfunction.parameters(), lr=1e-2)

    for epoch in count(1):
        loss = 0
        for i, data in enumerate(trainloader, 0):
            losses = []
            states, actions, reward_to_gos = data

            for state, action, reward_to_go in zip(states, actions, reward_to_gos):
                losses.append((qfunction(state)[int(action.item())] - reward_to_go)**2)

            optimizer.zero_grad()
            losses = torch.cat(losses).sum()
            loss = losses.item()
            qloss.append(loss)
            losses.backward()
            optimizer.step()

        if loss <= args.min_q_loss or epoch % 100 == 0:
            break

    if args.debug:
        plt.figure(3)
        plt.clf()
        plt.title('QFunction Loss')
        plt.plot(qloss)
        plt.pause(0.001)

# learn best policy based on q value
ploss = []
policy_optimizer = optim.RMSprop(policy.parameters(), lr=1e-3)
def learn_policy(dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.states.shape[0] + 1)

    for epoch in count(1):
        for i, data in enumerate(trainloader, 0):
            losses = []
            states, actions, reward_to_gos = data

            log_probs = []
            a = []
            #reward_to_gos = (reward_to_gos- reward_to_gos.mean()) / (reward_to_gos.std() + eps)
            for state, action, reward_to_go in zip(states, actions, reward_to_gos):
                log_probs.append(log_prob_action(state, action))
                a.append(action.item())
                if args.approx:
                    reward_to_go = qfunction(state)[int(action.item())]
                losses.append(-log_prob_action(state, action)* reward_to_go) #qfunction(state)[int(action.item())]

            if args.debug:
                plt.figure(5)
                plt.clf()
                plt.title('Log Probs')
                plt.plot(log_probs)
                plt.plot(a)
                plt.pause(0.001)

            policy_optimizer.zero_grad()
            losses = torch.cat(losses).sum()
            ploss.append(losses.item())
            losses.backward()
            policy_optimizer.step()

        if args.debug:
            plt.figure(4)
            plt.clf()
            plt.title('Policy Loss')
            plt.plot(ploss)
            plt.pause(0.001)

        if epoch % args.num_epochs_policy == 0:
            break

def createDataset():
    states = []
    actions = []
    reward_to_gos = []

    for exp in exps:
        for interaction in exp.interactions:
            states.append(interaction.state)
            actions.append(interaction.action)
            reward_to_gos.append(interaction.reward_to_go)

    return ExperienceDataset(torch.Tensor(states), torch.Tensor(actions).view(-1,1), torch.Tensor(reward_to_gos).view(-1,1))

def main():
    running_reward = 10
    all_rewards = []
    for step in count(1):
        new_exp = sample()
        update_exp(new_exp)
        dataset = createDataset()
        if args.approx:
            learn_q(dataset)
        learn_policy(dataset)
        all_rewards.append(new_exp.value)
        running_reward = running_reward * 0.99 + new_exp.value * 0.01

        print('Episode {}\tLast length: {:.2f}\tAverage length: {:.2f}'.format(
            step, new_exp.value, running_reward))
        plt.figure(2)
        plt.clf()
        plt.title('Rewards')
        plt.plot(all_rewards)
        plt.pause(0.001)

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()