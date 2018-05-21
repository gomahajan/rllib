import torch
from torch.distributions import Categorical
from itertools import count
import matplotlib.pyplot as plt
import numpy as np


def plot(episode_lengths, rewards):
    plt.figure(2)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.plot(episode_lengths)
    #plt.plot(rewards)
    plt.pause(0.001)


class BPenalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point):
        is_boundary = ((point == 0).float().mean() > 0).float()

        penalty = point.clone()

        # If on boundary, then zero loss
        if is_boundary > 0:
            penalty.fill_(0)

        # Otherwise, penalise distance from boundary.
        penalty[point > 0] = 0
        ctx.save_for_backward(penalty)
        return penalty

    @staticmethod
    def backward(ctx, grad_output):
        penalty, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[penalty >= 0] = 0
        return grad_input

penalty = BPenalty.apply

def naive_policy_gradient(env, transform, pi, optimizer, render=False, batch_size=100, gamma=0.99, resume=False,
                          data_dir='data/training.pt', save_step=100, action_bias=0, debug=False, bggd=False):
    if resume:
        pi.load_state_dict(torch.load(data_dir))

    running_reward = None
    episode_lengths = []
    trajectory_rewards = []

    for episode in count(1):
        observation = env.reset()
        reward_sum = 0
        rewards = []
        future_rewards = []

        # For each state action pair, save log probabilities and rewards
        for t in count():
            state = transform.get_state(observation)

            # Get action
            state = torch.from_numpy(state).float().unsqueeze(0)
            prob, h = pi(state)
            m = Categorical(prob)
            action = m.sample()

            # Save log probability for action
            pi.saved_logp.append(m.log_prob(action))

            pi.saved_penalty.append(100 * (penalty(h)** 2).mean())

            observation, reward, done, _ = env.step(action.item() + action_bias)
            if render: env.render()

            if done:
                reward = 0

            # Save reward for action
            rewards.append(reward)
            reward_sum = reward_sum + reward

            if done:
                R = 0
                for r in reversed(rewards):
                    R = gamma * R + r
                    future_rewards.insert(0, R)

                trajectory_rewards.append(future_rewards[-1])
                future_rewards = torch.Tensor(future_rewards)

                # Normalizing future rewards
                future_rewards = (future_rewards - future_rewards.mean()) / (future_rewards.std())

                pi.saved_rewards.extend(future_rewards)

                if debug:
                    episode_lengths.append(t+1)
                    plot(episode_lengths, pi.saved_rewards)
                break

        # Every batch_size episodes, backpropagate the gradients and update parameters
        if episode % batch_size == 0:
            # Calculating loss as sum of product of log probability and future reward
            losses = []
            for logp_loop, reward_loop in zip(pi.saved_logp, pi.saved_rewards):
                losses.append(-logp_loop * reward_loop)

            bloss = 0
            if bggd:
                bloss = np.sum(pi.saved_penalty)

            loss = torch.cat(losses).sum() + bloss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del pi.saved_rewards[:]
            del pi.saved_logp[:]
            del pi.saved_penalty[:]
            print('loss {} with reward {}'.format(loss.item(), running_reward))

        if episode % save_step == 0:
            print('episode: {} has reward: {} with running mean: {}'.format(episode, reward_sum, running_reward))
            torch.save(pi.state_dict(), data_dir)

        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = running_reward * 0.99 + reward_sum * 0.01
