import torch
import torch.nn as nn
from torch.distributions import Categorical
from itertools import count
import numpy as np
from utils.debug import plot


def sample(env, pi, transform, action_bias, render):
    states = []
    rewards = []
    logps = []
    values = []

    observation = env.reset()

    # get rewards for all state, actions
    for t in count():
        state = transform.get_state(observation)
        states.append(state)

        # Get action
        state = torch.from_numpy(state).float().unsqueeze(0)
        prob, value = pi(state)
        m = Categorical(prob)
        action = m.sample()
        logps.append(m.log_prob(action))
        values.append(value.item())

        observation, reward, done, _ = env.step(action.item() + action_bias)
        if render:
            env.render()

        rewards.append(reward)

        if done:
            break

    return states, rewards, logps, values


def compute_Q(rewards, c, k, values, gamma):
    rs = []
    lidx = len(rewards) - 1

    assert lidx >= 0, '[Qvalue] Last index should be non-negative'

    for i in count():
        if i == k:
            break

        if c + i == lidx:
            break

        rs.append((gamma ** i) * rewards[c + i])

    return np.sum(rs) + (gamma ** i) * values[c + i]


def compute_R(rewards, c, H, gamma):
    rs = []
    lidx = len(rewards) - 1

    assert H > 0, '[RValue] H should be greater than zero'

    for i in count():
        if i == H:
            break
        if c + i > lidx:
            break

        rs.append((gamma ** i) * rewards[c + i])

    return np.sum(rs)


def learn_V(pi, optimizer, X, Y, steps):
    criterion = nn.MSELoss()
    states = torch.Tensor(X)
    values = torch.Tensor(Y).view(-1, 1)

    for step in count():
        optimizer.zero_grad()
        if step > steps:
            break

        _, out = pi(states)
        loss = criterion(out, values)
        loss.backward()
        optimizer.step()


def actor_critic(env, transform, pi, optimizer, k, H, steps, render=False, batch_size=100, gamma=0.99, resume=False,
                 data_dir='data/ac-training.pt', save_step=100, action_bias=0, debug=False, bggd=False):
    if resume:
        pi.load_state_dict(torch.load(data_dir))

    reward_episodes = []
    loss_episodes = []
    Qs = []
    Rs = []
    S = []
    LP = []
    Vs = []
    for episode in count(1):
        states, rewards, logps, values = sample(env, pi, transform, action_bias, render)
        S.extend(states)
        LP.extend(logps)
        Vs.extend(values)

        n = len(rewards)

        for t in count():
            if t == n:
                break

            Qs.append(compute_Q(rewards, t, k, values, gamma))
            Rs.append(compute_R(rewards, t, H, gamma))

        reward_episodes.append(np.sum(rewards))
        print('episode: {} has reward: {}'.format(episode, np.sum(rewards)))

        if episode % batch_size == 0:
            learn_V(pi, optimizer, S, Rs, steps)
            A = np.subtract(Qs, np.squeeze(Vs))
            A = torch.Tensor(A)
            A = (A - A.mean())/A.std()
            losses = []
            for logp_loop, reward_loop in zip(LP, A):
                losses.append(-logp_loop * reward_loop)

            loss = torch.cat(losses).sum()
            optimizer.zero_grad()
            loss.backward()
            loss_episodes.append(loss)
            optimizer.step()

            print('episode: {} has loss: {}'.format(episode, loss.item()))

            if debug:
                plot(reward_episodes)

            Qs = []
            Rs = []
            S = []
            LP = []
            Vs = []

        if episode % save_step == 0:
            torch.save(pi.state_dict(), data_dir)
