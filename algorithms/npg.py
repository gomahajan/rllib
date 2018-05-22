import torch
from torch.distributions import Categorical
from itertools import count
from utils.debug import plot

def naive_policy_gradient(env, transform, pi, optimizer, render=False, batch_size=100, gamma=0.99, resume=False,
                          data_dir='data/training.pt', save_step=100, action_bias=0, debug=False):
    if resume:
        pi.load_state_dict(torch.load(data_dir))

    running_reward = None
    episode_lengths = []

    saved_logp = []
    saved_rewards = []
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
            saved_logp.append(m.log_prob(action))

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

                future_rewards = torch.Tensor(future_rewards)

                # Normalizing future rewards
                future_rewards = (future_rewards - future_rewards.mean()) / (future_rewards.std())

                saved_rewards.extend(future_rewards)

                if debug:
                    episode_lengths.append(t+1)
                    plot(episode_lengths)
                break

        # Every batch_size episodes, backpropagate the gradients and update parameters
        if episode % batch_size == 0:
            # Calculating loss as sum of product of log probability and future reward
            losses = []
            for logp_loop, reward_loop in zip(saved_logp, saved_rewards):
                losses.append(-logp_loop * reward_loop)

            loss = torch.cat(losses).sum()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del saved_rewards[:]
            del saved_logp[:]
            print('loss {} with reward {}'.format(loss.item(), running_reward))

        if episode % save_step == 0:
            print('episode: {} has reward: {} with running mean: {}'.format(episode, reward_sum, running_reward))
            torch.save(pi.state_dict(), data_dir)

        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = running_reward * 0.99 + reward_sum * 0.01
