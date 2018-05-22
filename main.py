import torch.optim as optim
from algorithms.ac import actor_critic
from algorithms.npg import naive_policy_gradient
from models.cart import get_model

# hyperparameters
batch_size = 1
lr = 0.001
gamma = 0.99
decay_rate = 0.99
data_dir = 'data/training.pt'
k = 5
steps = 5
H = 1000
one_dim = True

env, transform, pi, action_bias = get_model()
optimizer = optim.Adam(pi.parameters(), lr=lr)

#naive_policy_gradient(env, transform, pi, optimizer, render=True, gamma=gamma, batch_size=batch_size, data_dir=data_dir,
#                      resume=False, save_step=100, debug=True, action_bias=action_bias, one_dim=one_dim)

actor_critic(env, transform, pi, optimizer, k, H, steps, render=True, batch_size=batch_size, gamma=0.99, resume=False,
             data_dir='data/ac-cart-training.pt', save_step=100, action_bias=action_bias, debug=True, one_dim=one_dim)
