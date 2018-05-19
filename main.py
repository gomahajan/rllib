import torch.optim as optim
from algorithms.npg import naive_policy_gradient
from models.pong import get_model

# hyperparameters
batch_size = 5
lr = 0.01
gamma = 0.99
decay_rate = 0.99
data_dir = 'data/training.pt'
action_bias = 0

env, transform, pi, action_bias = get_model()
optimizer = optim.RMSprop(pi.parameters(), lr=lr)

naive_policy_gradient(env, transform, pi, optimizer, render=True, gamma=gamma, batch_size=batch_size, data_dir=data_dir,
                      resume=True, save_step=100, debug=True, action_bias=action_bias)
