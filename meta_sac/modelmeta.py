import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class MetaQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(MetaQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, alpha, alpha_embedding):
        if alpha_embedding:
            xu = torch.cat([state, action, alpha], dim = 1)
        else:
            xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class NewGaussianPolicy(nn.Module):
    def __init__(self, policy_params, action_space = None):
        super(NewGaussianPolicy, self).__init__()
        self.linear1_weight = policy_params[0]
        self.linear1_bias = policy_params[1]
        self.linear2_weight = policy_params[2]
        self.linear2_bias = policy_params[3]
        self.mean_linear_weight = policy_params[4]
        self.mean_linear_bias = policy_params[5]
        self.log_std_linear_weight = policy_params[6]
        self.log_std_linear_bias = policy_params[7]
        assert len(policy_params) == 8

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, alpha, alpha_embedding):
        if alpha_embedding:
            x = torch.cat([state, alpha], dim = 1)
        else:
            x = state
        x = F.relu(F.linear(x, self.linear1_weight, self.linear1_bias))
        x = F.relu(F.linear(x, self.linear2_weight, self.linear2_bias))
        mean = F.linear(x, self.mean_linear_weight, self.mean_linear_bias)
        log_std = F.linear(x, self.log_std_linear_weight, self.log_std_linear_bias)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, alpha, alpha_embedding):
        mean, log_std = self.forward(state, alpha, alpha_embedding)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_log_prob(self, state, action, alpha, alpha_embedding):
        mean, log_std = self.forward(state, alpha, alpha_embedding)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, min=-1 + epsilon, max=1 - epsilon)

        x_t = torch.log((1+y_t) / (1-y_t)) / 2
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(NewGaussianPolicy, self).to(device)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, alpha, alpha_embedding):
        if alpha_embedding:
            x = torch.cat([state, alpha], dim = 1)
        else:
            x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, alpha, alpha_embedding):
        mean, log_std = self.forward(state, alpha, alpha_embedding)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    def get_log_prob(self, state, action, alpha, alpha_embedding):
        mean, log_std = self.forward(state, alpha, alpha_embedding)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, min=-1 + epsilon, max=1 - epsilon)

        x_t = torch.log((1+y_t) / (1-y_t)) / 2
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob

