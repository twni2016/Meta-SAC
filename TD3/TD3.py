import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size=256):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
		self.l5 = nn.Linear(hidden_size, hidden_size)
		self.l6 = nn.Linear(hidden_size, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		lr=3e-4,
		hidden_size=256,
		parameter_noise_mean=0.01,
		parameter_noise_std=1,
		cuda=0,
	):

		self.device = torch.device('cuda:' + str(cuda)) if torch.cuda.is_available() and cuda >= 0 else torch.device('cpu')
		self.actor = Actor(state_dim, action_dim, max_action, hidden_size).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_explore = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = Critic(state_dim, action_dim, hidden_size).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

		self.parameter_noise_mean = parameter_noise_mean
		self.parameter_noise_std = parameter_noise_std

	def hard_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()

	def select_exploration_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor_explore(state).cpu().data.numpy().flatten()

	def inject_parameter_noise(self):
		parameter_explore_noise = torch.distributions.normal.Normal(loc = self.parameter_noise_mean, scale=self.parameter_noise_std)
		self.hard_update(self.actor_explore, self.actor)
		for param in self.actor_explore.parameters():
			param.data += parameter_explore_noise.sample(param.data.shape).to(self.device)

	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# Save model parameters
	def save_model(self, save_path = None, env_name = None, suffix = None):
		if save_path is None:
		    save_path = './models/'
		actor_path = '{}actor_{}_{}'.format(save_path, env_name, suffix)
		critic_path = "{}critic_{}_{}".format(save_path, env_name, suffix)
		print('Saving models to {}, {}'.format(actor_path, critic_path))
		torch.save(self.actor.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)
