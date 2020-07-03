import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import RMSprop, soft_update, hard_update
from modelmeta import GaussianPolicy, MetaQNetwork, NewGaussianPolicy, ValueNetwork
import math, numpy as np

class SAC_META(object):
    def __init__(self, num_inputs, action_space, config):

        self.gamma = config['gamma']
        self.tau = config['tau']

        self.target_update_interval = config['target_update_interval']
        self.alpha_embedding = config['alpha_embedding']
        self.meta_Q = config['meta_Q']

        self.device = torch.device('cuda:' + str(config['cuda'])) if torch.cuda.is_available() and config['cuda'] >= 0 else torch.device('cpu')

        self.critic = MetaQNetwork(num_inputs + 1 if self.alpha_embedding else num_inputs,
                action_space.shape[0], config['hidden_size']).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config['lr'])

        self.critic_target = MetaQNetwork(num_inputs + 1 if self.alpha_embedding else num_inputs,
                action_space.shape[0], config['hidden_size']).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.meta_Q:
            self.meta_critic = MetaQNetwork(num_inputs, action_space.shape[0], config['hidden_size']).to(device=self.device)
            self.meta_critic_optim = Adam(self.meta_critic.parameters(), lr=config['lr'])

            self.meta_v = ValueNetwork(num_inputs, config['hidden_size']).to(self.device)
            self.meta_v_optim = Adam(self.meta_v.parameters(), lr=config['lr'])

            self.meta_v_target = ValueNetwork(num_inputs, config['hidden_size']).to(self.device)
            hard_update(self.meta_v_target, self.meta_v)

        self.meta_train_interval = config['meta_train_interval']
        self.meta_grad_decay_coef = config['meta_grad_decay_coef']

        self.log_alpha = torch.tensor([config['log_alpha_max']], dtype=torch.float, requires_grad=True, device=self.device)
        assert config['alpha_optim'] == 'adam'
        self.alpha_optim = Adam([self.log_alpha], lr=float(config['meta_lr']))

        self.history_meta_grad = torch.zeros(1).to(self.device)

        self.policy = GaussianPolicy(num_inputs + 1 if self.alpha_embedding else num_inputs, 
            action_space.shape[0], config['hidden_size'], action_space).to(self.device)
        self.policy_optim = RMSprop(self.policy.parameters(), lr=config['lr'], eps=float(config['rmsprop_eps']))

        self.meta_clip_norm = config['clip_grad_norm']
        self.log_alpha_max = config['log_alpha_max']

        self.meta_obj_s0 = config['meta_obj_s0']
        
        self.resample = config['resample']

    def select_action(self, state, eval=False, mode=None):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if eval == False:
            alpha = torch.exp(self.log_alpha).unsqueeze(0)
            action, log_prob, _ = self.policy.sample(state, alpha, self.alpha_embedding)
        else:
            # zero: use alpha as zero.
            # running: use alpha as learning alpha.
            if mode == 'zero':
                alpha_ = torch.FloatTensor([0]).to(self.device)
                alpha = alpha_.repeat((len(state), 1)) 
                _, log_prob, action = self.policy.sample(state, alpha, self.alpha_embedding)
            elif mode == 'running':
                alpha = torch.exp(self.log_alpha).unsqueeze(0)
                _, log_prob, action = self.policy.sample(state, alpha, self.alpha_embedding)

        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]

    def update_parameters(self, memory, kl_memory, batch_size, updates, s0_list=None):

        # Sample a batch from memory, for SAC update
        state_batch, action_batch, log_prob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        log_prob_batch = torch.FloatTensor(log_prob_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        kl_state_batch, kl_action_batch, kl_log_prob_batch, kl_reward_batch, kl_next_state_batch, kl_mask_batch \
             = kl_memory.sample(batch_size=batch_size)
        kl_state_batch = torch.FloatTensor(kl_state_batch).to(self.device)
        kl_next_state_batch = torch.FloatTensor(kl_next_state_batch).to(self.device)
        kl_action_batch = torch.FloatTensor(kl_action_batch).to(self.device)
        kl_log_prob_batch = torch.FloatTensor(kl_log_prob_batch).to(self.device)
        kl_reward_batch = torch.FloatTensor(kl_reward_batch).to(self.device).unsqueeze(1)
        kl_mask_batch = torch.FloatTensor(kl_mask_batch).to(self.device).unsqueeze(1)

        if updates % self.meta_train_interval == 0:
            if self.meta_Q:
                alpha_ = torch.FloatTensor([0]).to(self.device)
                alpha = alpha_.repeat((len(kl_next_state_batch), 1))
                with torch.no_grad():
                    next_v = self.meta_v_target(kl_next_state_batch)
                    v_target = kl_reward_batch + kl_mask_batch * self.gamma * next_v
                
                qf1, qf2 = self.meta_critic(kl_state_batch, kl_action_batch, alpha, False)  # Two Q-functions to mitigate positive bias in the policy improvement step
                qf1_loss = F.mse_loss(qf1, v_target) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2, v_target) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

                self.meta_critic_optim.zero_grad()
                qf1_loss.backward() 
                self.meta_critic_optim.step()

                self.meta_critic_optim.zero_grad()
                qf2_loss.backward()
                self.meta_critic_optim.step()

                v = self.meta_v(kl_state_batch)
                v_loss = F.mse_loss(v, v_target)
                self.meta_v_optim.zero_grad()
                v_loss.backward()
                self.meta_v_optim.step()

            # compute the new policy loss with meta term
            alpha = torch.exp(self.log_alpha).repeat((len(next_state_batch), 1))       

            pi, log_pi, _ = self.policy.sample(state_batch, alpha.detach(), self.alpha_embedding) # policy needs to know alpha for sample
            qf1_pi, qf2_pi = self.critic(state_batch, pi, alpha.detach(), self.alpha_embedding) # Q-value also needs to know alpha
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = (alpha * log_pi - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            
            self.policy_optim.zero_grad()
            policy_loss.backward(create_graph=True)

            # simulate the rmsprop update procedure to get the new policy parameters
            new_policy_params = self.simulate_rmsprop(self.policy_optim)
            new_policy_net = NewGaussianPolicy(new_policy_params).to(self.device)

            # evaluate the new policy using policy gradient objective
            alpha_ = torch.FloatTensor([0]).to(self.device)
            alpha = alpha_.repeat((len(next_state_batch), 1))
            if self.meta_obj_s0:
                s0_batch = torch.FloatTensor(s0_list).to(self.device)
                a, log_pi, mean = new_policy_net.sample(s0_batch, alpha, self.alpha_embedding)
                if self.meta_Q:
                    qf1_pi, qf2_pi = self.meta_critic(s0_batch, a, alpha, False)
                    v = self.meta_v(s0_batch).detach()
                    adv = (torch.min(qf1_pi, qf2_pi) - v).detach()
                else:
                    qf1_pi, qf2_pi = self.critic(s0_batch, mean, alpha, self.alpha_embedding)

            else:
                _, log_pi, mean = new_policy_net.sample(state_batch, alpha, self.alpha_embedding)
                qf1_pi, qf2_pi = self.critic(state_batch, mean, alpha, self.alpha_embedding)
            
            if not self.meta_Q:
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                new_pg_loss = -min_qf_pi.mean() # this is the loss for the meta module
            else:
                new_pg_loss = -(log_pi * adv).mean()

            self.alpha_optim.zero_grad()
            new_pg_loss.backward()

            ### perform accumulation of the gradient on meta network
            self.log_alpha.grad += self.history_meta_grad * self.meta_grad_decay_coef
            self.history_meta_grad = self.log_alpha.grad.data
            
            norm = torch.nn.utils.clip_grad_norm_(self.log_alpha, self.meta_clip_norm)

            self.alpha_optim.step()
            self.log_alpha.data.clamp_(max=self.log_alpha_max)

        # redo the real meta Q network update
        if self.resample:
            state_batch, action_batch, log_prob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            log_prob_batch = torch.FloatTensor(log_prob_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # compute new alpha 
            alpha = torch.exp(self.log_alpha).repeat((len(next_state_batch), 1))

            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, alpha, self.alpha_embedding)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, alpha, self.alpha_embedding)
            min_qf_next_target_ = torch.min(qf1_next_target, qf2_next_target) 
            min_qf_next_target = min_qf_next_target_ - alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch, alpha, self.alpha_embedding)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        
        self.critic_optim.zero_grad()
        qf1_loss.backward() 
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        # redo the real policy update, after the meta module is updated
        pi, log_pi, _ = self.policy.sample(state_batch, alpha, self.alpha_embedding)

        qf1_pi, qf2_pi = self.critic(state_batch, pi, alpha, self.alpha_embedding)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (alpha * log_pi - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            if self.meta_Q:
                soft_update(self.meta_v_target, self.meta_v, self.tau)
        
        return np.array([self.log_alpha.item(), 
                        qf1_loss.item(), qf2_loss.item(), 
                        policy_loss.item(), new_pg_loss.item(), 
                        -log_pi.mean().item()])

    # Save model parameters
    def save_model(self, save_path = None, env_name = None, suffix = None):
        if save_path is None:
            save_path = './models/'

        actor_path = '{}actor_{}_{}'.format(save_path, env_name, suffix)
        critic_path = "{}critic_{}_{}".format(save_path, env_name, suffix)
        print('Saving models to {}, {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        if self.meta_Q:
            meta_v_path = '{}meta_v_{}_{}'.format(save_path, env_name, suffix)
            meta_q_path = "{}meta_q_{}_{}".format(save_path, env_name, suffix)
            print('Saving models to {}, {}'.format(meta_v_path, meta_q_path))
            torch.save(self.meta_v.state_dict(), meta_v_path)
            torch.save(self.meta_critic.state_dict(), meta_q_path)

    def simulate_rmsprop(self, optim):
        new_params = []
        for group in optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = optim.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                grad = p.grad
                square_avg = square_avg * alpha  + (1 - alpha) * torch.mul(grad, grad)
                avg = torch.sqrt(square_avg + group['eps']) 
                new_param = p - group['lr'] * grad / avg
                new_params.append(new_param)

        return new_params