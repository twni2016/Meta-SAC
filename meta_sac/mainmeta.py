import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) # custom hyperparams
import os

if config['exp_id'] != 'debug':
    if not os.path.exists('logs/' + config['exp_id']):
        os.makedirs('logs/' + config['exp_id'])

    log_file = 'logs/' + config['exp_id'] + '/' + '{}.txt'.format(config['seed'])
    sys.stdout = open(log_file, 'w')

print(config)
print(os.getpid())

cores = str(config['cores'])
os.environ["OMP_NUM_THREADS"] = cores # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = cores # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores # export NUMEXPR_NUM_THREADS=6

import datetime
import gym
import numpy as np
import itertools
import torch, random
from sacmeta import SAC_META as SAC
from replay_memory import ReplayMemoryKL as ReplayMemory

# Environment
env = gym.make(config['env_name'])
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
env.seed(config['seed'])
env.action_space.np_random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, config)

# Memory
memory = ReplayMemory(config['replay_size'])
kl_memory = ReplayMemory(config['kl_replay_size'])

# Training Loop
total_numsteps = 0
updates = 0

# make model save path
import os
from os import path
import time

def test(env, mode):
    '''
    mode='zero': use alpha = 0 in policy net.
    mode='running': use alpha as meta alpha in policy net.
    '''
    avg_reward = 0.
    episodes = 10
    for _  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(state, eval=True, mode = mode)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {} Mode: {}".format(episodes, round(avg_reward, 2), mode))
    print("----------------------------------------")

if config['meta_obj_s0']:
    s0_list = np.stack([env.reset() for _ in range(config['batch_size'])])

test_step = 10000
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    indicators = np.zeros((6), np.float32) # log_alpha, critic_1_loss, critic_2_loss, policy_loss, new_policy_loss, entropy

    while not done:
        if config['start_steps'] > total_numsteps:
            action, log_prob = env.action_space.sample(), [1 / 2**(env.action_space.shape[0])]  # Sample random action
        else:
            action, log_prob = agent.select_action(state)  # Sample action from policy


        if len(memory) > config['batch_size']:
            # Number of updates per step in environment
            for i in range(config['updates_per_step']):
                # Update parameters of all the networks
                if config['meta_obj_s0']:
                    indicators += agent.update_parameters(memory, kl_memory, config['batch_size'], updates, s0_list)
                else:
                    indicators += agent.update_parameters(memory, kl_memory, config['batch_size'], updates)
                updates += 1
        else:
            indicators[0] += agent.log_alpha.item()
            
        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, log_prob, reward, next_state, mask) # Append transition to memory
        kl_memory.push(state, action, log_prob, reward, next_state, mask)

        state = next_state

        # if total_numsteps in save_numsteps:
        #     agent.save_model(save_path, env_name=config['env_name'], suffix="{}".format(str(total_numsteps)))

    if total_numsteps > config['num_steps']:
        break
    
    indicators /= episode_steps
    print("Episode: {} total numsteps: {} episode steps: {} reward: {} log alpha: {} c1: {} c2: {} p: {} mp: {} ent: {}".format(
        i_episode, total_numsteps, episode_steps, str(round(episode_reward, 2)), str(round(indicators[0], 3)), 
        str(round(indicators[1], 2)), str(round(indicators[2], 2)), str(round(indicators[3], 2)), 
        str(round(indicators[4], 2)), str(round(indicators[5], 2))
        ))

    if total_numsteps > test_step and config['eval'] == True:
        test_step += 10000
        if config['alpha_embedding']:
            test(env, 'zero')
        test(env, 'running')
        print("Test Log Alpha: {}".format(agent.log_alpha.item()))
        print("---------------------------------------")

env.close()
