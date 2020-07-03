import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) # custom hyperparams
print(config)

import os
cores = str(config['cores'])
os.environ["OMP_NUM_THREADS"] = cores # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = cores # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores # export NUMEXPR_NUM_THREADS=6

import random
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory

env_names = {
    "Ant-v2": 'ant',
    "Hopper-v2": 'hopper',
    "HalfCheetah-v2": 'halfcheetah', 
    "Humanoid-v2": 'humanoid',
    "Walker2d-v2": 'walker2d',
    "Swimmer-v2": 'swimmer'
}

import time
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
save_path = 'models/' + config['exp_id'] + '/' + current_time + '/'
print(save_path)
os.makedirs(save_path)


if config['exp_id'] != 'debug':
    dir = '../common/vanilla_SAC_log/{}/'.format(config['seed'])
    os.makedirs(dir, exist_ok=True)
    version = 'v1' if not config['automatic_entropy_tuning'] else 'v2'
    log_file = dir + env_names[config['env_name']] + '_' + version + '.txt'
    print(log_file)
    sys.stdout = open(log_file, 'w')
    
print(config)
print(os.getpid())

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

# Training Loop
total_numsteps = 0
updates = 0
test_step = 10000

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    acc_log_alpha = 0.
    while not done:
        if config['start_steps'] > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > config['batch_size']:
            # Number of updates per step in environment
            for i in range(config['updates_per_step']):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, config['batch_size'], updates)

                updates += 1
                acc_log_alpha += np.log(alpha)

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

        # if total_numsteps in save_numsteps:
        #     if config['automatic_entropy_tuning']:
        #         version = 'v2'
        #     else:
        #         version = 'v1'
        #     agent.save_model(save_path, env_name=config['env_name'], suffix="{}{}".format(str(total_numsteps), version))

    if total_numsteps > config['num_steps']:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {} mean log alpha {}".format(
        i_episode, total_numsteps, episode_steps, round(episode_reward, 2), acc_log_alpha / episode_steps
        ))

    if total_numsteps > test_step and config['eval'] == True:
        test_step += 10000

        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        if config['automatic_entropy_tuning']:
            print("Test Log Alpha: {}".format(agent.log_alpha.item()))
        print("----------------------------------------")

env.close()

