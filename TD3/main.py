import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) # custom hyperparams
import os

env_names = {
    "Ant-v2": 'ant',
    "Hopper-v2": 'hopper',
    "HalfCheetah-v2": 'halfcheetah', 
    "Humanoid-v2": 'humanoid',
    "Walker2d-v2": 'walker2d',
    "Swimmer-v2": 'swimmer'
}

if config['exp_id'] != 'debug':
    dir = '../common/vanilla_TD3_log/{}/'.format(config['seed'])
    os.makedirs(dir, exist_ok=True)
    log_file = dir + '{}-{}.txt'.format(env_names[config['env_name']], config['noise_type'])
    print(log_file)
    sys.stdout = open(log_file, 'w')

print(config)
print(os.getpid())

cores = str(config['cores'])
os.environ["OMP_NUM_THREADS"] = cores # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = cores # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores # export NUMEXPR_NUM_THREADS=6

import gym
import numpy as np
import itertools
import torch, random
import TD3
from replay_memory import ReplayBuffer as ReplayMemory 

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
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": config['gamma'],
    "tau": config['tau'],
    "lr": config['lr'],
    "hidden_size": config['hidden_size'],
    'cuda': config['cuda'],
    'parameter_noise_mean': config['param_noise_mean'],
    'parameter_noise_std': config['param_noise_std']
}

# Target policy smoothing is scaled wrt the action scale
kwargs["policy_noise"] = config['policy_noise'] * max_action
kwargs["noise_clip"] = config['noise_clip'] * max_action
kwargs["policy_freq"] = config['policy_freq']
agent = TD3.TD3(**kwargs)


# Memory
device = torch.device('cuda:' + str(config['cuda'])) if torch.cuda.is_available() and config['cuda'] >= 0 else torch.device('cpu')
memory = ReplayMemory(state_dim, action_dim, config['replay_size'], device)

# Training Loop
total_numsteps = 0
updates = 0

# make model save path
from os import path
import time
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
save_path = 'models/' + config['env_name'] + '/' + current_time + '/'
print(save_path)
if not path.exists(save_path):
    os.makedirs(save_path)


def test(env):
    avg_reward = 0.
    episodes = 10
    for _  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(np.array(state))

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")

test_step = 10000
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    if config['noise_type'] == 'parameter':
        # print("inject noise!")
        agent.inject_parameter_noise()

    while not done:
        if total_numsteps < config['start_steps']:
            action = env.action_space.sample()  # Sample random action
        else:
            if config['noise_type'] == 'action':
                action = (
                    agent.select_action(np.array(state))
                    + np.random.normal(0, max_action * config['expl_noise'], size=action_dim)
                ).clip(-max_action, max_action)
            else: # parameter noise, agent use exploration policy to selection action
                # print("use exploration net!")
                action = (agent.select_exploration_action(np.array(state))).clip(-max_action, max_action)

        if total_numsteps >= config['start_steps']:
            # Number of updates per step in environment
            for i in range(config['updates_per_step']):
                # Update parameters of all the networks
                agent.train(memory, config['batch_size'])
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        done_bool = float(done) if episode_steps < env._max_episode_steps else 0
        memory.add(state, action, next_state, reward, done_bool) # Append transition to memory

        state = next_state

    if total_numsteps > config['num_steps']:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
        i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
        ))

    if total_numsteps > test_step and config['eval'] == True:
        test(env)
        test_step += 10000

    # if i_episode % config['save_interval'] == 0 and i_episode > 0:
    #     agent.save_model(save_path, env_name=config['env_name'], suffix="{}".format(str(i_episode)))

env.close()

