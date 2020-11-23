import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

from PIL import Image
from TD3 import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(tag = ""):
    """
    Main loop for training. Make sure to change tag to some different identifier depending on what
    sort of model or implementation you are running 
    """


    ######### Hyperparameters #########
    # change to whatever you want :-) 
    env_name = "BipedalWalkerHardcore-v3"

    log_interval = 100           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 500          # num of transitions sampled from replay buffer
    lr = 0.000001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 50000         # max num of episodes
    max_timesteps = 5000        # max timesteps in one episode
    directory = "preTrained/" # save trained models
    filename = tag + "TD3_{}_{}".format(env_name, random_seed)
    #RENDER_INTERVAL = 50
    ###################################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open(env_name + tag + "log.txt","w+")
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action, add exploration noise, and clip if too big.:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, _ = env.step(action) # some x dim vector, with each value bounded
            replay_buffer.add((state, action, reward, next_state, float(done))) # save to replay buffer, we'll see where this goes
            state = next_state # turn into next state, used for next timestep in env. 

            avg_reward += reward #update rolling reward
            ep_reward += reward #update episode reward

            # if episode is done then update policy:
            if done or t==(max_timesteps-1): # time to update the policy!
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        # if avg reward > 300 then save and stop traning:
        if (avg_reward/log_interval) >= 300:
            print("########## Solved! ###########")
            name = filename + '_solved'
            policy.save(directory, name)
            log_f.close()
            break

        if episode > 500:
            policy.save(directory, filename)

        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

      

    print("Done!")

if __name__ == '__main__':
	main(tag = "v3")

