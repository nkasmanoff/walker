"""
Implementation of TD3 Algorithm on Open AI gym environment BipedalWalkerHardcore v3


I'm changing it slightly, using a deeper rather than wider net. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 256)
        self.l4 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        """
        Pass the state through mlp, and emit a value between -1 and 1, scaled by max action size. 
        
        Returns the action to take (along each dim of action dim)
        """
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = torch.tanh(self.l4(a)) * self.max_action
        return a
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
       
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 256)
        self.l4 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        """
        Returns a Q(!) value for the network, given a state action input. The input is concatted and passed through an MLP
        of roughly the same size as before, but different input dim to account for the action(s) taken. 
        """

        state_action = torch.cat([state, action], 1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        q = self.l4(q)
        return q
    
class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.max_action = max_action
    
    def select_action(self, state):
        """
        Pass the state (after casted to right type) through the actor network 
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        """
        Update the actor and critic networks. The difference here between TD3 and DDPG is that there
        
        are now two critic newtworks, both of which compute the Q value, and we designate our
        target Q value as the smaller between the two critic network predictions. 
        """
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            
            # providing us with our s a r s' done tuple. 
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # Select next action according to target policy:
            
            # the next action we take is chosen by pasing our next state vector into the target policy network
            # the target policy network, a polyak averaged slowly updating version of the current policy network. 
            
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device) # we still use noise for our target? 
            noise = noise.clamp(-noise_clip, noise_clip) # this val pre-defined as at most .5
            next_action = (self.actor_target(next_state) + noise) # actor predicts next action choice, and we sprinkle in a little noise
            next_action = next_action.clamp(-self.max_action, self.max_action) #clipped. 
            
            # Compute target Q-value:
            # Using our next state from replay buffer and what the actor network chose as next action, can 
            # compute Q value for this action for both state critic nets.
            target_Q1 = self.critic_1_target(next_state, next_action) # Note these are the critic target networks, i.e a older version of critic network to avoid model chasing it's own tail. 
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) # take the min, to avoid over-estimation bias
            target_Q = reward + ((1-done) * gamma * target_Q).detach() # convert to proper form, i.e TD bellman etc. update rule
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action) # this is supposed to be the same Q value, but one timestep earlier
            loss_Q1 = F.mse_loss(current_Q1, target_Q) # MSE loss, 
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward() #and backward + step for both
            self.critic_2_optimizer.step()
            
            # Delayed policy updates: 
            
            # we also want to improve our policy net, but this is done less frequently than critic updates
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean() #actor loss is defined as 
                # - Q value of all batches, passed backward through actor.
                
                # why only the first critic? What if it the worse estimator of Q values than critic 1? 
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
        

######### Hyperparameters #########
env_name = "BipedalWalkerHardcore-v3"

log_interval = 100           # print avg reward after interval
random_seed = 0
gamma = 0.99                # discount for future rewards
batch_size = 32          # num of transitions sampled from replay buffer
lr = 0.0001
exploration_noise = 0.1 
polyak = 0.995              # target policy update parameter (1-tau)
policy_noise = 0.2          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter
max_episodes = 30000         # max num of episodes
max_timesteps = 5000        # max timesteps in one episode
directory = "preTrained/" # save trained models
filename = "DeepTD3_{}_{}".format(env_name, random_seed)
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
log_f = open(env_name + "Deep_log.txt","w+")
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

