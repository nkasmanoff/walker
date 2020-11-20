"""
Models to use as a part of the TD3 implementation. Depending on actor type used

will change what is used in train and TD3 scripts. 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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
