import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import FCNet, ConvNet, Policy

class PPOAgent:
    def __init__(self, 
                 state_size,
                 action_size, 
                 seed,
                 hidden_layers,
                 lr_policy, 
                 use_reset, 
                 device
                ):

        #self.main_net = ConvNet(state_size, feature_dim, seed, use_reset, input_channel).to(device)
        self.main_net = FCNet(state_size, seed, hidden_layers=[64,64], use_reset=True, act_fnc=F.relu).to(device)
        self.policy = Policy(state_size, action_size, seed, self.main_net).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.device = device

    def update(self, log_probs_old, states, actions, returns, advantages, cliprange=0.1, beta=0.01):
    
        traj_info = self.policy.act(states, actions)
        
        ratio = torch.exp(traj_info['log_pi_a'] - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = 0.5*(returns - traj_info['v']).pow(2).mean()
        entropy = traj_info['ent'].mean()

        self.optimizer.zero_grad()
        (policy_loss + value_loss - beta*entropy).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.optimizer.step()

        return policy_loss.data.cpu().numpy(), value_loss.data.cpu().numpy(), entropy.data.cpu().numpy()