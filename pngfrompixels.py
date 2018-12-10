#%% [markdown]
# ### Pong from pixels in Pytorch - Part 1
# This notebook is based on Andrej's Karpathy gist https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 
# Main difference is that it uses PyTorch. 
# Necessary imports - OpenAI's Gym, PyTorch and NumPy

#%%
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import log
import numpy as np

#%% [markdown]
# This is a simple 2 layer fully-connected neural network such as was used in original Karpathy's blogpost

#%%
class PolicyNetK(nn.Module):
    """Predict the probability of moving UP"""
    def __init__(self):
        super(PolicyNetK, self).__init__()        
        self.fc1 = nn.Linear(80*80, 100, bias=False)
        self.fc2 = nn.Linear(100, 1, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):        
        x = F.relu(self.fc1(x))
        logp = self.fc2(x)
        return torch.sigmoid(logp) # the output is p_up (sigmoid puts it into (0,1) )

pnet = PolicyNetK()
# This part will move the network to the NVIDIA GPU if you have one
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pnet.to(device)


#%%
# from Karpathy's post, preprocessing function
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2  
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()  

# from Karpathy's post, discounting rewards function
def discount_rewards(r):
  gamma = 0.99 # discount factor for reward
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


#%%
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
episode_number = 0
batch_size = 4
lr = 1e-4
p_ups, fake_labels, rewards = [], [], []


#%%
# Minimizing BCELoss is equivalent to maximizing logP if network outputs P.
# Note that this variant of loss expects probability not logits
# reduction='none' because we need to weight by advantage before reducing
# therefore, we'll reduce manually later


#%%
loss_computator = torch.nn.BCELoss(reduction='none') 
optimizer = optim.RMSprop(pnet.parameters(), lr=lr)
running_reward = None
optimizer.zero_grad()

import os.path
if os.path.isfile("model-2fc.pt"):
    pnet.load_state_dict(torch.load("model-2fc.pt"))

while True:
    if episode_number % 100 == 0 and episode_number>0: 
        torch.save(pnet.state_dict(), "model-2fc.pt")
        # to see your AI play game's built-in AI call env.render()
        env.render()
    
    # create input for the policy network (e.g. difference between two frames)
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x) 
    prev_x = cur_x    
    x = torch.from_numpy(x).float().to(device)         
    
    p_up = pnet(x) # probability of going UP. P_up         
    p_ups.append(p_up)
    action = 2 if np.random.uniform() < p_up.data[0] else 3
    y = 1.0 if action == 2 else 0.0 # fake label
    fake_labels.append(y)
        
    observation, reward, done, info = env.step(action)        
    rewards.append(reward) # reward for action can be seen after executing action
    
    if done: # an episode finished        
        episode_number += 1        
        eprewards = np.vstack(rewards)        
        discounted_epr = discount_rewards(eprewards)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)        

        lx = torch.stack(p_ups).float().to(device).squeeze(1)
        ly = torch.tensor(fake_labels).float().to(device)
        losses = loss_computator(lx, ly)
        t_discounted_epr = torch.from_numpy(discounted_epr).squeeze(1).float().to(device)
        losses *=  t_discounted_epr                
        loss = torch.mean(losses)         
        reward_sum = sum(rewards)
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if episode_number % 10 == 1: 
            print("EPNUM: {0}, LOSS: {1}. REWARDS: {2} RUNNING_REWARDS: {3}"
                  .format(episode_number, loss, reward_sum, running_reward))            
        
        loss.backward(torch.tensor(1.0/batch_size).to(device))
        if episode_number % batch_size == 0 and episode_number > 0:                        
            optimizer.step()
            optimizer.zero_grad()        
        
        p_ups, fake_labels, rewards = [], [], [] # reset
        observation = env.reset() # reset env
        prev_x = None       
    


#%%



