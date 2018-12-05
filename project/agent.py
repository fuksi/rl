from pong import Pong
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import random
import numpy as np
from utils import discount_rewards, softmax_sample

class NaiveAI(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.train_device = 'cpu'
        self.policy = Policy()
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=5e-3)
        self.player_id = player_id
        self.bpe = 4
        self.name = "NaiveAI"
        self.gamma = 0.98
        self.actions = []
        self.rewards = []

    def get_name(self):
        return self.name

    def get_action(self, ob, episode_number):
        x = torch.from_numpy(ob).float().to(self.train_device) 
        prob_up = self.policy.forward(x)
        prob_up_float = prob_up.tolist()[0]
        
        temp = np.random.uniform()
        action = 1 if temp < prob_up_float else 2

        # 0: stay, 1: up, 2: down
        return action, prob_up

    def store_outcome(self, reward, action_log_prob):
        self.actions.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))

    def episode_finished(self):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.actions, self.rewards = [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        weighted_probs = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        loss.backward()

        # Update policy
        self.optimizer.step()
        self.optimizer.zero_grad()

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        state_space = 10500
        action_space = 1
        neurons = 200
        self.fc1 = torch.nn.Linear(state_space, neurons)
        self.fc2 = torch.nn.Linear(neurons, action_space)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
