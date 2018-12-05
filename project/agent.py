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
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
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
        action_prob = self.policy.forward(ob)
        action = softmax_sample(action_prob)

        return action, action_prob 
        prob_up_float = prob_up.tolist()[0]

    def store_outcome(self, reward, action_taken, action_log_prob):
        dist = torch.distributions.Categorical(action_log_prob)
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        log_action_prob = -dist.log_prob(action_taken)
        self.actions.append(log_action_prob)
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
        state_space = 5
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
        return F.softmax(x, dim=-1)
