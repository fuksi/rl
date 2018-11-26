import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torchvision import transforms
import numpy as np
from utils import discount_rewards, softmax_sample
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        neurons = 20
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(4, neurons)
        self.action_head = torch.nn.Linear(neurons, 2)
        self.value_head = torch.nn.Linear(neurons, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-2)
        self.gamma = 0.98
        self.saved_actions = []
        self.rewards = []

    def episode_finished(self, episode_number):
        rreturn = 0
        saved_actions = self.saved_actions
        rewards, policy_losses, value_losses = [], [], []

        # calc G
        for r in self.rewards[::-1]:
            rreturn = r + self.gamma * rreturn
            rewards.insert(0, rreturn)

        # normalize rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        # calc loss
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

        # Update policy
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        # Reset grad, state
        self.optimizer.zero_grad()
        del self.rewards[:]
        del self.saved_actions[:]

    def get_action(self, observation, episode_number):
        state = torch.from_numpy(observation).float()
        probs, state_value = self.policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # Limit action space to [-25, 25]
        action = 50 * action - 25
        return action.item()

    def store_outcome(self, reward):
        self.rewards.append(reward)
