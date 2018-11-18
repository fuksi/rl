import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import transforms
import numpy as np
from utils import discount_rewards, softmax_sample


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        neurons = 20 
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, neurons)
        self.fc2 = torch.nn.Linear(neurons, action_space)
        self.fc3 = torch.nn.Linear(neurons, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2(x)
        var = self.fc3(x)

        return torch.sigmoid(mu), torch.sigmoid(var)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.observations = []
        self.actions = []
        self.rewards = []

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        weighted_probs = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        loss.backward()

        # Update policy
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, episode_number):
        x = torch.from_numpy(observation).float().to(self.train_device)
        mu, var = self.policy.forward(x)

        # decay var exponential with decay alpha 0.001
        var = var * np.exp(-0.0001 * episode_number)

        dist = Normal(torch.tensor(mu), torch.tensor(var))
        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        # Limit to [-25, 25] action space
        action = 50 * sample - 25
        return action, -log_prob

    def store_outcome(self, reward, action_log_prob):
        self.actions.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
