import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import transforms
import numpy as np
from utils import discount_rewards, softmax_sample


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        # Input state, NN with 20 neurons, Output action
        self.fc1 = torch.nn.Linear(state_space, 20)
        self.fc2 = torch.nn.Linear(20, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return Normal(x, 1)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.batch_size = 1
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

        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        aprob = self.policy.forward(x)
        if evaluation:
            pass
            # action = torch.argmax(aprob).item()
        else:
            action = aprob.sample()
        return action, aprob

    def store_outcome(self, observation, action_output, action_taken, reward):
        dist = action_output
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        log_action_prob = -dist.log_prob(action_taken)

        self.observations.append(observation)
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))

# class Policy(torch.nn.Module):
#     def __init__(self, state_space, action_space):
#         super().__init__()
#         # Create layers etc
#         # Initialize neural network weights
#         self
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if type(m) is torch.nn.Linear:
#                 torch.nn.init.uniform_(m.weight)
#                 torch.nn.init.zeros_(m.bias)

#     def forward(self, x):
#         return 0


# class Agent(object):
#     def __init__(self):
#         return

#     def episode_finished(self, episode_number):
#         return

#     def get_action(self, state, evaluation=False):
#         return np.random.random()*10-5
