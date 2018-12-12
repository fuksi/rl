from pong import Pong
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import random
import numpy as np
import os
from utils import discount_rewards, softmax_sample


class NaiveAI(object):
    def __init__(self, env, player_id=1):
        self.env = env

        # Init policy
        learningrate = 1e-4
        self.train_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.policy = Policy()
        self.policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(), lr=learningrate)
        self.loss = torch.nn.BCELoss(reduction='none')

        # Load model if exists
        self.load_model("model-nn.pt")

        # Misc
        self.player_id = player_id
        self.name = "NaiveAI"
        self.gamma = 0.99
        self.batch_size = 20 
        self.prop_ups = []
        self.rewards = []
        self.fake_labels = []
        self.running_reward = None

    def load_model(self, filename):
        if os.path.isfile(filename):
            self.policy.load_state_dict(torch.load(filename))

    def get_name(self):
        return self.name

    def get_action(self, input):
        x = torch.from_numpy(input).float().to(self.train_device)
        prob_up = self.policy(x)
        self.prop_ups.append(prob_up)

        # Sample action based on prob_up with some random probability of picking others
        # 0: stay, 1: up, 2: down, we'll ignore stay action for now
        prob_up_float = prob_up.tolist()[0]
        action = 1 if np.random.uniform() < prob_up_float else 2

        # Fake label. Read more here https://medium.com/deep-math-machine-learning-ai/ch-13-deep-reinforcement-learning-deep-q-learning-and-policy-gradients-towards-agi-a2a0b611617e
        fake_label = 1.0 if action == 1 else 0.0
        self.fake_labels.append(fake_label)

        return action

    def store_outcome(self, reward):
        self.rewards.append(torch.Tensor([reward]))

    def episode_finished(self, episode_number):
        # Save nn every 200th eps
        if episode_number % 200 == 0 and episode_number > 0:
            torch.save(self.policy.state_dict(), 'model-nn.pt')

        # Calc discounted rewards
        all_rewards = torch.stack(self.rewards, dim=0).to(
            self.train_device).squeeze(1)
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # Stack prop_ups, fake_labels, measure losses
        all_actions = torch.stack(self.prop_ups).float().to(
            self.train_device).squeeze(1)
        all_labels = torch.tensor(
            self.fake_labels).float().to(self.train_device)
        losses = self.loss(all_actions, all_labels)
        losses *= discounted_rewards
        loss = torch.mean(losses)

        # Reset
        self.reset()

        # Compute grad
        loss.backward(torch.tensor(1.0/self.batch_size).to(self.train_device))

        # Output loss and rewards every now and then
        reward_sum = sum(all_rewards)
        if episode_number % 10 == 1:
            print(f'Episode: {episode_number}, Loss: {loss}. Rewards: {reward_sum}')

        # Update policy depends on batch size
        if episode_number % self.batch_size == 0 and episode_number > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def reset(self):
        self.prop_ups, self.rewards, self.fake_labels = [], [], []



class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        state_space = 4690
        action_space = 1
        neurons = 100
        self.fc1 = torch.nn.Linear(state_space, neurons, bias=False)
        self.fc2 = torch.nn.Linear(neurons, action_space, bias=False)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
