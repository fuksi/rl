import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from utils import discount_rewards, softmax_sample


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = 0.99

        # Input state, NN with 20 neurons, Output action
        self.fc1 = torch.nn.Linear(state_space, 128)
        self.fc2 = torch.nn.Linear(128, action_space)

        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        # model = torch.nn.Sequential(
        #     self.fc1,
        #     torch.nn.Dropout(p=0.6),
        #     torch.nn.ReLU(),
        #     self.fc2,
        #     torch.nn.Softmax(dim=-1)
        # )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Agent(object):
    def __init__(self, policy):
        # self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        # self.policy = policy.to(self.train_device)
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        self.observations = []
        self.actions = []
        self.rewards = []

    def episode_finished(self, episode_number):
        R = 0
        rewards = []
    
        # Discount future rewards back to the present using gamma
        for r in self.policy.reward_episode[::-1]:
            R = r + self.policy.gamma * R
            rewards.insert(0,R)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy.policy_history, Variable(rewards)).mul(-1), -1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy.loss_history.append(loss.data[0])
        self.policy.reward_history.append(np.sum(self.policy.reward_episode))
        self.policy.policy_history = Variable(torch.Tensor())
        self.policy.reward_episode= []

    def get_action(self, observation, evaluation=False):
        state = torch.from_numpy(observation).type(torch.FloatTensor)
        state = self.policy.forward(Variable(state))
        c = Categorical(state)
        action = c.sample()
        
        # Add log probability of our chosen action to our history    
        if self.policy.policy_history.dim() != 0:
            log_prob = c.log_prob(action).view(1)
            self.policy.policy_history = torch.cat((self.policy.policy_history, log_prob), 0)
        else:
            self.policy.policy_history = (c.log_prob(action))
        return action
        # x = torch.from_numpy(observation).float().to(self.train_device)
        # a_est = self.policy.forward(x)
        # a_prob = Normal(a_est, 0.5)
        
        # if evaluation:
        #     pass
        #     # action = torch.argmax(aprob).item()
        # else:
        # action = a_prob.sample()
        # return action, a_prob

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
