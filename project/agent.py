from pong import Pong
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import random

class NaiveAI(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.train_device = 'cpu'
        self.policy = Policy()
        self.player_id = player_id
        self.bpe = 4
        self.name = "NaiveAI"

    def get_name(self):
        return self.name

    def get_action(self, ob=None):
        if ob.any():
            x = torch.from_numpy(ob).float().to(self.train_device) 
            x = self.policy.forward(x)
            x_prob = 0.99
            other_prob = (1-x_prob) / 2
            inital_dist = [other_prob, x_prob, other_prob]
            dist = Categorical(torch.tensor(inital_dist))
            action = dist.sample()
            action_prob = dist.log_prob(action) 

            # 0: stay, 1: up, 2: down
            return action
        else:
            player = self.env.player1 if self.player_id == 1 else self.env.player2
            my_y = player.y
            # 6
            ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)
            # ball 4 ->  [2, 6] 
            y_diff = my_y - ball_y

            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down

            return action

    def reset(self):
        # Nothing to done for now...
        return


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
        return F.softmax(x, dim=-1)
