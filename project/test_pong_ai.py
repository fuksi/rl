from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import random
import pickle
import numpy as np
from simple_ai import PongAi
from agent import NaiveAI
import argparse
import torch
from math import floor

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

y_pos_min = 0
y_pos_max = 255
x_pos_min = -10 
x_pos_max = 210 

p1_y_grid = np.linspace(y_pos_min, y_pos_max, num=256)
ball_y_grid = np.linspace(y_pos_min, y_pos_max, num=256)
ball_x_grid = np.linspace(x_pos_min, x_pos_max, num=221)
ball_y_dir = [1, -1]
ball_x_dir = [1, -1]

action_dim = 1

q_grid = np.zeros((256, 256, 221, 2, 2, 3))



def plot(observation):
    plt.imshow(observation/255)
    plt.show()


env = Pong(headless=args.headless)
episodes = 100000000000

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = NaiveAI(env, player_id)

def get_p1_state():
    p1_y_pos = round(env.player1.y)
    ball_y_pos = round(env.ball.y)
    ball_x_pos = round(env.ball.x)
    ball_y_dir = 1 if not prev_ball_y_pos else (1 if env.ball.y > prev_ball_y_pos else -1)
    ball_x_dir = 1 if not prev_ball_x_pos else (1 if env.ball.x > prev_ball_x_pos else -1)

    return [p1_y_pos, ball_y_pos, ball_x_pos, ball_y_dir, ball_x_dir]

env.set_names(player.get_name(), opponent.get_name())
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.0002
alpha = 0.1
gamma = 0.99
for episode_number in range(0,episodes):
    reward_sum = 0
    done = False
    prev_ball_y_pos = None
    prev_ball_x_pos = None
    while not done:
        is_exploitation = random.uniform(0, 1) > epsilon

        p1_state = get_p1_state()
        action_values = q_grid[p1_state[0]][p1_state[1]][p1_state[2]][p1_state[3]][p1_state[4]]
        # action1, action1_prob = player.get_action(state1, episode_number)

        if is_exploitation:
            action1 = np.argmax(action_values)
        else:
            action1 = floor(random.random()*3)

        val = action_values[action1]
        if val > 0:
            foo = 5
        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        if rew1 == 0:
            rew1 = 1
        reward_sum += rew1

        # Update Q value
        p1_state_new = get_p1_state()
        new_action_values = q_grid[p1_state_new[0]][p1_state_new[1]][p1_state_new[2]][p1_state_new[3]][p1_state_new[4]]
        prev_q = action_values[action1]

        new_q = prev_q + alpha * (rew1 + gamma * np.max(new_action_values) - prev_q)
        action_values[action1] = new_q
        # if not args.headless:
        #     env.render()
        if done:
            observation= env.reset()
            #plot(ob1) # plot the reset observation
            print(f'episode {episode_number} over. Total rewards: {reward_sum}')
    
    # player.episode_finished()
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode_number)

# Needs to be called in the end to shut down pygame
env.end()


