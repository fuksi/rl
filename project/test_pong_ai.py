from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi
from agent import NaiveAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

def plot(observation):
    plt.imshow(observation/255)
    plt.show()

# env = Pong(headless=args.headless)
env = Pong(headless=True)
episodes = 1000000

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = NaiveAI(env, player_id)

env.set_names(player.get_name(), opponent.get_name())

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  M, N, R = I.shape
  for m in range(0,M):
    for n in range(0,N):
        color = I[m][n].tolist()
        if color != [0.0, 0.0, 0.0]:
            I[m][n][0] = 255
  I = I[::3,::3,0] # downsample by factor of 2
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

for episode_number in range(0,episodes):
    reward_sum = 0
    done = False
    ob1 = None
    prev_state_1 = None
    while not done:
        # Init current state if not exist 
        cur_state_1 = prepro(ob1) if ob1 is not None else prepro(np.zeros((210, 200, 3)))
        # Input is diff between state
        input_1 = cur_state_1 - prev_state_1 if prev_state_1 is not None else np.zeros_like(cur_state_1)
        # Save prev_state
        prev_state_1 = cur_state_1

        action1 = player.get_action(input_1)
        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        reward_sum += rew1
        player.store_outcome(rew1)
        if done:
            observation= env.reset()
    
    player.episode_finished(episode_number)

# Needs to be called in the end to shut down pygame
env.end()


