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


env = Pong(headless=args.headless)
episodes = 100000000000

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = NaiveAI(env, player_id)

env.set_names(player.get_name(), opponent.get_name())

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[::2,::2,0] # downsample by factor of 2
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

for episode_number in range(0,episodes):
    reward_sum = 0
    done = False
    ob1 = np.zeros((210, 200, 3))
    while not done:
        state = prepro(ob1)
        action1, action1_prob = player.get_action(state, episode_number)
        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        player.store_outcome(rew1, action1_prob)
        # for idx, row in enumerate(ob1):
        #     for col in row:
        #         if col[0] == 255 and col[1] == 255 and col[2] == 255:
        #             print('vow')
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            #plot(ob1) # plot the reset observation
            print("episode {} over".format(episode_number))
    
    player.episode_finished()

# Needs to be called in the end to shut down pygame
env.end()


