import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
from utils import get_space_dim

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
parser.add_argument("--train_episodes", type=int, default=20000, help="Number of episodes to train for")
parser.add_argument("--render_training", action='store_true',
                    help="Render each frame during training. Will be slower.")
args = parser.parse_args()


# Policy training function
def train(train_episodes, agent):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # observation = env.step(action)
            observation, reward, done, info = env.step(action.detach().numpy())
            # action_array = action_probabilities.detach().numpy()
            # observation, reward, done, info = env.step(action)
            # observation, reward, done, info = env.step(action_array)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if args.render_training:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
              .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 30 full episodes, assume it's learned
        # (in the default setting)
        if np.mean(timestep_history[-30:]) == env._max_episode_steps:
            print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Training is finished - plot rewards
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.legend(["Reward", "100-episode average"])
    plt.title("Reward history (sig=%f, net 18)" % agent.policy.sigma.item())
    plt.show()
    print("Training finished.")


# Create a Gym environment
env = CartPoleEnv()

# For CartPole - maximum episode length
env._max_episode_steps = 1000

# Get dimensionalities of actions and observations
# action_space_dim = 1
# observation_space_dim = 4
action_space_dim = get_space_dim(env.action_space)
observation_space_dim = get_space_dim(env.observation_space)

# Create the agent and the policy
policy = Policy(observation_space_dim, action_space_dim)
agent = Agent(policy)
# agent = None
# policy = None

# If no model was passed, train a policy from scratch.
# Otherwise load the policy from the file and go directly to testing.
train(args.train_episodes, agent)
model_file = "%s_params.mdl" % args.env
torch.save(policy.state_dict(), model_file)
print("Model saved to", model_file)
# if args.test is None:
#     try:
#         train(args.train_episodes, agent)
#     # Handle Ctrl+C - save model and go to tests
#     except KeyboardInterrupt:
#         print("Interrupted!")
#     model_file = "%s_params.mdl" % args.env
#     torch.save(policy.state_dict(), model_file)
#     print("Model saved to", model_file)
# else:
#     state_dict = torch.load(args.test)
#     policy.load_state_dict(state_dict)


