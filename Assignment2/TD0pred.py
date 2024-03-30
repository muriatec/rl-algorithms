import sys
import numpy as np
import random
from gridworld import GridWorld


def TemporalDifference0(env, policy, num_episodes, discount_factor, learning_rate):
    V = np.zeros(env.nS)

    for episode_index in range(1, num_episodes + 1):
        if episode_index % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode_index, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()

        while True:
            action = random.choice(policy[state])
            next_state, reward, done = env.step(state, action)

            V[state] = V[state] + learning_rate * (reward + discount_factor * V[next_state] - V[state])
            state = next_state

            if done:
                break

    # V = np.around(V)

    return V

env = GridWorld((6,6))
policy = [[i for i in range(env.nA)] for s in range(env.nS)]
eval_v = TemporalDifference0(env, policy, num_episodes=100000, discount_factor=0.9, learning_rate=1e-5)
print('\n', eval_v.reshape(env.shape))