import sys
import numpy as np
import random
from gridworld import GridWorld


def firstVisitMonteCarlo(env, policy, num_episodes, discount_factor):
    Returns = {i:0 for i in range(env.nS)}
    N = {j:0 for j in range(env.nS)}
    V = np.zeros(env.nS)

    for episode_index in range(1, num_episodes + 1):
        if episode_index % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode_index, num_episodes), end="")
            sys.stdout.flush()
        # Generate an episode, which is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        G = 0

        for step in range(100):
            action = random.choice(policy[state])
            next_state, reward, done = env.step(state, action)

            actions = ['N', 'E', 'S', 'W']
            action = actions[action]

            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        for index, state_tuple in enumerate(episode[::-1]):
            G = discount_factor * G + state_tuple[2]
            # first visit check
            if str(state_tuple[0]) not in np.asarray(episode)[:, 0][:len(episode)-index-1]:
                # Returns[state_tuple[0]] += G
                N[state_tuple[0]] += 1
                # incremental update
                V[state_tuple[0]] = V[state_tuple[0]] + (G - V[state_tuple[0]]) / N[state_tuple[0]]
    
    # for s in range(env.nS):
    #     V[s] = round(Returns[s] / N[s])
    # V = np.around(V)

    return V


def everyVisitMonteCarlo(env, policy, num_episodes, discount_factor):
    Returns = {i:0 for i in range(env.nS)}
    N = {j:0 for j in range(env.nS)}
    V = np.zeros(env.nS)

    for episode_index in range(1, num_episodes + 1):
        if episode_index % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode_index, num_episodes), end="")
            sys.stdout.flush()
        # Generate an episode, which is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        G = 0

        for step in range(100):
            action = random.choice(policy[state])
            next_state, reward, done = env.step(state, action)

            actions = ['N', 'E', 'S', 'W']
            action = actions[action]

            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        for index, state_tuple in enumerate(episode[::-1]):
            G = discount_factor * G + state_tuple[2]

            # Returns[state_tuple[0]] += G
            N[state_tuple[0]] += 1
            # incremental update
            V[state_tuple[0]] = V[state_tuple[0]] + (G - V[state_tuple[0]]) / N[state_tuple[0]]
    
    # for s in range(env.nS):
    #     V[s] = round(Returns[s] / N[s])
    # V = np.around(V)

    return V


env = GridWorld((6,6))
policy = [[i for i in range(env.nA)] for s in range(env.nS)]
# eval_v_f = firstVisitMonteCarlo(env, policy, num_episodes=10000, discount_factor=0.9)
# print('\n', eval_v_f.reshape(env.shape))
eval_v_e = everyVisitMonteCarlo(env, policy, num_episodes=10000, discount_factor=0.9)
print('\n', eval_v_e.reshape(env.shape))