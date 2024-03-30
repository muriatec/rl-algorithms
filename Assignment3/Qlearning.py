import numpy as np
from collections import defaultdict
import sys
from cliffwalking import CliffWalking
from pprint import pprint

def epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        action_space = np.ones(nA, dtype=float) * epsilon / nA
        optimal_action = np.argmax(Q[state])
        action_space[optimal_action] += (1.0 - epsilon)
        return action_space
    return policy_fn

def Q_learning(env, num_episodes, learning_rate, epsilon, discount_factor=1.0):
    print("Learning rate: {}, Epsilon: {}".format(learning_rate, epsilon))
    # Initialize Q
    Q = defaultdict(lambda: np.zeros(env.nA))

    policy = epsilon_greedy_policy(Q, epsilon, env.nA)

    for episode_index in range(1, num_episodes + 1):
        if episode_index % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode_index, num_episodes), end="")
            sys.stdout.flush()
        
        state = env.reset()

        while True:
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done = env.step(action)

            optimal_next_action = np.argmax(Q[next_state])
            TD_target = reward + discount_factor * Q[next_state][optimal_next_action]
            TD_error = TD_target - Q[state][action]
            Q[state][action] += learning_rate * TD_error

            if done:
                break

            state = next_state

    return Q

env = CliffWalking()
Q = Q_learning(env, num_episodes=10000, learning_rate=0.5, epsilon=0.1)

policy = []
for i in sorted(Q):
    policy.append(np.argmax(Q[i]))
policy = np.reshape(np.asarray(policy), env.shape)

def visualize_policy(policy):
    vis_policy_opt = []
    vis_policy_opt_tmp = []
    index = 0

    for action_index in np.nditer(policy):
        index += 1
        action = None
        if action_index == 0:
            action = 'N'
        if action_index == 1:
            action = 'E'
        if action_index == 2:
            action = 'S'
        if action_index == 3:
            action = 'W'
        vis_policy_opt_tmp.append(action)

        if index % policy.shape[1] == 0:
            vis_policy_opt.append(vis_policy_opt_tmp)
            vis_policy_opt_tmp = []

    print('\n')
    pprint(vis_policy_opt)

visualize_policy(policy)
