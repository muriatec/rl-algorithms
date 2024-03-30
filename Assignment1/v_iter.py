import numpy as np
from gridworld import GridWorld
from pprint import pprint

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    while True:
        delta = 0
        
        for s in range(env.nS):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value        

        if delta < theta:
            break
    
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    
    return policy, V

env = GridWorld(shape=(6,6))

policy_opt, value_opt = value_iteration(env)
policy_opt = np.reshape(np.argmax(policy_opt, axis=1), env.shape)
# print(policy_opt)

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

        if index % policy.shape[0] == 0:
            vis_policy_opt.append(vis_policy_opt_tmp)
            vis_policy_opt_tmp = []

    pprint(vis_policy_opt)

visualize_policy(policy_opt)
print(value_opt.reshape(env.shape))