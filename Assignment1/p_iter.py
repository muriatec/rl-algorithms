import numpy as np
from gridworld import GridWorld
from pprint import pprint

def policy_eval(policy, env, discount_factor=1.0, theta=1e-5):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for  prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        
        policy_stable = True
        
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        if policy_stable:
            return policy, V
        

env = GridWorld(shape=(6,6))

# evaluation of initial random policy
uniform_random_policy = np.ones([env.nS, env.nA]) / env.nA
value = policy_eval(uniform_random_policy, env)
# print(np.around(value.reshape(env.shape)))

# policy iteration
policy_opt, value_opt = policy_improvement(env)
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