import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9 
GAMMA = 0.9
TARGET_UPDATE_FREQ = 50
MEMORY_CAPACITY = 50000
env = gym.make('MountainCar-v0', render_mode='human')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
N_EPISODES = 500
MAX_STEPS = 500

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.out = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_prob = self.out(x)
        return actions_prob


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     
        self.memory_counter = 0                                         
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        hyper_parameters = f'LR={LR}-Memory={MEMORY_CAPACITY}-Update@{TARGET_UPDATE_FREQ}'
        self.writer = SummaryWriter(f'./logs/dqn/{hyper_parameters}')

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_prob = self.eval_net.forward(x)
            action = torch.max(actions_prob, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.writer.add_scalar('train/loss', loss, self.learn_step_counter)
        self.learn_step_counter += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = DQN()

print('\nCollecting experience...')
for i_episode in range(N_EPISODES):
    s = env.reset()[0]
    ep_r = 0
    steps = 0

    while True:
        env.render()
        a = agent.choose_action(s)

        # take action
        next_s, r, done, info, _ = env.step(a)

        # update the memory buffer
        agent.store_transition(s, a, r, next_s)

        ep_r += r
        steps += 1

        # if agent.memory_counter > MEMORY_CAPACITY:
        # update the Q network
        agent.learn()

        if steps == MAX_STEPS:
            done = True

        if done:
            if steps < MAX_STEPS:
                result = 'Success!'
            else: result = 'Failure!'
            print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2),
                      '| Ep_steps: ', steps, '|', result)
            agent.writer.add_scalar('train/total_steps', steps, i_episode)
            agent.writer.add_scalar('train/episode_rewards', round(ep_r, 2), i_episode)

            break

        s = next_s