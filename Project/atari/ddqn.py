import argparse
import os
import random
import torch
import numpy as np
import math
from torch import nn
from torch.optim import Adam
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from utils import get_class_attr_val
from utils import get_output_folder
from logger import TensorBoardLogger

class Config:
    env: str = None
    gamma: float = None
    learning_rate: float = None
    frames: int = None
    episodes: int = None
    max_buff: int = None
    batch_size: int = None

    epsilon: float = None
    eps_decay: float = None
    epsilon_min: float = None

    state_dim: int = None
    state_shape = None
    state_high = None
    state_low = None
    seed = None
    output = 'out'

    action_dim: int = None
    action_high = None
    action_low = None
    action_lim = None

    use_cuda: bool = None

    checkpoint: bool = False
    checkpoint_interval: int = None

    record: bool = False
    record_ep_interval: int = None

    log_interval: int = None
    print_interval: int = None

    update_tar_interval: int = None

    win_reward: float = None
    win_break: bool = None

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done

    def size(self):
        return len(self.buffer)

class Network(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(Network, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

class DDQN:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.model = Network(self.config.state_shape, self.config.action_dim)
        self.target_model = Network(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()
        next_q_state_values = self.target_model(s1).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, pre_fr=0):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset()
        for fr in range(pre_fr + 1, self.config.frames + 1):
            epsilon = self.epsilon_by_frame(fr)
            action = self.agent.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = 0
            if self.agent.buffer.size() > self.config.batch_size:
                loss = self.agent.learning(fr)
                losses.append(loss)
                self.board_logger.scalar_summary('Loss per frame', fr, loss)

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (fr, np.mean(all_rewards[-10:]), loss, ep_num))

            if fr % self.config.log_interval == 0:
                self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')

class Tester(object):

    def __init__(self, agent, env, model_path, num_episodes=50, max_ep_steps=400, test_ep_steps=100):
        self.num_episodes = num_episodes
        self.max_ep_steps = max_ep_steps
        self.test_ep_steps = test_ep_steps
        self.agent = agent
        self.env = env
        self.agent.is_training = False
        self.agent.load_weights(model_path)
        self.policy = lambda x: agent.act(x)

    def test(self, debug=True, visualize=True):
        avg_reward = 0
        for episode in range(self.num_episodes):
            s0 = self.env.reset()
            # episode_steps = 0
            episode_reward = 0.

            done = False
            while not done:
                if visualize:
                    self.env.render()

                action = self.policy(s0)
                s0, reward, done, info = self.env.step(action)
                episode_reward += reward
                # episode_steps += 1

                # if episode_steps + 1 > self.test_ep_steps:
                #     done = True

            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))

            avg_reward += episode_reward
        avg_reward /= self.num_episodes
        print("avg reward: %5f" % (avg_reward))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env_name', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()
    # atari_ddqn.py --train --env PongNoFrameskip-v4

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 30000
    config.frames = 2000000
    config.use_cuda = True
    config.learning_rate = 1e-4
    config.max_buff = 100000
    config.update_tar_interval = 1000
    config.batch_size = 32
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18  # PongNoFrameskip-v4
    config.win_break = True

    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    agent = DDQN(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path, num_episodes=10)
        tester.test()