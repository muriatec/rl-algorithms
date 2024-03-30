from configparser import ConfigParser
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import gym
import numpy as np
from network import Actor, Critic
from utils import ReplayBuffer, convert_to_tensor, make_transition, Dict, RunningMeanStd, OUNoise

class DDPG(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args, noise):
        super(DDPG,self).__init__()
        self.device = device
        self.writer = writer
        
        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std)
        
        self.target_actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std)
        
        self.q = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, None)
        
        self.target_q = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, None)
        
        self.soft_update(self.q, self.target_q, 1.)
        self.soft_update(self.actor, self.target_actor, 1.)
        
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.args.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim)
        
        self.noise = noise
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
            
    def get_action(self,x):
        return self.actor(x)[0] + torch.tensor(self.noise.sample()).to(self.device), self.actor(x)[1]
    
    def put_data(self,transition):
        self.data.put_data(transition)
        
    def train_net(self, batch_size, n_epi):
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        targets = rewards + self.args.gamma * (1 - dones) * self.target_q(next_states, self.target_actor(next_states)[0])
        q_loss = F.smooth_l1_loss(self.q(states,actions), targets.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        actor_loss = - self.q(states, self.actor(states)[0]).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.q, self.target_q, self.args.soft_update_rate)
        self.soft_update(self.actor, self.target_actor, self.args.soft_update_rate)
        if self.writer != None:
            self.writer.add_scalar("loss/q", q_loss, n_epi)
            self.writer.add_scalar("loss/actor", actor_loss, n_epi)


os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'Hopper-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default = 'ddpg', help = 'algorithm to adjust (default : ddpg)')
parser.add_argument('--train', dest='train', action='store_true', help="train model")
parser.add_argument('--test', dest='test', action='store_true', help="test model")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: True)')
parser.add_argument("--model_path", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser,args.algo)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
    
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None
    
env = gym.make(args.env_name, render_mode='human')
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)

noise = OUNoise(action_dim,0)
agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

    
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.model_path != 'no':
    agent.load_state_dict(torch.load("./model_weights/" + args.model_path))
    
score_lst = []
state_lst = []

for n_epi in range(args.epochs):
    score = 0.0
    state = env.reset()[0]
    done = False
    while not done:
        if args.render:    
            env.render()
        action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
        action = action.cpu().detach().numpy()
        next_state, reward, done, info, _ = env.step(action)
        transition = make_transition(state,\
                                        action,\
                                        np.array([reward*args.reward_scaling]),\
                                        next_state,\
                                        np.array([done])\
                                    )
        agent.put_data(transition) 

        state = next_state

        score += reward
        if agent.data.data_idx > agent_args.learn_start_size and args.train == True: 
            agent.train_net(agent_args.batch_size, n_epi)
    score_lst.append(score)
    if args.tensorboard:
        writer.add_scalar("score/score", score, n_epi)
    if (n_epi + 1) % args.print_interval == 0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst) / len(score_lst)))
        score_lst = []
    if (n_epi + 1) % args.save_interval == 0 and args.train == True:
        torch.save(agent.state_dict(),'./model_weights/agent_' + str(n_epi + 1))