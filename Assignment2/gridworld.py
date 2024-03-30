import sys
import io
import numpy as np
import random

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text.utils import categorical_sample

class DiscreteEnv(Env):
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, current_state, action):
        transitions = self.P[current_state][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.lastaction = action

        return int(s), r, d


class GridWorld(DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[6,6]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        N = 0
        E = 1
        S = 2
        W = 3

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 1 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # terminal state
            if is_done(s):
                P[s][N] = [(1.0, s, reward, True)]
                P[s][E] = [(1.0, s, reward, True)]
                P[s][S] = [(1.0, s, reward, True)]
                P[s][W] = [(1.0, s, reward, True)]
            # non-terminal state
            else:
                next_s_north = s if y == 0 else s - MAX_X
                next_s_east = s if x == (MAX_X - 1) else s + 1
                next_s_south = s if y == (MAX_Y - 1) else s + MAX_X
                next_s_west = s if x == 0 else s - 1
                P[s][N] = [(1.0, next_s_north, reward, is_done(next_s_north))]
                P[s][E] = [(1.0, next_s_east, reward, is_done(next_s_east))]
                P[s][S] = [(1.0, next_s_south, reward, is_done(next_s_south))]
                P[s][W] = [(1.0, next_s_west, reward, is_done(next_s_west))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        self.P = P

        super(GridWorld, self).__init__(nS, nA, P, isd)

    # def step(self, action):
    #     nS = self.nS
    #     is_done = lambda s: s == 1 or s == (nS - 1)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 1 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()