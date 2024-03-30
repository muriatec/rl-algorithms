import sys
import io
import numpy as np

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

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a

        return (int(s), r, d)


class CliffWalking(DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    
    def calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self.limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = -100.0 if self._cliff[tuple(new_position)] else -1.0
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (3,11))
        return [(1.0, new_state, reward, is_done)]

    def __init__(self, shape=(4,12)):
        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        N = 0
        E = 1
        S = 2
        W = 3

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][N] = self.calculate_transition_prob(position, [-1, 0])
            P[s][E] = self.calculate_transition_prob(position, [0, 1])
            P[s][S] = self.calculate_transition_prob(position, [1, 0])
            P[s][W] = self.calculate_transition_prob(position, [0, -1])

        # Start in (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        self.P = P

        super(CliffWalking, self).__init__(nS, nA, P, isd)

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
            elif s == self.nS - 12:
                output = " S "
            elif s == self.nS - 1:
                output = " G "
            elif s >= (self.nS - 11) and s <= (self.nS - 2):
                output = " C "
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

# env = CliffWalking()
# env._render('human')