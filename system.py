import numpy as np
import numpy.random as rn
import simpy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import measurements as mm
import random as rnd


class state:
    def __init__(self, L, q, dp, dq, agent_num):
        self.L  = L
        self.q  = q
        self.dp = dp
        self.dq = dq

        self.rng      = rn.RandomState(int(rnd.random()*100))
        self.parents  = self.rng.rand(self.L, self.L)
        self.weights  = self.rng.rand(self.L, self.L)
        self.children = self.rng.rand(self.L, self.L)

        self.agent_num = agent_num          # basic agent number
        self.agents    = []                 # these are the agents
        self.blacklist = np.zeros((self.L, self.L))
        
        self.parcolating = None
        self.purchased = None

    def gen_agents(self):
        self.agents = []

        for i in range(self.agent_num):
            x, y = (rn.randint(self.L), rn.randint(self.L))
            if self.blacklist[x][y] == 0:
                self.agents.append((x, y))
                self.blacklist[x][y] = 1
        
    def neighbors(self, pair, L):
        x, y = pair
        for i in [1, -1]:
            xx = x + i
            if 0 <= xx < self.L:
                yield (xx, y)
        for i in [1, -1]:
            yy = y + i
            if 0 <= yy < self.L:
                yield (x, yy)

    def increment_time(self):
        purchased = np.zeros((self.L, self.L))
        self.gen_agents()

        while len(self.agents) > 0:

            for agent in self.agents:
                x, y = agent
                p    = self.parents[x][y]
                c    = self.children[x][y]
                w    = self.weights[x][y]

                # decision value
                decision = c * w + p * (1-w)
                # decision = p

                self.agents.remove(agent)

                if decision < self.q:
                    purchased[x][y] = 1
                    # adding the purchasers neighbors to the list
                    for neighbor in self.neighbors(agent, self.L):
                        xx, yy = neighbor
                        if self.blacklist[xx][yy] == 0:
                            self.agents.append(neighbor)
                            self.blacklist[xx][yy] = 1

        self.increment_values(purchased)

        # resetting all the temporary stuffs
        self.blacklist = np.zeros((self.L, self.L))
        self.parcolating = None

        self.purchased = purchased


    def increment_values(self, purchased):
        if self.is_percolating(purchased):
            self.q -= self.dq
        else:
            self.q += self.dq
        
        L = self.L
        for i in range(L):
            for j in range(L):
                if purchased[i][j]:
                    self.children[i][j] += self.dp
                    # self.parents[i][j] += self.dp
                else:
                    self.children[i][j] -= self.dp
                    # self.parents[i][j] -= self.dp
    
    def check_if_matches(self, a1, a2, n):
        a = [0 for i in range(n+1)]
        for i in a1:
            a[i] = 1
        for i in a2:
            if i != 0 and a[i] == 1:
                return True
        return False

    def is_percolating(self, array):
        lw, num = mm.label(array)
        t, b = lw[0], lw[-1]

        l, r = [], []
        for i in range(self.L):
            l.append(lw[i][0])
            r.append(lw[i][-1])

        if not self.check_if_matches(l, r, num) and not self.check_if_matches(t, b, num):
            return False
        return True
    

def simulate(state, time):
    for i in range(time):
        state.increment_time()
        print(state.q)

state = state(100, .5654, 0.00001, 0.0001, 100)
simulate(state, 10000)

