import numpy as np
import numpy.random as rn
import simpy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import measurements as mm
import random as rnd


class state:
    def __init__(self, L, q, dp, dq):
        self.L  = L
        self.q  = q
        self.dp = dp
        self.dq = dq

        self.rng      = rn.RandomState(int(rnd.random()*100))
        self.parents  = self.rng.rand(self.L, self.L)
        self.weights  = self.rng.rand(self.L, self.L)
        self.children = self.rng.rand(self.L, self.L)

        self.agents    = []                 # these are the agents
        self.blacklist = []                 # they have made their decisions
        self.purchased = np.zeros((L, L))   # these have made purchases

    def gen_agents(self, agent_num):        # generates some random agents
        for i in range(agent_num):
            pair = (rn.randint(self.L), rn.randint(self.L))
            if pair not in self.agents:
                self.agents.append(pair)
        
    def is_percolating(self):               # checks if the current state is percolating
        array = self.purchased

        top = False
        bottom = False

        for i in array[0]:
            if i == 1:
                top = True

        for i in array[-1]:
            if i == 1:
                bottom = True

        if top and bottom:
            return True
        return False

    def update(self, agent_num):
        L = self.L
        for i in range(L):
            for j in range(L):
                if self.purchased[i][j]:
                    # self.children[i][j] += self.dp
                    self.parents[i][j] += self.dp
                else:
                    # self.children[i][j] -= self.dp
                    self.parents[i][j] -= self.dp

        if self.is_percolating():
            self.q -= self.dq
        else:
            self.q += self.dq

        # resetting all the temporary stuffs
        self.gen_agents(agent_num)
        self.purchased = np.zeros((L, L))
        self.blacklist = []

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

    def run_simulation(self):
        while len(self.agents) > 0:
            # for all agent, let them make the decision remove them from agents, add them
            # to blacklist if they purchased, add their neighbors to agents if not they
            # are in blacklist.

            for agent in self.agents:
                x, y = agent
                p    = self.parents[x][y]
                c    = self.children[x][y]
                w    = self.weights[x][y]

                # decision value
                # decision = c * w + p * (1-w)
                decision = p

                self.agents.remove(agent)
                self.blacklist.append(agent)

                if decision < self.q:
                    self.purchased[x][y] = True
                    # adding the purchasers neighbors to the list
                    for i in self.neighbors(agent, self.L):
                        if i not in self.agents and i not in self.blacklist:
                            self.agents.append(i)


run1 = state(20, .2, 1e-5, 1e-3)
for i in range(5000):
    run1.gen_agents(10)
    run1.run_simulation()
    # print(run1.purchased)
    # print(run1.is_percolating())
    run1.update(10)
    if i%50 == 0:
        print(run1.q)


# wild wild results, doesnt seem to be working for even the normal case
# can you try to check if it works for the base case: only considering parents and not the
# children?
