import numpy as np
import numpy.random as rn
import simpy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import measurements as mm
import random as rnd

# system class, holds all the main functions and values
class state:
    def __init__(self, L, q, dp, dq, ratio, agent_num=None):
        self.L  = L
        self.q  = q
        self.dp = dp
        self.dq = dq
        self.ratio = ratio

        self.rng      = rn.RandomState(int(rnd.random()*100))
        self.parents  = self.rng.rand(self.L, self.L)
        self.weights  = self.rng.rand(self.L, self.L)
        self.children = self.rng.rand(self.L, self.L)

        if agent_num == None:
            agent_num = L

        self.agent_num = agent_num          # default agent number
        self.agents    = []                 # these are the agents

        # this will mark every agents, both past and present
        self.blacklist = np.zeros((self.L, self.L))
        
        # these will tell us whether to increase or decrease product quality
        self.parcolating = None
        self.purchased = None

    # a function to generate random agents 
    def gen_agents(self):
        self.agents = []
        for i in range(self.agent_num):
            x, y = (rn.randint(self.L), rn.randint(self.L))
            if self.blacklist[x][y] == 0:
                self.agents.append((x, y))
                self.blacklist[x][y] = 1

    # returns a generator     
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


    # return the decision for a agent in the lattice
    def decision(self, agent):
        x, y = agent

        p    = self.parents[x][y]
        c    = self.children[x][y]
        w    = self.weights[x][y]

        d = c * w + p * (1-w)
        return d


    # this is the core part of the state that makes t -> t+1
    # the tasks it does are:
    # 1. Generate new random agents, make local purchased list
    # 2. For each agent, let them make decision
    # 3. If they bought the stuff, mark them in purchased
    # 4. Add all the neighbors to this agent to the agents list, mark in blacklist
    # 5. Check if system is percolating, save it in self.percolating var
    # 6. Save the purchased list in self.purchased for debugging
    #
    def increment_time(self):
        purchased = np.zeros((self.L, self.L))
        self.gen_agents()

        while len(self.agents) > 0:
            for agent in self.agents:
                self.agents.remove(agent)
                x, y = agent
                d = self.decision(agent)

                if d < self.q:
                    purchased[x][y] = 1
                    for neighbor in self.neighbors(agent, self.L):
                        xx, yy = neighbor
                        if self.blacklist[xx][yy] == 0:
                            self.agents.append(neighbor)
                            self.blacklist[xx][yy] = 1

        # print(purchased)
        self.increment_values(purchased)

        # resetting all the temporary stuffs
        self.blacklist = np.zeros((self.L, self.L))
        self.parcolating = None

        self.purchased = purchased


    # this function takes care of the dynamic stuffs
    # updates the values of q and parents, children accourding
    # whether percolating or purchased
    #
    def increment_values(self, purchased):
        if self.spanning(purchased):
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
    
    def density(self, array):
        count= 0
        for i in array:
            for j in i:
                count+= j

        if count >= self.ratio*self.L*self.L:
            return True
        else:
            return False

    def spanning(self, array):
        lw, num = mm.label(array)
        a1, a2 = lw[0,:], lw[-1,:]

        perc_x = np.intersect1d(a1, a2)
        perc = perc_x[np.where(perc_x>0)]
        if len(perc)>0:
            return True
        return False


    # runs the simulation and returns the final q value
    def __call__(self, time, increment):
        for i in range(time):
            self.increment_time()
            if i % increment == 0:
                yield self.q 
            print(self.q)


L   = 100
q   = .5
dq  = 0.001
dp  = 0.000001
r   = .5
num = 10

x = np.linspace(1, 220, 50)
y = []

for i in range(num):
    s = state(L, q, dp, dq, r)

    for i in s(500, 10):
        y.append(i)
        
    plt.plot(x, y)
    y = []

# plt.savefig('q_converges_spanning_L50_q575_r5_num10_dq1e-4.png')
plt.show()
