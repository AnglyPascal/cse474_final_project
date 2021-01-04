import numpy as np
import numpy.random as rn
import simpy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import measurements as mm
import random as rnd

# system class, holds all the main functions and values
class state:
    def __init__(self, L, q, dp, dq, agent_num=None):
        self.L  = L
        self.q  = q
        self.dp = dp
        self.dq = dq

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
    
    def is_percolating(self, array):
        count= 0
        for i in array:
            for j in i:
                count+= j

        if 2*count >= self.L*self.L:
            return True
        else:
            return False

    
# now just run the system for a bounded number of times
def simulate(s, time):
    for i in range(time):
        s.increment_time()
    print(s.q)
    return s.q 

# simulate(state, 10000)
# np.set_printoptions(precision=2)
x = range(20)
y = []
for i in x:
    s = state(100, .59, 0.0001, 0.001, 100)
    y.append(simulate(s, 200))

plt.plot(x, y)
plt.savefig("100x100, 20")
plt.show()

