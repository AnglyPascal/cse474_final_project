import numpy as np
import numpy.random as rn
import simpy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import measurements as mm
import random as rnd


# inits the random states
L = 10
rng = rn.RandomState(int(rnd.random()*100))
parents = rng.rand(L, L)
weights = rng.rand(L, L)
children = rng.rand(L, L)

# the product quality
q = .5

# inits the initial agents
init_agent_num = L//2
agents = []
# generates some random agents
for i in range(init_agent_num):
    pair = (rn.randint(L), rn.randint(L))
    if pair not in agents:
        agents.append(pair)
# stores the people who have already made decisions
blacklist = []      

# to keep track of who purchased the product
purchased = np.zeros((L, L))

# generates the neighbors of a site in the lattice
def neighbors(pair, L):
    x, y = pair
    for i in [1, -1]:
        xx = x + i
        if 0 <= xx < L:
            yield (xx, y)
    for i in [1, -1]:
        yy = y + i
        if 0 <= yy < L:
            yield (x, yy)

# runs the similation
count = 0
while len(agents) > 0:
    # for all agent, let them make the decision remove them from agents, add them to
    # blacklist if they purchased, add their neighbors to agents if not they are in
    # blacklist.

    for agent in agents:
        x, y = agent
        p = parents[x][y]
        c = children[x][y]
        w = weights[x][y]
       
        # decision value
        decision = c * w + p * (1-w)
        
        agents.remove(agent)
        blacklist.append(agent)

        if decision < q:
            count += 1
            purchased[x][y] = True
            # adding the purchasers neighbors to the list
            for i in neighbors(pair, L):
                if i not in agents and i not in blacklist:
                    agents.append(i)

# function to check if the purchased array is percolating
def if_percolating(array):
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

print(if_percolating(purchased))
print(purchased)
