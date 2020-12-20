import numpy as np
import numpy.random as rn
import simpy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import measurements as mm
import random as rnd

L = 3
rng = rn.RandomState(int(rnd.random()*100))
parents = rng.rand(L, L)
weights = rng.rand(L, L)
children = rng.rand(L, L)

init_agen_num = rn.randint(L*L)
# initial_agents_x = []

print(init_agen_num)
