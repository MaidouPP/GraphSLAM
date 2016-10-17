# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import time

# Load data
vfile = 'killian-v.dat'
efile = 'killian-e.dat'

graph = PoseGraph()
graph.readGraph(efile, vfile)

plt.ion()
plt.figure()
plt.scatter(graph.nodes[:, 0], graph.nodes[:,1])
plt.draw()
time.sleep(1)

graph.optimize(5, plt)

plt.show(block=True)
