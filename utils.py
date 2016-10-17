# -*- encoding: utf-8 -*-

import numpy as np
from math import *
from numpy.linalg import inv
import scipy
import scipy.sparse
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt

#	VERTEX2 id pose.x pose.y pose.theta
#	EDGE2 idFrom idTo mean.x mean.y mean.theta inf.xx inf.xy inf.xt inf.yy inf.yt inf.tt
#        0     1      2    3     4      5           6      7      8       9      10    11
#       no.... this is not right.... 

def readPoseGraph(efile, vfile):
    fp = open(vfile, "r")
    pose = []
    constraints = []
    lines = fp.readlines()
    for line in lines:
        line = line.split(' ')
       #  print line[0]
        if line[0] == 'VERTEX2':
            try:
                pose.append( [ float(line[2]), float(line[3]), float(line[4]) ] )
               # print pose
            except ValueError, e:
                print "error", e, "on line", line
    fp.close()
    
    fp = open(efile, "r")
    lines = fp.readlines()
    for line in lines:
        line = line.split(' ')
        if line[0] == 'EDGE2':
            mean = line[3:6]
            infm = np.zeros((3,3), dtype = np.float64)
            infm[0, 0] = line[6]
            infm[0, 1] = infm[1, 0] = line[7]
            infm[1, 1] = line[8]
            infm[0, 2] = infm[2, 0] = line[10]
            infm[2, 2] = line[9]
            infm[1, 2] = infm[2, 1] = line[11]
            constraints.append( [ int(line[1]), int(line[2]), mean, infm ] )            
    pose = np.array(pose)
    fp.close()
    return pose, constraints

# pose, constraints = readPoseGraph('killian-e.dat', 'killian-v.dat')
# print pose[0, :]
# print constraints[0][3]

def v2t(v):
    c = cos( float(v[2]) )
    s = sin( float(v[2]) )
    A = np.zeros( (3,3), dtype=np.float64 )
    A[0,:] = [c, -s, v[0]]
    A[1,:] = [s, c, v[1]]
    A[2,:] = [0, 0, 1]
    return A

def t2v(t):
    B = np.zeros( (3, 1), dtype=np.float64)
    B[0:2, 0] = t[0:2, 2]
    B[2, 0] = np.arctan2(t[1, 0], t[0, 0])
    return B

class PoseEdge(object):
    def __init__( self, id_from = None, id_to = None, mean = None, infm = None ):
        self.id_from = id_from 
        self.id_to = id_to   # pose being observed
        self.mean = mean.flatten() if type(mean) == np.ndarray else mean
        self.infm = infm # Information matrix of this edge

class PoseGraph(object):

    def __init__(self, nodes=[], edges=[]):
        self.nodes = nodes
        self.edges = []
        self.H = []  # information matrix
        self.b = []  # information vector
        if len(edges) > 0:
            for e in edges:
                self.edges.append(PoseEdge(e))

    def readGraph(self, vfile, efile):
        self.nodes, edges = readPoseGraph(vfile, efile)
        self.edges = []   # reinitialize...
        for e in edges:
            self.edges.append( PoseEdge( int(e[0]), int(e[1]), e[2], e[3] ))
        # print self.edges[3].infm

    def linear(self):
        for e in self.edges:
            i = e.id_from
            j = e.id_to  # being observed

            # extract the poses of the vertices and mean of edge
            v_i = self.nodes[i] 
            v_j = self.nodes[j]

            infm = e.infm

            # transform into matrix (pose)
            T_i = v2t(v_i)
            T_j = v2t(v_j)
            T_z = v2t(e.mean)   # measurement 
            
            R_i = T_i[0:2, 0:2]
            R_z = T_z[0:2, 0:2]

            si = sin(v_i[2])
            ci = cos(v_i[2])

            # calculate error
            F_ij = np.dot( inv(T_i), T_j)
            e = t2v( np.dot( inv(T_z), F_ij) )

            dR_i = np.array( [[-si, -ci], [ci, -si]], dtype = np.float64)
            dt_ij = np.array([v_j[:2] - v_i[:2]], dtype=np.float64).T

            # calculate jacobian   vstack:vertical hstack:horizontal
            A = np.vstack(( np.hstack(( np.dot( (-R_z.T), R_i.T ),  np.dot(np.dot(R_z.T, dR_i.T), dt_ij ))), [0, 0, -1] ))
            B = np.vstack(( np.hstack( ( np.dot(R_z.T, R_i.T), np.zeros((2, 1), dtype=np.float64) ) ), [0,0,1] ))

            # calculate H block (Hx = b)
            H_ii =  np.dot(np.dot(A.T , infm), A)
            H_ij =  np.dot(np.dot(A.T , infm), B)
            H_jj =  np.dot(np.dot(B.T , infm), B)
            b_i  = np.dot(np.dot(-A.T , infm), e)
            b_j  = np.dot(np.dot(-B.T , infm), e)

            # update H and b
            self.H[(3*i):(3*(i+1)),(3*i):(3*(i+1))] += H_ii
            self.H[(3*i):(3*(i+1)),(3*j):(3*(j+1))] += H_ij
            self.H[(3*j):(3*(j+1)),(3*i):(3*(i+1))] += H_ij.T
            self.H[(3*j):(3*(j+1)),(3*j):(3*(j+1))] += H_jj
            self.b[(3*i):(3*(i+1))] += b_i
            self.b[(3*j):(3*(j+1))] += b_j

    def solve(self):
        # solve the Hx=b equation
        #note that the system (H b) is obtained only from
	#relative constraints. H is not full rank.
	#we solve the problem by anchoring the position of
	#the the first vertex.
	#this can be expressed by adding the equation
        self.H[:3, :3] += np.eye(3)

        # H -> sparse representation
        H_sparse = scipy.sparse.csr_matrix(self.H)

        # solve the linear equation Hx=b
        dx = scipy.sparse.linalg.spsolve(H_sparse, self.b)

        # 因为第一个点是不动的。。。
        dx[:3] = [0,0,0]
        # dx[np.isnan(dx)] = 0

        dx = np.reshape( dx, (len(self.nodes), 3) )

        self.nodes += dx

    def optimize(self, iter, plt=None):
        for i in range(iter):
            self.H = np.zeros((len(self.nodes)*3,len(self.nodes)*3), dtype=np.float64)
            self.b = np.zeros((len(self.nodes)*3,1), dtype=np.float64)

            self.linear()

            self.solve()

            if plt is not None:
                plt.clf()
                plt.scatter( self.nodes[:,0], self.nodes[:,1] )
                time.sleep(0.1)
                plt.draw()
