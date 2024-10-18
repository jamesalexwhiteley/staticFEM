import numpy as np 
from staticFEM.FEM import Truss 

# Author: James Whiteley (github.com/jamesalexwhiteley)

nodes = np.array([
    [15, 10], 
    [15, 0], 
    [0, 10], 
    [0, 0]]) 

elements = np.array([
            [2, 0], 
            [3, 0], 
            [0, 1], 
            [2, 1], 
            [3, 1], 
            [2, 3]]) 

# UKB 127x76x13
E = 200e9 # Pa 
A = 1.65e-3 # m2 

truss = Truss(nodes, elements, E, A)
truss.add_loads(loads=[[0, -20e3],], nodes=[0,])
truss.add_constraints(dofs=[[1, 1], [1, 0]], nodes=[2, 3])
# add support with finite stiffness
truss.add_constraints(stiffness=[[10e6, 10e6], ], nodes=[1,]) 

truss.initialise()  
truss.solve()
truss.show(member_id=True, 
           node_id=False, 
           supports=True, 
           nodal_forces=True, 
           nodal_disp=False)

