import numpy as np 
from staticFEM.models import Frame  

# Author: James Whiteley (github.com/jamesalexwhiteley)

nodes = np.array([
    [0, 0], 
    [2, 0], 
    [4, 0],
    [6, 0]]) 

elements = np.array([
            [0, 1], 
            [1, 2],
            [2, 3]]) 

# UKB 127x76x13
E = 200e9 # Pa 
A = 1.65e-3 # m2 
I = 4.73e-6 # m4

frame = Frame(nodes, elements, E, A, I)
# add loads and constraints 
frame.add_loads(loads=[[0, -20e3, 0], [0, -20e3, 0]], nodes=[1, 3]) 
frame.add_constraints(dofs=[[1, 1, 0], [1, 1, 0]], nodes=[0, 2]) 
# add support with finite stiffness
frame.add_constraints(stiffness=[[0, 0, 10e6], ], nodes=[3,]) 

frame.initialise()   
frame.solve()
frame.show(figsize=(8, 4),
           member_id=False, 
           node_id=False, 
           supports=True, 
           nodal_forces=True, 
           nodal_disp=False)