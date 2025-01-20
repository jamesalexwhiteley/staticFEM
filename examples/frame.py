import numpy as np 
from staticFEM.models import Frame  

# Author: James Whiteley (github.com/jamesalexwhiteley)


nodes = np.array([
    [0, 0], 
    [0, 2], 
    [1.5, 2],
    [3, 2],
    [3, 0]]) 

elements = np.array([
            [0, 1], 
            [1, 2],
            [2, 3],
            [3, 4]]) 

# UKB 127x76x13
E = 200e9 # Pa 
A = 1.65e-3 # m2 
I = 4.73e-6 # m4

frame = Frame(nodes, elements, E, A, I)
# add loads and constraints 
frame.add_loads(loads=[[20e3, -20e3, 0], ], nodes=[2,]) 
frame.add_constraints(dofs=[[1, 1, 1], [1, 1, 0]], nodes=[0, 4]) 
# add support with finite stiffness
# frame.add_loads(loads=[[100e3, -100e3, -100e3], ], nodes=[4,]) 
# frame.add_constraints(stiffness=[[10e6, 10e6, 10e6], ], nodes=[4,]) 

frame.initialise()   
frame.solve()
frame.show(member_id=False, 
           node_id=False, 
           supports=True, 
           nodal_forces=True, 
           nodal_disp=False)