import numpy as np 
from staticFEM.FEM import Beam  

nodes = np.array([
    [0, 0], 
    [1, 0], 
    [2, 0],
    [4, 0]]) 

elements = np.array([
            [0, 1], 
            [1, 2],
            [2, 3]]) 

# UKB 127x76x13
E = 200e9 # Pa 
I = 4.73e-6 # m4

beam = Beam(nodes, elements, E, I)
beam.add_loads(loads=[[-10e3, 0], [-1e3, 0]], nodes=[1, 3])
beam.add_constraints(dofs=[[1, 0], [1, 0]], nodes=[0, 2])
beam.initialise()
beam.solve()
beam.show(member_id=True, 
           node_id=False, 
           supports=True, 
           nodal_forces=True, 
           nodal_disp=False)