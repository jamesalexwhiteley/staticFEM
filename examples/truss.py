import numpy as np 
from staticFEM.FEM import Truss 

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

E = 200e9 # Pa 
A = 0.2 # m2 

truss = Truss(nodes, elements, E, A)
truss.add_loads(loads=[[0, -500e3],], nodes=[0,])
truss.add_constraints(dofs=[[1, 1], [1, 0]], nodes=[2, 3])
truss.add_element_properties(A=0.4, elems=(0, 2, 4, 5)) 

truss.initialise()  
truss.solve()
truss.show(member_id=True, 
           node_id=False, 
           supports=True, 
           nodal_forces=True, 
           nodal_disp=False)

