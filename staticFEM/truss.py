import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

class TrussFEM():
    def __init__(self, **kwargs):
        """
        FEM model of a truss.

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.ndof = 2 # dof per node 
        self.ndofs = self.ndof * self.nnodes  
        self.K = np.zeros((self.ndofs , self.ndofs)) 
        self.u = np.zeros((self.ndofs, 1))
        self.dof_free = np.setdiff1d(np.arange(self.K.shape[0]), self.bc)  
        self.dof_con = self.bc # NOTE

    def k(self, elem_nodes, E, A):
        """
        Get the local stiffness matrix in the global coordinates.

        Parameters
        ----------
        elem_nodes : np.array(
            [[x1, y1], 
            [x2, y2]])

        Returns
        -------
        k : np.array, size=(4, 4)
            stiffness matrix 
            
        """
        l = np.linalg.norm(elem_nodes)
        delta = elem_nodes[1] - elem_nodes[0] # [delx, dely]
        cos_theta = delta[0] / l 
        sin_theta = delta[1] / l 
        R = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]) # rotation matrix 

        k_local = A*E / l*np.array([[1,0,-1,0], [0,0,0,0], [-1,0,1,0], [0,0,0,0]]) # local Truss element stiffnes matrix  
        zero = np.zeros((2,2))
        T = np.bmat([[R, zero],[zero, R]]) # translation matrix 
        ke = np.transpose(T)*k_local*T # element stiffness matrix 
        return ke 

    def solve(self):
        """
        Assemble and solve stiffness matrix.

        """
        for e in range(self.nelems):

            nodes, elems, E, A = self.nodes, self.elems, self.E, self.A
            
            elemConn = elems[e, :] # element connectivity
            x1, y1 = nodes[elemConn[0]]
            x2, y2 = nodes[elemConn[1]]
            elem_nodes = np.array([[x1, y1], [x2, y2]]) 
            ke = self.k(elem_nodes, E[e], A[e])

            # dof (global) <- local dof 
            dof = np.zeros(4)
            dof[0:2] = self.ndof * elemConn[0] + np.array([0, 1])
            dof[2:4] = self.ndof * elemConn[1] + np.array([0, 1])
            dof = dof.astype(int)

            self.K[np.ix_(dof, dof)] += ke # global stiffness matrix 

        # solve reduced systems, for u and f 
        Kf = self.K[np.ix_(self.dof_free, self.dof_free)] # (rows, cols)
        ff = self.f[self.dof_free]
        self.u[self.dof_free] = np.linalg.solve(Kf, ff) # free node displacement
        print(self.u) 

        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ self.u[self.dof_free] # support reactions 

    def show(self, scale=1000, figsize=(6, 6)):
        
        plt.figure(figsize=figsize)

        for n in range(self.nelems):
            node_ind = self.elems[n, :]
            x, y = self.nodes[node_ind].T
            v = self.u[2 * node_ind]   
            z = self.u[2 * node_ind + 1]
            v = np.transpose(v)[0]
            z = np.transpose(z)[0]
            plt.plot(x, y, color='black', linestyle='--', dashes=(10, 4),  linewidth=0.5)
            plt.plot(x + scale*v, y + scale*z, color='b')

        plt.axis('off') 
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

class Truss(TrussFEM):   
    def __init__(self, nodes, elements, E, A):
        """
        Truss object.
        
        # TODO inputs ...

        """
        self.nodes = nodes
        self.elems = elements 
        self.nnodes = np.size(self.nodes , 0)
        self.nelems = np.size(self.elems , 0)
        self.E = E * np.ones(self.nelems)
        self.A = A * np.ones(self.nelems)

    def add_loads(self, loads, nodes):
        # TODO 
        self.f = np.zeros((8, 1))
        self.f[1] = -200e3

    def add_element_properties(self, A=None, E=None, elems=None):
        # TODO 
        pass 

    def add_constraints(self, nodes, dofs):
        # TODO 
        self.bc = np.array([4,5,6,7]) # boundary conditions
         
    def initialise(self):
        super().__init__(nodes=self.nodes, elems=self.elems, E=self.E, A=self.A, f=self.f, bc=self.bc)

if __name__ == "__main__":

    nodes = np.array([
        [20,10], 
        [20,0], 
        [0,10], 
        [0,0]]) 
    
    elements = np.array([
                [2,0], 
                [3,0], 
                [0,1], 
                [2,1], 
                [3,1], 
                [2,3]]) 

    E = 200e9 # Pa 
    A = 0.2 # m2 

    truss = Truss(nodes, elements, E, A)
    truss.add_loads(loads=((0, -200),), nodes=(1,))
    truss.add_constraints(nodes=(2, 3), dofs=((1, 1), (1, 1)))
    # truss.add_element_properties(A=A, elems=(0, 2, 4, 5)) 
    # truss.add_element_properties(E=A, elems=(1, 3)) 

    truss.initialise()
    truss.solve()
    truss.show()

    # TODO complete Truss methods 
    # TODO compare against virtual work 
    # TODO main -> truss.py, Truss -> models, TrussFEM -> FEM  
    # TODO implement plotting features: node num, element num, bar force, node disp, supports  
