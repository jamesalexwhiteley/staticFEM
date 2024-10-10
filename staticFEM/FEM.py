import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

class TrussFEM():
    def __init__(self, **kwargs):
        """
        FEM model of a truss.

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.K = np.zeros((self.ndofs , self.ndofs)) 
        self.u = np.zeros((self.ndofs, 1))
        self.dof_free = np.setdiff1d(np.arange(self.K.shape[0]), self.dof_con)  

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

        Kf = self.K[np.ix_(self.dof_free, self.dof_free)] # np.ix_(rows, cols)
        ff = self.f[self.dof_free]
        self.u[self.dof_free] = np.linalg.solve(Kf, ff) # free node displacement

        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ self.u[self.dof_free] # support reactions 
        # TODO add option for support with finite stiffness 
        # TODO member forces in local coords 

    def normal_vector(self, x, y):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        normal = np.array([-dy, dx]) 
        normal = normal / np.linalg.norm(normal) 
        return normal

    def show(self, scale=2000, figsize=(6, 6), fontsize=10,
             supports=False, 
             nodal_forces=False,
             nodal_disp=False,
             node_id=False,
             member_id=False,
             offset=0.4):
        
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

            normal = self.normal_vector(x, y)
            label_x = x.mean() + normal[0] * offset
            label_y = y.mean() + normal[1] * offset

            if member_id:
                plt.text(label_x, label_y, str(n), color='b', fontsize=fontsize, ha='center', va='center')
                
        if supports:
            for node_ind in self.dof_conx:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)
            for node_ind in self.dof_cony:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)

        if node_id or nodal_forces or nodal_disp:
            for i, node in enumerate(self.nodes):
                nx, ny = node[0], node[1]
                nx += offset
                ny += offset
                if node_id:
                    plt.text(nx, ny, str(i), fontsize=fontsize, ha='center', va='center')
                if nodal_forces:
                    fx = self.f[2*i, :][0]
                    fy = self.f[2*i+1, :][0]
                    plt.text(nx, ny, f"{fx:.0f},{fy:.0f}", fontsize=fontsize, ha='center', va='center')
                if nodal_disp:
                    ux = self.u[2*i, :][0]
                    uy = self.u[2*i+1, :][0]
                    plt.text(nx, ny, f"{ux:.3f},{uy:.3f}", fontsize=fontsize, ha='center', va='center')
            
        plt.axis('off') 
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

class Truss(TrussFEM):   
    def __init__(self, nodes, elements, E=200e9, A=0.1):
        """
        Truss object. Input node xy locations and element connectivity array. 

        Parameters
        ----------
        nodes : np.array(
                    [[x1, y2], 
                     [x2, y2], 
                    ...
        elements : np.array(
                    [[0,1], 
                     [1,2], 
                     ...

        """
        self.nodes = nodes
        self.elems = elements 
        self.nnodes = np.size(self.nodes , 0)
        self.nelems = np.size(self.elems , 0)
        self.ndof = 2 # dof per node 
        self.ndofs = self.ndof * self.nnodes  
        self.dof_con = []
        self.dof_conx = []
        self.dof_cony = []

        self.E = E * np.ones(self.nelems)
        self.A = A * np.ones(self.nelems)
        self.f = np.zeros((self.ndofs, 1))

    def add_loads(self, loads, nodes):
        for load, node in zip(loads, nodes):
            self.f[node*self.ndof : node*self.ndof+2, :] = np.array(load)[:, np.newaxis]

    def add_element_properties(self, A=None, E=None, elems=None):
        for elem in elems:
            if A != None: 
                self.A[elem] = A
            if E != None: 
                self.E[elem] = E

    def add_constraints(self, nodes, dofs):
        """
        Add boundary conditions
        dofs=[[1, 0], ], nodes=[1,] -> Node 1 constrained in x 
        
        """
        for dof, node in zip(dofs, nodes):
            if dof[0] == 1: 
                self.dof_con.append(node*self.ndof)
                self.dof_conx.append(node)
            if dof[1] == 1:
                self.dof_con.append(node*self.ndof + 1)
                self.dof_cony.append(node)
         
    def initialise(self):
        super().__init__(nodes=self.nodes, elems=self.elems, E=self.E, A=self.A, f=self.f)