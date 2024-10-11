import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

# ======================================================================= # 
# Notes 
# ======================================================================= # 
# TODO member forces in local coords/member displacements 
# TODO automated ULD loading for frame 
# TODO truss linear support with finite stiffness 

# ======================================================================= # 
# Truss model 
# ======================================================================= # 
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
        delta = elem_nodes[1] - elem_nodes[0] # [delx, dely]
        L = np.linalg.norm(delta)
        cos_theta = delta[0] / L
        sin_theta = delta[1] / L
        k_local = A*E / L*np.array([[1,0,-1,0], [0,0,0,0], [-1,0,1,0], [0,0,0,0]]) # local Truss element stiffnes matrix  

        R = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]) # rotation matrix (single node)
        T = np.bmat([[R, np.zeros((2,2))],[np.zeros((2,2)), R]]) # transformation matrix (two nodes per member)

        return np.transpose(T)*k_local*T 

    def solve(self):
        """
        Assemble and solve stiffness matrix.

        """
        for e in range(self.nelems):

            nodes, elems, E, A = self.nodes, self.elems, self.E, self.A
            
            elem_conn = elems[e, :] # element connectivity
            x1, y1 = nodes[elem_conn[0]]
            x2, y2 = nodes[elem_conn[1]]
            elem_nodes = np.array([[x1, y1], [x2, y2]]) 
            ke = self.k(elem_nodes, E[e], A[e])

            # dof (global) <- local dof 
            dof = np.zeros(4)
            dof[0:2] = self.ndof * elem_conn[0] + np.array([0, 1])
            dof[2:4] = self.ndof * elem_conn[1] + np.array([0, 1])
            dof = dof.astype(int)

            self.K[np.ix_(dof, dof)] += ke # global stiffness matrix 

        Kf = self.K[np.ix_(self.dof_free, self.dof_free)] # np.ix_(rows, cols)
        ff = self.f[self.dof_free]
        self.u[self.dof_free] = np.linalg.solve(Kf, ff) # free node displacement

        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ self.u[self.dof_free] # support reactions 

    def normal_vector(self, x, y):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        normal = np.array([-dy, dx]) 
        normal = normal / np.linalg.norm(normal) 
        return normal

    def show(self, scale=300, figsize=(6, 6), fontsize=10,
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
            u = self.u[self.ndof * node_ind].flatten()
            v = self.u[self.ndof * node_ind + 1].flatten()
            plt.plot(x, y, color='black', linestyle='--', dashes=(10, 4),  linewidth=0.5)
            plt.plot(x + scale*u, y + scale*v, color='b')
            plt.scatter(x + scale*u, y + scale*v, color='b', s=fontsize)

            if member_id:
                normal = self.normal_vector(x, y)
                label_x = x.mean() + normal[0] * offset
                label_y = y.mean() + normal[1] * offset
                plt.text(label_x, label_y, str(n), color='b', fontsize=fontsize, ha='center', va='center')
                
        if supports:
            for node_ind in self.dof_con_x:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)
            for node_ind in self.dof_con_y:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)

        if node_id or nodal_forces or nodal_disp:
            for i, node in enumerate(self.nodes):
                nx, ny = node[0], node[1]
                nx += offset
                ny += offset
                if node_id:
                    plt.text(nx, ny, str(i), fontsize=fontsize, ha='center', va='center')
                if nodal_forces:
                    fx = self.f[self.ndof*i, :][0]
                    fy = self.f[self.ndof*i+1, :][0]
                    plt.text(nx, ny, f"{fx:.0f},{fy:.0f}", fontsize=fontsize, ha='center', va='center')
                if nodal_disp:
                    ux = self.u[self.ndof*i, :][0]
                    uy = self.u[self.ndof*i+1, :][0]
                    plt.text(nx, ny, f"{ux:.3f},{uy:.3f}", fontsize=fontsize, ha='center', va='center')
            
        plt.axis('off') 
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig("truss.png", dpi=300) 
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
                    [[0, 1], 
                     [1, 2], 
                     ...

        """
        self.nodes = nodes
        self.elems = elements 
        self.nnodes = np.size(self.nodes, 0)
        self.nelems = np.size(self.elems, 0)
        self.ndof = 2 # dof per node 
        self.ndofs = self.ndof * self.nnodes  
        self.dof_con = []
        self.dof_con_x = []
        self.dof_con_y = []
        self.E = E * np.ones(self.nelems)
        self.A = A * np.ones(self.nelems)
        self.f = np.zeros((self.ndofs, 1))

    def initialise(self):
        super().__init__(nodes=self.nodes, elems=self.elems, E=self.E, A=self.A, f=self.f)

    def add_loads(self, loads, nodes):
        """
        Add loads
        loads=[[0, -10],], nodes=[0,] -> Node 0 loaded in y direction

        """
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
        dofs=[[1, 0], ], nodes=[1,] -> Node 1 constrained in x but not y
        
        """
        for dof, node in zip(dofs, nodes):
            if dof[0] == 1: 
                self.dof_con.append(node*self.ndof)
                self.dof_con_x.append(node)
            if dof[1] == 1:
                self.dof_con.append(node*self.ndof + 1)
                self.dof_con_y.append(node)

# ======================================================================= # 
# Beam model 
# ======================================================================= #     
class BeamFEM(Truss):
    def __init__(self, **kwargs):
        """
        FEM model of a beam.

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.K = np.zeros((self.ndofs , self.ndofs)) 
        self.u = np.zeros((self.ndofs, 1))
        self.dof_free = np.setdiff1d(np.arange(self.K.shape[0]), self.dof_con)  

    def k(self, elem_nodes, E, I):
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
        delta = elem_nodes[1] - elem_nodes[0] 
        L = np.linalg.norm(delta)
        k_local = E * I / L**3 * np.array([[  12,     6*L,   -12,     6*L],   
                                           [ 6*L,  4*L**2,  -6*L,  2*L**2],
                                           [ -12,    -6*L,    12,    -6*L],
                                           [ 6*L,  2*L**2,  -6*L,  4*L**2]])
        return k_local 

    def solve(self):
        """
        Assemble and solve stiffness matrix.

        """
        for e in range(self.nelems):

            nodes, elems, E, I = self.nodes, self.elems, self.E, self.I
            
            elem_conn = elems[e, :] 
            x1, y1 = nodes[elem_conn[0]]
            x2, y2 = nodes[elem_conn[1]]
            elem_nodes = np.array([[x1, y1], [x2, y2]]) 
            ke = self.k(elem_nodes, E[e], I[e])

            # dof (global) <- local dof 
            dof = np.zeros(4)
            dof[0:2] = self.ndof * elem_conn[0] + np.array([0, 1])
            dof[2:4] = self.ndof * elem_conn[1] + np.array([0, 1])
            dof = dof.astype(int)

            self.K[np.ix_(dof, dof)] += ke 

        Kf = self.K[np.ix_(self.dof_free, self.dof_free)] 
        ff = self.f[self.dof_free]
        self.u[self.dof_free] = np.linalg.solve(Kf, ff)

        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ self.u[self.dof_free] 

    def show(self, scale=20, figsize=(6, 6), fontsize=10,
             supports=False, 
             nodal_forces=False,
             nodal_disp=False,
             node_id=False,
             member_id=False,
             offset=0.01, 
             interpolation=100):
        
        plt.figure(figsize=figsize)

        for n in range(self.nelems):
            node_ind = self.elems[n, :]
            x, y = self.nodes[node_ind].T
            v = self.u[2 * node_ind].flatten() 
            theta = self.u[2 * node_ind + 1].flatten()
            plt.plot(x, y, color='black', linestyle='--', dashes=(10, 4),  linewidth=0.5)
            plt.scatter(x, y + scale*v, color='b', s=fontsize)
            
            spline = CubicHermiteSpline(x, v, theta) # interpolate deflected shape 
            xs = np.linspace(x[0], x[1], interpolation) 
            ys = spline(xs)
            plt.plot(xs, scale*ys, color='b')

            if member_id:
                normal = self.normal_vector(x, y)
                label_x = x.mean() + normal[0] * offset
                label_y = y.mean() + normal[1] * offset
                plt.text(label_x, label_y, str(n), color='b', fontsize=fontsize, ha='center', va='center')
                
        if supports:
            for node_ind in self.dof_con_x:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)
            for node_ind in self.dof_con_y:
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
        plt.gca().set_aspect(10, adjustable='datalim') # NOTE 
        plt.show()

class Beam(BeamFEM):   
    def __init__(self, nodes, elements, E=200e9, I=0.1):
        """
        Beam object. Input node xy (y=0) locations and element connectivity array. 

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
        self.nnodes = np.size(self.nodes, 0)
        self.nelems = np.size(self.elems, 0)
        self.ndof = 2 # dof per node 
        self.ndofs = self.ndof * self.nnodes  
        self.dof_con = []
        self.dof_con_x = []
        self.dof_con_y = []
        self.E = E * np.ones(self.nelems)
        self.I = I * np.ones(self.nelems)
        self.f = np.zeros((self.ndofs, 1))

    def add_element_properties(self, I=None, E=None, elems=None):
        for elem in elems:
            if I != None: 
                self.I[elem] = I
            if E != None: 
                self.E[elem] = E

    def initialise(self):
        super().__init__(nodes=self.nodes, elems=self.elems, E=self.E, I=self.I, f=self.f)

# ======================================================================= # 
# Beam model 
# ======================================================================= #     
class FrameFEM(Truss):
    def __init__(self, **kwargs):
        """
        FEM model of a beam.

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.K = np.zeros((self.ndofs , self.ndofs)) 
        self.u = np.zeros((self.ndofs, 1))
        self.dof_free = np.setdiff1d(np.arange(self.K.shape[0]), self.dof_con)  

    def k(self, elem_nodes, E, A, I):
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
        delta = elem_nodes[1] - elem_nodes[0]
        L = np.linalg.norm(delta)
        cos_theta = delta[0] / L
        sin_theta = delta[1] / L

        k_local = np.array([
            [ A*E / L, 0, 0, -A*E / L, 0, 0],                
            [ 0, 12*E*I / L**3, 6*E*I / L**2, 0, -12*E*I / L**3, 6*E*I / L**2], 
            [ 0, 6*E*I / L**2, 4*E*I / L, 0, -6*E*I / L**2, 2*E*I / L],         
            [-A*E / L, 0, 0, A*E / L, 0, 0],             
            [ 0, -12*E*I / L**3, -6*E*I / L**2, 0, 12*E*I / L**3, -6*E*I / L**2], 
            [ 0, 6*E*I / L**2, 2*E*I / L, 0, -6*E*I / L**2, 4*E*I / L]          
        ])
        
        R = np.array([
            [cos_theta,  sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [ 0, 0, 1]
        ])
        
        T = np.block([
            [R, np.zeros((3, 3))],
            [np.zeros((3, 3)), R]
        ])
        
        return T.T @ k_local @ T

    def solve(self):
        """
        Assemble and solve stiffness matrix.

        """
        for e in range(self.nelems):

            nodes, elems, E, A, I = self.nodes, self.elems, self.E, self.A, self.I
            
            elem_conn = elems[e, :] 
            x1, y1 = nodes[elem_conn[0]]
            x2, y2 = nodes[elem_conn[1]]
            elem_nodes = np.array([[x1, y1], [x2, y2]]) 
            ke = self.k(elem_nodes, E[e], A[e], I[e])

            # dof (global) <- local dof 
            dof = np.zeros(6)
            dof[0:3] = self.ndof * elem_conn[0] + np.array([0, 1, 2])
            dof[3:6] = self.ndof * elem_conn[1] + np.array([0, 1, 2])
            dof = dof.astype(int)

            self.K[np.ix_(dof, dof)] += ke 

        Kf = self.K[np.ix_(self.dof_free, self.dof_free)] 
        ff = self.f[self.dof_free]
        self.u[self.dof_free] = np.linalg.solve(Kf, ff)

        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ self.u[self.dof_free] 

    def transform_global_to_local(self, elem_conn):
        """
        Transform global displacements (u, v, theta) to local displacements 
        using the element's orientation.
        
        Parameters
        ----------
        node_ind : array
            Indices of the nodes of the element.

        Returns
        -------
        u_local : np.array
            Local displacements (u, v, theta) for the element in local coordinates.
        """

        x1, y1 = self.nodes[elem_conn[0]]
        x2, y2 = self.nodes[elem_conn[1]]
        elem_nodes = np.array([[x1, y1], [x2, y2]]) 

        delta = elem_nodes[1] - elem_nodes[0]
        L = np.linalg.norm(delta)
        cos_theta, sin_theta = delta / L
        
        u_global = self.u[self.ndof * elem_conn].flatten() 
        v_global = self.u[self.ndof * elem_conn + 1].flatten() 
        theta_global = self.u[self.ndof * elem_conn + 2].flatten() 

        u_g = np.array([u_global[0], v_global[0], theta_global[0],
                        u_global[1], v_global[1], theta_global[1]])

        R = np.array([
            [cos_theta,  sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [ 0, 0, 1]
        ])
        
        T = np.block([
            [R, np.zeros((3, 3))],
            [np.zeros((3, 3)), R]
        ])

        return T @ u_g, R
        
    def show(self, scale=10, figsize=(6, 6), fontsize=10,
             supports=False, 
             nodal_forces=False,
             nodal_disp=False,
             node_id=False,
             member_id=False,
             offset=0.01, 
             interpolation=100): 
        
        plt.figure(figsize=figsize)

        for elem in self.elems:
            
            x1, y1 = self.nodes[elem[0]]
            x2, y2 = self.nodes[elem[1]]
            delta = np.array([x2 - x1, y2 - y1])
            L = np.linalg.norm(delta)

            # local displacements (u1, v1, θ1, u2, v2, θ2)
            u1, v1, theta1 = self.u[self.ndof * elem[0]:self.ndof * elem[0] + 3]
            u2, v2, theta2 = self.u[self.ndof * elem[1]:self.ndof * elem[1] + 3]

            xy = np.zeros((interpolation + 1, 2))   # interpolated points 
            for i in range(interpolation + 1):
                xi = i / interpolation  
                x_local = u1 * (1 - xi) + u2 * xi   # linear interpolation in x 
                y_local = (                         # cubic interpolation in y 
                    v1 * (1 - 3 * xi**2 + 2 * xi**3) +  
                    v2 * (3 * xi**2 - 2 * xi**3) +      
                    theta1 * L * (xi - 2 * xi**2 + xi**3) + 
                    theta2 * L * (-xi**2 + xi**3)            
                )

                # local displacements vector
                xy_local = np.array([x_local, y_local])
                xy[i, 0] = x1 + xi * delta[0] + scale * xy_local[0] 
                xy[i, 1] = y1 + xi * delta[1] + scale * xy_local[1] 

            plt.plot(xy[:, 0], xy[:, 1], 'b-')
            plt.plot([x1, x2], [y1, y2], '--', color='black', linewidth=0.5,  dashes=(10, 4))
            plt.scatter([x1 + scale * u1, x2 + scale * u2], 
                        [y1 + scale * v1, y2 + scale * v2], color='b', s=fontsize)           

            if member_id:   
                normal = self.normal_vector(x, y)
                label_x = x.mean() + normal[0] * offset
                label_y = y.mean() + normal[1] * offset
                plt.text(label_x, label_y, str(n), color='b', fontsize=fontsize, ha='center', va='center')
                
        if supports:
            for node_ind in self.dof_con_x:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)
            for node_ind in self.dof_con_y:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='^', color='b', s=fontsize*10)
            for node_ind in self.dof_con_theta:
                plt.scatter(self.nodes[node_ind][0], self.nodes[node_ind][1], marker='s', color='b', s=fontsize*10)

        if node_id or nodal_forces or nodal_disp:
            for i, node in enumerate(self.nodes):
                nx, ny = node[0], node[1]
                nx += offset
                ny += offset
                if node_id:
                    plt.text(nx, ny, str(i), fontsize=fontsize, ha='center', va='center')
                if nodal_forces:
                    fx = self.f[self.ndof*i, :][0]
                    fy = self.f[self.ndof*i+1, :][0]
                    m = self.f[self.ndof*i+2, :][0]
                    plt.text(nx, ny, f"{fx:.0f},{fy:.0f},{m:.0f}", fontsize=fontsize, ha='center', va='center')
                if nodal_disp:
                    ux = self.u[self.ndof*i, :][0]
                    uy = self.u[self.ndof*i+1, :][0]
                    theta = self.u[self.ndof*i+2, :][0]
                    plt.text(nx, ny, f"{ux:.1f},{uy:.1f},{theta:.1f}", fontsize=fontsize, ha='center', va='center')
            
        plt.axis('off') 
        plt.gca().set_aspect('equal', adjustable='box') 
        # plt.savefig("frame.png", dpi=300) 
        plt.show()

class Frame(FrameFEM):   
    def __init__(self, nodes, elements, E=1, A=1, I=1):
        """
        Frame object. Input node xy (y=0) locations and element connectivity array. 

        Parameters
        ----------
        nodes : np.array(
                    [[x1, y2], 
                     [x2, y2], 
                    ...
        elements : np.array(
                    [[0, 1], 
                     [1, 2], 
                     ... 

        """
        self.nodes = nodes
        self.elems = elements 
        self.nnodes = np.size(self.nodes, 0)
        self.nelems = np.size(self.elems, 0)
        self.ndof = 3 # dof per node 
        self.ndofs = self.ndof * self.nnodes  
        self.dof_con = []
        self.dof_con_x = []
        self.dof_con_y = []
        self.dof_con_theta = []
        self.E = E * np.ones(self.nelems)
        self.A = A * np.ones(self.nelems)
        self.I = I * np.ones(self.nelems)
        self.f = np.zeros((self.ndofs, 1)) 

    def initialise(self):
        super().__init__(nodes=self.nodes, elems=self.elems, E=self.E, I=self.I, f=self.f)

    def add_element_properties(self, I=None, E=None, A=None, elems=None):
        for elem in elems:
            if I != None: 
                self.I[elem] = I
            if E != None: 
                self.E[elem] = E
            if A != None: 
                self.A[elem] = A

    def add_loads(self, loads, nodes):
        """
        Add loads
        loads=[[0, -10, 4],], nodes=[0,] -> Node 0 loaded in y direction, and moment applied (counterclockwise positive)

        """
        for load, node in zip(loads, nodes):
            self.f[node*self.ndof : node*self.ndof+3, :] = np.array(load)[:, np.newaxis]

    def add_constraints(self, nodes, dofs):
        """
        Add boundary conditions
        dofs=[[1, 1, 0], ], nodes=[1,] -> Node 1 constrained in x and y, but free rotationally. 
        
        """
        for dof, node in zip(dofs, nodes):
            if dof[0] == 1: 
                self.dof_con.append(node*self.ndof)
                self.dof_con_x.append(node)
            if dof[1] == 1:
                self.dof_con.append(node*self.ndof + 1)
                self.dof_con_y.append(node)
            if dof[2] == 1:
                self.dof_con.append(node*self.ndof + 2)
                self.dof_con_theta.append(node)

