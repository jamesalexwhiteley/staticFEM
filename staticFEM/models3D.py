import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ======================================================================= # 
# 3D Frame model (WIP)
# ======================================================================= # 
class Frame3DFEM:
    def __init__(self, nodes, elems, E, G, A, Iy, Iz, J, a=None):
        """
        3D FEM model of a frame (Euler-Bernoulli).

        Parameters
        ----------
        nodes : np.array(
                    [[x1, y1, z1], 
                     [x2, y2, z2], 
                    ... 
        elements : np.array( 
                    [[0, 1], 
                     [1, 2], 
                     ...

        """
        self.nodes = nodes
        self.elems = elems
        self.nnodes = nodes.shape[0]
        self.nelems = elems.shape[0]
        self.ndof = 6 # 6 dofs per node

        self.E  = E  if hasattr(E,  '__len__') else np.full(self.nelems, E)
        self.G  = G  if hasattr(G,  '__len__') else np.full(self.nelems, G)
        self.A  = A  if hasattr(A,  '__len__') else np.full(self.nelems, A)
        self.Iy = Iy if hasattr(Iy, '__len__') else np.full(self.nelems, Iy)
        self.Iz = Iz if hasattr(Iz, '__len__') else np.full(self.nelems, Iz)
        self.J  = J  if hasattr(J,  '__len__') else np.full(self.nelems, J)

        self.K = np.zeros((self.nnodes*self.ndof, self.nnodes*self.ndof))
        self.f = np.zeros(self.nnodes*self.ndof)
        if a is None:
            self.a = np.zeros(self.nnodes*self.ndof)
        else:
            self.a = a

        self.dof_free = np.arange(self.nnodes*self.ndof, dtype=int) # DOFs free by default 
        self.dof_con  = np.array([], dtype=int)
        self.dof_stiff = [] # finite stiffness supports 

    def k_local_eb(self, E, G, A, Iy, Iz, J, L):
        """
        Local Euler-Bernoulli beam stiffness matrix 12x12 (ignores shear deformation)

        """

        k = np.zeros((12,12))

        # 1) Axial
        k_axial = A*E / L
        k[0,0]   =  k_axial
        k[0,6]   = -k_axial
        k[6,0]   = -k_axial
        k[6,6]   =  k_axial

        # 2) Torsion
        k_torsion = G*J / L
        k[3,3]    =  k_torsion
        k[3,9]    = -k_torsion
        k[9,3]    = -k_torsion
        k[9,9]    =  k_torsion

        # 3) Bending about y 
        EI = E*Iz

        a = 12*EI/(L**3)
        b = 6*EI/(L**2)
        c = 4*EI/L
        d = 2*EI/L

        # w-w coupling
        k[2,2]   += a
        k[2,8]   += -a
        k[8,2]   += -a
        k[8,8]   += a

        # w-ry coupling
        k[2,4]   +=  b
        k[4,2]   +=  b
        k[2,10]  +=  b*-1  
        k[10,2]  +=  b*-1

        k[8,4]   += -b
        k[4,8]   += -b
        k[8,10]  += -b*-1
        k[10,8]  += -b*-1

        # ry-ry coupling
        k[4,4]   += c
        k[4,10]  += d*-1
        k[10,4]  += d*-1
        k[10,10] += c

        # 4) Bending about z 
        EI = E*Iy

        a = 12*EI/(L**3)
        b = 6*EI/(L**2)
        c = 4*EI/L
        d = 2*EI/L

        # v-v
        k[1,1]   += a
        k[1,7]   += -a
        k[7,1]   += -a
        k[7,7]   += a

        # v-rz
        k[1,5]   +=  b
        k[5,1]   +=  b
        k[1,11]  +=  b*-1
        k[11,1]  +=  b*-1

        k[7,5]   += -b
        k[5,7]   += -b
        k[7,11]  += -b*-1
        k[11,7]  += -b*-1

        # rz-rz
        k[5,5]   += c
        k[5,11]  += d*-1
        k[11,5]  += d*-1
        k[11,11] += c

        return k

    def element_rotation_matrix(self, n1, n2):
        """
        Compute a (3x3) rotation matrix R for the element local axes {x_l, y_l, z_l}:
        - x_l along the element
        - y_l, z_l chosen to form a right-handed system 

        """
        x1, y1, z1 = self.nodes[n1]
        x2, y2, z2 = self.nodes[n2]
        dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)

        L = np.sqrt(dx*dx + dy*dy + dz*dz)
        if L < 1e-14:
            return np.eye(3)

        # local x-axis
        x_local = np.array([dx, dy, dz]) / L

        # "reference up" vector
        ref_up = np.array([0, 0, 1], dtype=float)

        # try cross(x_local, ref_up)
        y_l_temp = np.cross(x_local, ref_up)
        nrm = np.linalg.norm(y_l_temp)
        if nrm < 1e-12:
            # fallback
            ref_up = np.array([0, 1, 0], dtype=float)
            y_l_temp = np.cross(x_local, ref_up)
            nrm = np.linalg.norm(y_l_temp)
            if nrm < 1e-12:
                return np.eye(3)

        y_l_temp /= nrm
        # now z_l = x_l x y_l
        z_local = np.cross(x_local, y_l_temp)
        z_local /= np.linalg.norm(z_local)
        # refine y_l to ensure orthogonality
        y_local = np.cross(z_local, x_local)
        y_local /= np.linalg.norm(y_local)

        R = np.vstack([x_local, y_local, z_local])
        return R

    def transform_to_global(self, k_local, R):
        """
        Transform a 12x12 local element stiffness to global:
        K_global = T^T @ k_local @ T, where T = blockdiag(R, R).

        """
        T = np.zeros((12,12))
        # top-left 3x3 -> R, mid-left 3x3 -> 0, bottom-right -> R, etc.
        T[0:3, 0:3]   = R
        T[3:6, 3:6]   = R
        T[6:9, 6:9]   = R
        T[9:12, 9:12] = R
        return T.T @ k_local @ T

    def assemble(self):
        """
        Build/assemble the global stiffness matrix K 

        """
        self.K[:] = 0.0  # reset

        for e in range(self.nelems):
            n1, n2 = self.elems[e]
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
            L = np.sqrt(dx*dx + dy*dy + dz*dz)
            if L < 1e-14:
                continue

            # local k
            k_loc = self.k_local_eb(
                E = self.E[e], G = self.G[e], A = self.A[e],
                Iy= self.Iy[e], Iz= self.Iz[e], J = self.J[e],
                L = L
            )

            # rotation matrix
            R = self.element_rotation_matrix(n1, n2)

            # transform to global
            k_g = self.transform_to_global(k_loc, R)

            # global DOFs
            dof = np.zeros(12, dtype=int)
            dof[0:6]  = self.ndof*n1 + np.arange(6)
            dof[6:12] = self.ndof*n2 + np.arange(6)

            # add to global K
            for i in range(12):
                for j in range(12):
                    self.K[dof[i], dof[j]] += k_g[i,j]

        # add any boundary springs
        for sup in self.dof_stiff:
            dof_id   = sup['dof']
            k_spring = sup['stiffness']
            self.K[dof_id, dof_id] += k_spring

    def solve(self):
        """
        Partition the global system, solve for the free DOFs, and compute reactions.
        """
        self.assemble() 

        Kf = self.K[np.ix_(self.dof_free, self.dof_free)]
        ff = self.f[self.dof_free]

        # solve
        af = np.linalg.solve(Kf, ff)
        self.a[self.dof_free] = af

        # compute reactions
        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ af

    def extract_local_dofs(self, n1, n2, R):
        """
        For element from n1->n2, extract the 12 DOFs in local coordinates
        [u1, v1, w1, rx1, ry1, rz1,  u2, v2, w2, rx2, ry2, rz2]

        """
        dof1 = self.a[self.ndof*n1 : self.ndof*n1 + 6]
        dof2 = self.a[self.ndof*n2 : self.ndof*n2 + 6]

        # translation/rotation
        t1g, r1g = dof1[:3], dof1[3:]
        t2g, r2g = dof2[:3], dof2[3:]

        # local translations, rotations
        t1l = R.T @ t1g
        t2l = R.T @ t2g
        r1l = R.T @ r1g
        r2l = R.T @ r2g

        return np.hstack([t1l, r1l, t2l, r2l]) 

    def shape_euler_bernoulli_3d(self, xi, L, dof_loc):
        """
        Hermite polynomial shape functions.

        """

        (u1, v1, w1, rx1, ry1, rz1,
         u2, v2, w2, rx2, ry2, rz2) = dof_loc

        # linear interpolation for axial 
        Nx1 = 1 - xi
        Nx2 = xi
        u_xl = Nx1*u1 + Nx2*u2

        # bending about z_l => deflection in y_l, rotation about z_l => (v, rz)
        H1 = 1 - 3*xi**2 + 2*xi**3
        H2 = 3*xi**2 - 2*xi**3
        H3 = L*(xi - 2*xi**2 + xi**3)
        H4 = L*(-xi**2 + xi**3)

        v_yl = (H1*v1 + H2*v2 + H3*rz1 + H4*rz2)

        # bending about y_l => deflection in w_l, rotation about y_l => (w, ry)
        w_zl = (H1*w1 + H2*w2 + H3*ry1 + H4*ry2)

        # linear interpolation for torsion 
        rx_xl = Nx1*rx1 + Nx2*rx2

        return (u_xl, v_yl, w_zl, rx_xl)

    def plot_deformed_shape(self, scale=1.0, npoints=20, figsize=(8,8)):
        """
        3D cubic interpolation for each element

        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for e in range(self.nelems):
            n1, n2 = self.elems[e]
            x1, y1, z1 = self.nodes[n1]
            x2, y2, z2 = self.nodes[n2]
            dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
            L = np.sqrt(dx*dx + dy*dy + dz*dz)
            if L < 1e-14:
                continue

            ax.plot([x1,x2], [y1,y2], [z1,z2], 'k--', lw=0.5) # undeformed (dashed)

            # local rotation matrix
            R = self.element_rotation_matrix(n1, n2)
            # local dofs
            dof_loc = self.extract_local_dofs(n1, n2, R)

            # build array of points along the element
            xyz_def = np.zeros((npoints+1, 3))
            for i in range(npoints+1):
                xi = i / npoints
                (u_xl, v_yl, w_zl, rx_xl) = self.shape_euler_bernoulli_3d(xi, L, dof_loc)
                # local displacement vector
                disp_loc = np.array([u_xl, v_yl, w_zl])
                disp_g   = R @ disp_loc
                # base point
                base = np.array([x1, y1, z1]) + xi*np.array([dx, dy, dz])
                xyz_def[i] = base + scale*disp_g

            ax.plot(xyz_def[:,0], xyz_def[:,1], xyz_def[:,2], 'b-', lw=1.5) # deformed curve
            ax.scatter(xyz_def[0,0], xyz_def[0,1], xyz_def[0,2], color='b', s=25) # deformed endpoints
            ax.scatter(xyz_def[-1,0], xyz_def[-1,1], xyz_def[-1,2], color='b', s=25) 

        # self._set_aspect_equal_3d(ax)
        plt.axis('off') 
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout() 
        # plt.savefig("grillage.png", dpi=600) 
        plt.show()


class Frame3D(Frame3DFEM):   
    def __init__(self, nodes, elements, E=200e9, A=0.1):
        """
        Frame object. Input node xyz locations and element connectivity array. 

        Parameters
        ----------
        nodes : np.array(
                    [[x1, y1, z1], 
                     [x2, y2, z2], 
                    ... 
        elements : np.array( 
                    [[0, 1], 
                     [1, 2], 
                     ...

        """
        nelems = elements.shape[0]

        E = np.full(nelems, 210e9)  
        G = np.full(nelems, 80e9)   
        A = np.full(nelems, 0.01)   
        Iy = np.full(nelems, 1e-5)  
        Iz = np.full(nelems, 2e-5)  
        J = np.full(nelems, 1e-5)   

        frame = Frame3DFEM(nodes, elems, E, G, A, Iy, Iz, J)

        # dof_constrained = np.array([0,1,2,3,4,5,
        #                             6,7,8,9,10,11,
        #                             12,13,14,15,16,17,
        #                             18,19,20,21,22,23], dtype=int) 

        dof_constrained = np.array([0,1,2,
                                    6,7,8,
                                    12,13,14,
                                    18,19,20], dtype=int) 

        frame.dof_con = dof_constrained
        frame.dof_free = np.setdiff1d(np.arange(frame.nnodes*frame.ndof), frame.dof_con)
        node2_dof_z = 6*4 + 2 
        frame.f[node2_dof_z] = -3e6

        frame.solve()
        frame.plot_deformed_shape(scale=1.0, npoints=30)


if __name__ == "__main__":

    nodes = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [2.5, 2.5, 0.0]
    ])

    elems = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4]
    ])

    Frame3D(nodes, elems)

