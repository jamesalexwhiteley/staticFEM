import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ======================================================================= # 
# 3D Frame model 
# ======================================================================= # 
class Frame3DFEM:
    def __init__(self, nodes, elems, E, G, A, Iy, Iz, J, a=None):
        """
        3D FEM model of a frame (Euler-Bernoulli).

        Parameters
        ----------
        nodes : (nnodes x 3) array
            [x, y, z] coordinates of each node.
        elems : (nelems x 2) array
            Each row = [n1, n2], defining a beam element between nodes n1 and n2.
        E, G : (nelems,) or scalar
            Young's modulus, shear modulus.
        A : (nelems,) or scalar
            Cross-sectional area.
        Iy, Iz : (nelems,) or scalar
            Second moments of area about local y and z.
        J : (nelems,) or scalar
            Torsional constant.
        a : (6*nnodes,) array or None
            Optional global displacement vector. If None, we set it to zeros.
        """
        self.nodes = nodes
        self.elems = elems
        self.nnodes = nodes.shape[0]
        self.nelems = elems.shape[0]
        self.ndof = 6  # 6 dofs per node in 3D

        # Allow scalar or per-element arrays
        self.E  = E  if hasattr(E,  '__len__') else np.full(self.nelems, E)
        self.G  = G  if hasattr(G,  '__len__') else np.full(self.nelems, G)
        self.A  = A  if hasattr(A,  '__len__') else np.full(self.nelems, A)
        self.Iy = Iy if hasattr(Iy, '__len__') else np.full(self.nelems, Iy)
        self.Iz = Iz if hasattr(Iz, '__len__') else np.full(self.nelems, Iz)
        self.J  = J  if hasattr(J,  '__len__') else np.full(self.nelems, J)

        # Global stiffness, force, displacement
        self.K = np.zeros((self.nnodes*self.ndof, self.nnodes*self.ndof))
        self.f = np.zeros(self.nnodes*self.ndof)
        if a is None:
            self.a = np.zeros(self.nnodes*self.ndof)
        else:
            self.a = a

        # By default, treat all DOFs as free until user sets constraints
        self.dof_free = np.arange(self.nnodes*self.ndof, dtype=int)
        self.dof_con  = np.array([], dtype=int)

        # Optional: boundary springs
        self.dof_stiff = []

    # ----------------------------------------------------------------
    # 1) Compute local element stiffness (12x12) in local coords,
    #    then transform to global for assembly
    # ----------------------------------------------------------------
    def k_local_eb(self, E, G, A, Iy, Iz, J, L):
        """
        Local Euler-Bernoulli beam stiffness matrix (12x12),
        ignoring shear deformation, using standard formula:
        - Axial
        - Torsion
        - Bending about y, z
        """
        # For brevity, here's the standard 3D Euler-Bernoulli local 'k' (12x12).
        # We'll build it piecewise:

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

        # 3) Bending about y (deflection in z, rotation about y)
        #    formula akin to a 2D beam for w, ry
        #    The standard '12x12' sub-block is well known:
        #    E*Iz => replaced by E*Iy or E*Iz depending on orientation
        #    Here we do bending about local y => second moment is Iz
        EI = E*Iz

        a = 12*EI/(L**3)
        b = 6*EI/(L**2)
        c = 4*EI/L
        d = 2*EI/L

        # local dofs in the standard order for 3D:
        # dof => [u, v, w, rx, ry, rz,   u2, v2, w2, rx2, ry2, rz2]
        # For bending about y, the relevant deflections are w=2, rz=5
        # But in many references, the index for w is 2, rotation about y is 4, etc.
        # We'll do the usual:
        #   w = dof[2], ry = dof[4]   for node1
        #   w = dof[8], ry = dof[10]  for node2
        # Then the sub-block indices are (2,8) for w, (4,10) for ry. We'll fill that in carefully.

        # w-w coupling
        k[2,2]   += a
        k[2,8]   += -a
        k[8,2]   += -a
        k[8,8]   += a

        # w-ry coupling
        k[2,4]   +=  b
        k[4,2]   +=  b
        k[2,10]  +=  b*-1  # sign depends on sign convention
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

        # 4) Bending about z (deflection in y, rotation about z)
        #    same form but with Iy => replaced by Ix, etc.
        EI = E*Iy

        a = 12*EI/(L**3)
        b = 6*EI/(L**2)
        c = 4*EI/L
        d = 2*EI/L

        # dofs for v => 1, rx => 3? or rotation about z => 5? We must be consistent.
        # Typically for local bending about z, the deflection is v(dof=1),
        # rotation about z is rz(dof=5).
        # Then node2: v=7, rz=11.

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
         - y_l, z_l chosen to form a right-handed triad consistently.

        We do:
          x_l = (node2 - node1)/|...|
          y_l_temp = x_l x global_up (if near zero, fallback to x_l x [0,1,0])
          z_l = x_l x y_l_temp
          y_l = z_l x x_l   (or normalise y_l_temp, then cross in consistent order)
        This ensures a stable right-handed system, avoiding sign flips that break symmetry.
        """
        x1, y1, z1 = self.nodes[n1]
        x2, y2, z2 = self.nodes[n2]
        dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)

        L = np.sqrt(dx*dx + dy*dy + dz*dz)
        if L < 1e-14:
            return np.eye(3)

        # local x-axis
        x_local = np.array([dx, dy, dz]) / L

        # pick a "reference up" vector
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
                # degenerate fallback
                return np.eye(3)

        y_l_temp /= nrm
        # now z_l = x_l x y_l
        z_local = np.cross(x_local, y_l_temp)
        z_local /= np.linalg.norm(z_local)
        # finally refine y_l to ensure exact orthogonality
        y_local = np.cross(z_local, x_local)
        y_local /= np.linalg.norm(y_local)

        # build R
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
        Build/assemble the global stiffness matrix K by looping through elements.
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
        self.assemble()  # build K

        Kf = self.K[np.ix_(self.dof_free, self.dof_free)]
        ff = self.f[self.dof_free]
        # solve
        af = np.linalg.solve(Kf, ff)
        self.a[self.dof_free] = af

        # compute reactions
        Kc = self.K[np.ix_(self.dof_con, self.dof_free)]
        self.f[self.dof_con] = Kc @ af

    # ----------------------------------------------------------------
    # 2) Plotting the Deformed Shape (Correct 3D Euler–Bernoulli)
    # ----------------------------------------------------------------
    def extract_local_dofs(self, n1, n2, R):
        """
        For element from n1->n2, extract the 12 DOFs in local coordinates:
          [u1, v1, w1, rx1, ry1, rz1,  u2, v2, w2, rx2, ry2, rz2]
        """
        dof1 = self.a[self.ndof*n1 : self.ndof*n1 + 6]
        dof2 = self.a[self.ndof*n2 : self.ndof*n2 + 6]
        # break out translation/rotation
        t1g, r1g = dof1[:3], dof1[3:]
        t2g, r2g = dof2[:3], dof2[3:]

        # local translations, rotations
        t1l = R.T @ t1g
        t2l = R.T @ t2g
        r1l = R.T @ r1g
        r2l = R.T @ r2g

        return np.hstack([t1l, r1l, t2l, r2l])  # length=12

    def shape_euler_bernoulli_3d(self, xi, L, dof_loc):
        """
        3D Euler-Bernoulli shape functions (no shear).
        dof_loc = [u1, v1, w1, rx1, ry1, rz1,  u2, v2, w2, rx2, ry2, rz2] in local coords.
        Returns (u_xl, v_yl, w_zl).
        """

        (u1, v1, w1, rx1, ry1, rz1,
         u2, v2, w2, rx2, ry2, rz2) = dof_loc

        # linear interpolation for axial 
        Nx1 = 1 - xi
        Nx2 = xi
        u_xl = Nx1*u1 + Nx2*u2

        # bending about z_l => deflection in y_l, rotation about z_l => (v, rz)
        # Hermite polynomials
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
        Correct 3D Euler-Bernoulli cubic interpolation for each element,
        with local coords -> global coords.
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

    # def _set_aspect_equal_3d(self, ax):
    #     x_limits = ax.get_xlim3d()
    #     y_limits = ax.get_ylim3d()
    #     z_limits = ax.get_zlim3d()

    #     x_range = x_limits[1] - x_limits[0]
    #     y_range = y_limits[1] - y_limits[0]
    #     z_range = z_limits[1] - z_limits[0]
    #     max_range = max(x_range, y_range, z_range) / 2

    #     x_mid = 0.5*(x_limits[0] + x_limits[1])
    #     y_mid = 0.5*(y_limits[0] + y_limits[1])
    #     z_mid = 0.5*(z_limits[0] + z_limits[1])

    #     ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    #     ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    #     ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

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

        E = np.full(nelems, 210e9)  # Young's modulus (Pa)
        G = np.full(nelems, 80e9)   # shear modulus
        A = np.full(nelems, 0.01)   # cross-sectional area
        Iy = np.full(nelems, 1e-5)  # second moment about local y
        Iz = np.full(nelems, 2e-5)  # second moment about local z
        J = np.full(nelems, 1e-5)   # torsional constant

        frame = Frame3DFEM(nodes, elems, E, G, A, Iy, Iz, J)

        dof_constrained = np.array([0,1,2,3,4,5,
                                    6,7,8,9,10,11,
                                    12,13,14,15,16,17,
                                    18,19,20,21,22,23], dtype=int) 

        frame.dof_con = dof_constrained
        frame.dof_free = np.setdiff1d(np.arange(frame.nnodes*frame.ndof), frame.dof_con)
        node2_dof_z = 6*4 + 2 
        frame.f[node2_dof_z] = 5e6

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

