from math import gcd, ceil
import itertools
from scipy import sparse
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
import sys

# warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

# Calculate equilibrium matrix B
def calcB(Nd, Cn, dof):
    m, n1, n2 = len(Cn), Cn[:,0].astype(int), Cn[:,1].astype(int)
    l, X, Y = Cn[:,2], Nd[n2,0]-Nd[n1,0], Nd[n2,1]-Nd[n1,1]
    d0x, d0y, d0z, d1x, d1y, d1z = dof[n1*3], dof[n1*3+1], dof[n1*3+2], dof[n2*3], dof[n2*3+1], dof[n2*3+2]
    Y0, X0, l0 = np.multiply(np.divide(Y,l),d0x), np.multiply(np.divide(X,l),d0y), np.multiply(np.divide(1,l),d0z)
    Y1, X1, l1 = np.multiply(np.divide(Y,l),d1x), np.multiply(np.divide(X,l),d1y), np.multiply(np.divide(1,l),d1z)
    v = np.concatenate((-Y0, X0, l0, -l0, Y1, -X1, -l1, l1))
    r = np.concatenate((n1*3, n1*3+1, n1*3+2, n1*3+2, n2*3, n2*3+1, n2*3+2, n2*3+2))
    M0, M1 = np.linspace(0,m*2-2,m), np.linspace(1,m*2-1,m)
    c = np.concatenate((M0, M0, M0, M1, M1, M1, M0, M1))
    return sparse.coo_matrix((v, (r, c)), shape = (len(Nd)*3, m*2))

# Solve linear programming problem
def solveLP(Nd, Cn, f, dof, st, sc, jc):
    l0 = [col[2]/2 + jc for col in Cn]
    l = np.tile(np.repeat(l0, 2), 1)
    B = calcB(Nd, Cn, dof) 
    a = cvx.Variable(len(Cn)*2)
    obj = cvx.Minimize(np.transpose(l) @ a)
    q, eqn, cons= [],  [], [a>=0]
    for k, fk in enumerate(f):
        q.append(cvx.Variable(len(Cn)*2))
        eqn.append(B @ q[k] == fk * dof)
        cons.extend([eqn[k], q[k] >= -sc * a, q[k] <= st * a])
    prob = cvx.Problem(obj, cons)
    vol = prob.solve()
    q = [np.array(qi.value).flatten() for qi in q]
    a = np.array(a.value).flatten()
    # u = [-np.array(eqnk.dual_value).flatten() for eqnk in eqn]
    return vol, a, q

# Visualize moment plot 
def plotMoments(Nd, Cn, q, threshold, str, width, height, f, dof, update = True):
    plt.ion() if update else plt.ioff()
    plt.clf(); plt.axis('off'); plt.axis('equal');  plt.draw()
    plt.title(str)
    tk = 5 / max(abs(q))
    q2 = q.reshape(int(len(q)/2),2)
    for i in [i for i in range(int(len(q2))) if abs(q2[i,0]) + abs(q2[i,1]) >= threshold]:
        con = Nd[Cn[i, [0, 1]].astype(int), :]
        start, end, num_segments = con[0], con[1], 100
        x = np.linspace(start[0], end[0], num_segments)
        y = np.linspace(start[1], end[1], num_segments)
        thickness = np.linspace(q2[i,0]*tk, q2[i,1]*tk, num_segments)
        for i in range(num_segments - 1):
            if thickness[i+1] < 0: c = 'b'
            else: c = 'r'
            plt.plot([x[i], x[i+1]], [y[i], y[i+1]], linewidth=abs(thickness[i+1]), color=c)

    # NdCn = Nd[Cn[:,[0,1]].astype(int)]
    # for line in NdCn:
    #     x_coords, y_coords = zip(*line)
    #     plt.plot(x_coords, y_coords, 'k', linewidth=1.2, alpha=0.05)

    bounding_box = np.array([[0, 0], [0, height], [width, height], [width, 0], [0, 0]])  
    plt.plot(bounding_box[:, 0], bounding_box[:, 1], '--', linewidth=1)
    
    vdof = np.array(dof.reshape(int(len(dof)/3),3))
    support_id = np.array([0 not in vd for vd in vdof])
    supports = Nd[support_id == False]
    plt.scatter(supports[:, 0], supports[:, 1], s=50)

    # vf = np.array(f[0]).reshape(int(len(f[0])/3),3)
    # loaded_id = np.array([all(vv == 0 for vv in vrow) for vrow in vf])   
    # loaded = Nd[loaded_id == False] 
    # plt.scatter(loaded[:, 0], loaded[:, 1], s=100, c='g', marker='s')
    plt.savefig("test.png",bbox_inches='tight')
    plt.pause(0.01) if update else plt.show()

# Main function 
def grillageopt(width, height, st, sc, jc):
    poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    convex = True if poly.convex_hull.area == poly.area else False
    xv, yv = np.meshgrid(range(width+1), range(height+1))
    pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
    Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
    dof, f, PML = np.ones((len(Nd),3)), [], []

    # Load and support conditions
    for i, nd in enumerate(Nd):
        if (nd == [width,height]).all(): dof[i,:] = [1,1,0]
        if (nd == [0,0]).all(): dof[i,:] = [1,1,0]
        if (nd == [width,0]).all(): dof[i,:] = [1,1,0]
        if (nd == [0,height]).all(): dof[i,:] = [1,1,0]
        # if nd[0] == 0: dof[i,:] = [1,1,0]
        # if nd[0] == width: dof[i,:] = [1,1,0]
        # if nd[1] == 0: dof[i,:] = [1,1,0]
        # if nd[1] == height: dof[i,:] = [1,1,0]
        # if nd[0] == height and nd[1] == height: dof[i,:] = [1,1,0]

        f += [0,0,-10] if all(value not in nd for value in [0, height, width]) else [0,0,0]
        # f += [0,0,-10] if (nd == [int(width/2),int(height/2)]).all() else [0,0,0]
        # f += [0,0,-10] if all(nd == 0) else [0,0,0]

    # Create the 'ground structure'
    for i, j in itertools.combinations(range(len(Nd)), 2):
        dx, dy = abs(Nd[i][0] - Nd[j][0]), abs(Nd[i][1] - Nd[j][1])
        if gcd(int(dx), int(dy)) == 1 or jc != 0:
            seg = [] if convex else LineString([Nd[i], Nd[j]])
            if convex or poly.contains(seg) or poly.boundary.contains(seg):
                PML.append( [i, j, np.sqrt(dx**2 + dy**2), False] )
    PML, dof = np.array(PML), np.array(dof).flatten()
    f = [f[i:i+len(Nd)*3] for i in range(0, len(f), len(Nd)*3)]
    # print('Nodes: %d Connections: %d' % (len(Nd), len(PML)))
    for pm in [p for p in PML if p[2] <= 1.42]: 
        pm[3] = True

    # Start optimization 
    Cn = PML
    Cn = PML[PML[:,3] == True]
    vol, a, q = solveLP(Nd, Cn, f, dof, st, sc, jc)
    print("vol: %.1f, Connections: %d" % (vol, len(Cn)))
    plotMoments(Nd, Cn, q[0], max(a) * 1e-3, "", width, height, f, dof, False)

if __name__ =='__main__': 
    grillageopt(width = 5, height = 5, st = 1, sc = 1, jc = 1e-6)