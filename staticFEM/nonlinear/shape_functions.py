import sympy as sp

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================== #
# Axial 
# ============================================== # 

z, L = sp.symbols('z L')
xi = z/L 

# Beam has assumed deflected shape 
# h(x) = a + b z

# Write polynomial coefficients a and b in terms of nodal DOFs 
# h(0) => u_1 = a 
# h(L) => u_2 = a + b L 

A = sp.Matrix([
    [1, 0],   
    [1, 1]    
])

# Create matrices for each shape function's boundary conditions
b1 = sp.Matrix([1, 0])  # N_1: N(0)=1, N(1)=0
b2 = sp.Matrix([0, 1])  # N_2: N(0)=0, N(1)=1

# Solve for coefficients of each shape function
c1 = A.inv() * b1
c2 = A.inv() * b2

# Create the shape functions
xi = z/L
N1 = c1[0] + c1[1]*xi
N2 = c2[0] + c2[1]*xi

N1 = sp.simplify(N1)
N2 = sp.simplify(N2)

print("\nAxial shape functions:")
print(f"N1 = {N1}")
print(f"N2 = {N2}")

z, L, E, A = sp.symbols('z L E A')
u1, u2 = sp.symbols('u1 u2')

# Use the fact that for axial/truss elements: 
# K_{i,j} = ∫ EA(N_i')(N_j') dz

# When looking at the i,j term, we only need shape functions i and j (or their derivatives). 
# This comes from the principle of virtual work 
# when we apply a virtual displacement in direction i, 
# we only need to consider how it interacts with the actual displacement in direction j. 

dN1, dN2 = sp.diff(N1, z), sp.diff(N2, z) 

K11 = sp.integrate(E*A*dN1*dN1, (z, 0, L))
K12 = sp.integrate(E*A*dN1*dN2, (z, 0, L))
K22 = sp.integrate(E*A*dN2*dN2, (z, 0, L))

print("\nStiffness matrix terms:")
print(f"K11 = {K11}")
print(f"K12 = K21 = {K12}")
print(f"K22 = {K22}")

# [ 1 -1]
# [-1  1] * EA/L

# ============================================== # 
# Bending 
# ============================================== # 
z, L = sp.symbols('z L')
xi = z/L 

# Beam has assumed deflected shape 
# h(x) = a + b z + c z^2 + d z^3 
# h'(x) = b + 2 c z + 3 d z^2

# Write polynomial coefficients a, b, c, d in terms of nodal DOFs 
# h(0)  => u_1  =  a 
# h'(0) => u_2  =  b  
# h(L)  => u_3  =  a + b L + c L^2 + d L^3 
# h'(L) => u_4  =  b + 2c L + 3d L^2 

A = sp.Matrix([
    [1, 0, 0, 0],  
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 2, 3]
])

# Deflected shape written in terms of shape functions 
# h(x) = u_1 N_1(x) + N_2(x) u_2 + N_3(x) u_3 + N_4(x) u_4

# Now enforce conditions on N(0), N'(0), N(1), N'(1)
# e.g., for N_1 we have [1 0 0 0] 
b1 = sp.Matrix([1, 0, 0, 0])  # N_1: N(0)=1, N'(0)=0, N(1)=0, N'(1)=0
b2 = sp.Matrix([0, 1, 0, 0])  # N_2: N(0)=0, N'(0)=1, N(1)=0, N'(1)=0
b3 = sp.Matrix([0, 0, 1, 0])  # N_3: N(0)=0, N'(0)=0, N(1)=1, N'(1)=0
b4 = sp.Matrix([0, 0, 0, 1])  # N_4: N(0)=0, N'(0)=0, N(1)=0, N'(1)=1

# Solve for coefficients of each shape function
c1 = A.inv() * b1
c2 = A.inv() * b2
c3 = A.inv() * b3
c4 = A.inv() * b4

# Create the shape functions
xi = z/L
N1 = c1[0] + c1[1]*xi + c1[2]*xi**2 + c1[3]*xi**3
N2 = L*(c2[0] + c2[1]*xi + c2[2]*xi**2 + c2[3]*xi**3)  
N3 = c3[0] + c3[1]*xi + c3[2]*xi**2 + c3[3]*xi**3
N4 = L*(c4[0] + c4[1]*xi + c4[2]*xi**2 + c4[3]*xi**3)  

N1 = sp.simplify(N1)
N2 = sp.simplify(N2)
N3 = sp.simplify(N3)
N4 = sp.simplify(N4)

print("\nBending shape functions:")
print(f"N1 = {N1}")
print(f"N2 = {N2}")
print(f"N3 = {N3}")
print(f"N4 = {N4}")

N1 = 1 - 3*xi**2 + 2*xi**3
N2 = L*(xi - 2*xi**2 + xi**3)
N3 = 3*xi**2 - 2*xi**3
N4 = L*(xi**3 - xi**2)

z, L, E, I = sp.symbols('z L E I')
u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')
xi = z/L

# For beam bending: K_{i,j} = ∫ EI(N_i'')(N_j'') dz
dN1, dN2, dN3, dN4 = sp.diff(N1, z), sp.diff(N2, z) , sp.diff(N3, z) , sp.diff(N4, z) 

d2N1 = sp.diff(N1, z, 2)  
d2N2 = sp.diff(N2, z, 2)  
d2N3 = sp.diff(N3, z, 2)  
d2N4 = sp.diff(N4, z, 2)  

K = sp.zeros(4, 4) 

for i in range(4):
    for j in range(4):
        d2Ni = sp.diff([N1, N2, N3, N4][i], z, 2)
        d2Nj = sp.diff([N1, N2, N3, N4][j], z, 2)
        K[i,j] = sp.integrate(E*I*d2Ni*d2Nj, (z, 0, L))

print("\nStiffness matrix:")
print(sp.simplify(K))

# [12    6L   -12   6L  ]
# [6L    4L²  -6L   2L² ] * EI/L³
# [-12   -6L   12   -6L ]
# [6L    2L²  -6L   4L² ]

# ============================================== # 
# Torsion 
# ============================================== # 
z, L, E, G, Is, Iw, kappa = sp.symbols('z L E G Is Iw kappa')
xi = z/L

N1 = 1 - 3*xi**2 + 2*xi**3
N2 = L*(xi - 2*xi**2 + xi**3)
N3 = 3*xi**2 - 2*xi**3
N4 = L*(xi**3 - xi**2)

dN1 = sp.diff(N1, z)
dN2 = sp.diff(N2, z)
dN3 = sp.diff(N3, z)
dN4 = sp.diff(N4, z)

d2N1 = sp.diff(N1, z, 2)
d2N2 = sp.diff(N2, z, 2)
d2N3 = sp.diff(N3, z, 2)
d2N4 = sp.diff(N4, z, 2)

K = sp.zeros(4, 4)

# For torsion with warping: K_{i,j} = ∫ (GJ*N_i'*N_j' + EIω*N_i''*N_j'') dz
for i in range(4):
    for j in range(4):
        dNi = sp.diff([N1, N2, N3, N4][i], z)
        dNj = sp.diff([N1, N2, N3, N4][j], z)
        d2Ni = sp.diff([N1, N2, N3, N4][i], z, 2)
        d2Nj = sp.diff([N1, N2, N3, N4][j], z, 2)
        
        # Combine St-Venant and warping terms
        integrand = G*Is*dNi*dNj + E*Iw*d2Ni*d2Nj
        K[i,j] = sp.integrate(integrand, (z, 0, L))

print("\nTorsional stiffness matrix (including warping):")
print(sp.simplify(K))


