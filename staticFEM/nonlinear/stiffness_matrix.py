import sympy as sp

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================== #
# Stiffness matrix 
# ============================================== # 

z, L = sp.symbols('z L')
xi = z/L

# Node 1: [w, u, v, θ, -v', u', θ'] indices [0, 1, 2, 3, 4, 5, 6]
# Node 2: [w, u, v, θ, -v', u', θ'] indices [7, 8, 9, 10, 11, 12, 13]

# Polynomial coefficients a, b, c and b in terms of nodal DOFs 
A_w = sp.Matrix([
    [1, 0], 
    [1, 1]   
])

# For u and θ
A_cubic = sp.Matrix([
    [1, 0, 0, 0],       # f(0)
    [0, 1, 0, 0],       # f'(0)
    [1, 1, 1, 1],       # f(L)
    [0, 1, 2, 3]        # f'(L)
])

# For v (noting -v')
A_cubic_v = sp.Matrix([
    [1, 0, 0, 0],        # v(0)
    [0, -1, 0, 0],       # -v'(0)  
    [1, 1, 1, 1],        # v(L)
    [0, -1, -2, -3]      # -v'(L)  
])

A = sp.zeros(14, 14)
w_indices = [0, 7]         
u_indices = [1, 5, 8, 12]   
v_indices = [2, 4, 9, 11]   
theta_indices = [3, 6, 10, 13] 

def fill_block(A, start_indices, block):
    n = len(start_indices)
    for i in range(n):
        for j in range(n):
            A[start_indices[i], start_indices[j]] = block[i,j]

fill_block(A, w_indices, A_w)
fill_block(A, u_indices, A_cubic)
fill_block(A, v_indices, A_cubic_v) 
fill_block(A, theta_indices, A_cubic)

# Boundary condition vectors
def create_unit_vector(size, pos):
    vec = sp.zeros(size, 1)
    vec[pos] = 1
    return vec

b_vectors = [create_unit_vector(14, i) for i in range(14)]
c_vectors = [A.inv() * b for b in b_vectors]

# Shape functions
def create_linear_shape_function(coeff):
    return coeff[0] + coeff[1]*xi

def create_cubic_shape_function(coeff, with_L=False):
    base = coeff[0] + coeff[1]*xi + coeff[2]*xi**2 + coeff[3]*xi**3
    return L*base if with_L else base

N = [None] * 14

def extract_coeffs(c_vector, indices):
    return [c_vector[i] for i in indices]

# Using correct indices 
for i in range(14):
    if i in w_indices:
        coeffs = extract_coeffs(c_vectors[i], w_indices)
        N[i] = create_linear_shape_function(coeffs)
    elif i in u_indices:
        coeffs = extract_coeffs(c_vectors[i], u_indices)
        N[i] = create_cubic_shape_function(coeffs, i in [5, 12])
    elif i in v_indices:
        coeffs = extract_coeffs(c_vectors[i], v_indices)
        N[i] = create_cubic_shape_function(coeffs, i in [4, 11])
    elif i in theta_indices:
        coeffs = extract_coeffs(c_vectors[i], theta_indices)
        N[i] = create_cubic_shape_function(coeffs, i in [6, 13])

N = [sp.simplify(n) for n in N]
# for n in N:
#     print(n)

E, G = sp.symbols('E G')
A = sp.Symbol('A')
Ix, Iy = sp.symbols('I_x I_y')
Is = sp.Symbol('J')
Iw = sp.Symbol('I_w')

P0 = sp.Symbol('P0')
Mx0, My0 = sp.symbols('Mx0, My0')
y0, x0 = sp.symbols('y0, x0')
beta_x, beta_y = sp.symbols('beta_x, beta_y')
B0, W = sp.symbols('B0, W')
r = sp.Symbol('r')

# Stiffness matrix
K = sp.zeros(14, 14)

def get_derivatives(N_list, max_order=2):
    derivatives = []
    for N in N_list:
        d_list = [N]
        for order in range(1, max_order+1):
            d_list.append(sp.diff(N, z, order))
        derivatives.append(d_list)
    return derivatives

all_derivs = get_derivatives(N, 2)

# 1. Axial strain energy (w terms) ∫ EA(N_i')(N_j') dz
for i in w_indices:
    for j in w_indices:
        dNi = all_derivs[i][1]
        dNj = all_derivs[j][1]
        integrand = E*A*dNi*dNj
        K[i,j] = sp.integrate(integrand, (z, 0, L))

# 2. Bending strain energy (u terms) ∫ EI(N_i'')(N_j'') dz
for i in u_indices:
    for j in u_indices:
        d2Ni = all_derivs[i][2]
        d2Nj = all_derivs[j][2]
        integrand = E*Iy*d2Ni*d2Nj
        K[i,j] = sp.integrate(integrand, (z, 0, L))

# 3. Bending strain energy (v terms) ∫ EI(N_i'')(N_j'') dz
for i in v_indices:
    for j in v_indices:
        d2Ni = all_derivs[i][2]
        d2Nj = all_derivs[j][2]
        integrand = E*Ix*d2Ni*d2Nj
        K[i,j] = sp.integrate(integrand, (z, 0, L))

# 4. Torsional strain energy (θ terms) ∫ (GJ*N_i'*N_j' + EIω*N_i''*N_j'') dz
for i in theta_indices:
    for j in theta_indices:
        dNi = sp.diff(N[i], z)
        dNj = sp.diff(N[j], z)
        d2Ni = sp.diff(N[i], z, 2)
        d2Nj = sp.diff(N[j], z, 2)
        
        integrand_torsion = G*Is*dNi*dNj
        integrand_warping = E*Iw*d2Ni*d2Nj
        K_torsion = sp.integrate(integrand_torsion, (z, 0, L))
        K_warping = sp.integrate(integrand_warping, (z, 0, L))
        K[i,j] = sp.simplify(K_torsion + K_warping)

# # 1. P0 (w term)
# for i in w_indices:
#     for j in w_indices:
#         dNi = all_derivs[i][1]
#         integrand = P0*dNi
#         K[i,j] += sp.integrate(integrand, (z, 0, L)) # linear force term

# 1. P0 (u,v,θ terms)
for i in range(14):
    for j in range(14):
        if i in u_indices and j in u_indices:
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = P0*dNi*dNj # 1/2 ?
            K[i,j] += sp.integrate(integrand, (z, 0, L))
            
        if i in v_indices and j in v_indices:
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = P0*dNi*dNj # 1/2 ?
            K[i,j] += sp.integrate(integrand, (z, 0, L))
            
        if i in theta_indices and j in theta_indices:
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = P0*r**2*dNi*dNj # 1/2 ?
            K[i,j] += sp.integrate(integrand, (z, 0, L))

# 1. P0 (coupled terms)
for i in range(14):
    for j in range(14):
        if (i in u_indices and j in theta_indices) or (i in theta_indices and j in u_indices):
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = 0.5*P0*y0*dNi*dNj 
            K[i,j] += sp.integrate(integrand, (z, 0, L))
            
        if (i in v_indices and j in theta_indices) or (i in theta_indices and j in v_indices):
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = -0.5*P0*x0*dNi*dNj 
            K[i,j] += sp.integrate(integrand, (z, 0, L))

# 2. Mx0 terms
for i in range(14):
    for j in range(14):
        # if i in v_indices and j in v_indices:
        #     d2Ni = all_derivs[i][2]
        #     integrand = -Mx0*d2Ni
        #     K[i,j] += sp.integrate(integrand, (z, 0, L)) # linear force term
            
        if i in theta_indices and j in theta_indices:
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = 0.5*Mx0*beta_y*dNi*dNj
            K[i,j] += sp.integrate(integrand, (z, 0, L))
            
        if (i in u_indices and j in theta_indices) or (i in theta_indices and j in u_indices):
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = -0.5*Mx0*dNi*dNj
            K[i,j] += sp.integrate(integrand, (z, 0, L))

# 3. My0 terms 
for i in range(14):
    for j in range(14):
        # if i in u_indices and j in u_indices:
        #     d2Ni = all_derivs[i][2]
        #     integrand = My0*d2Ni
        #     K[i,j] += sp.integrate(integrand, (z, 0, L)) # linear force term
            
        if i in theta_indices and j in theta_indices:
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = -0.5*My0*beta_x*dNi*dNj
            K[i,j] += sp.integrate(integrand, (z, 0, L))
            
        if (i in v_indices and j in theta_indices) or (i in theta_indices and j in v_indices):
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = -0.5*My0*dNi*dNj
            K[i,j] += sp.integrate(integrand, (z, 0, L))

# 4. B0 terms (bimoment)
for i in range(14):
    for j in range(14):
        if i in theta_indices and j in theta_indices:
            # d2Ni = all_derivs[i][2]
            # integrand = B0*d2Ni
            # K[i,j] += sp.integrate(integrand, (z, 0, L)) # linear force term
            
            dNi = all_derivs[i][1]
            dNj = all_derivs[j][1]
            integrand = -B0*W*dNi*dNj 
            K[i,j] += sp.integrate(integrand, (z, 0, L))

print(K[11, 13])
# sp.init_printing(use_unicode=True, fontsize='tiny')
# # sp.pprint(K)

def dict_to_string(dok_dict):
    value_groups = {}
    for (i,j), val in dok_dict.items():
        if i <= j: # upper triangular 
            val_str = str(val).replace('*', '')
            if val_str not in value_groups:
                value_groups[val_str] = []
            value_groups[val_str].append((i,j))
    
    items = []
    used_pairs = set()
    
    for val_str, pairs in value_groups.items():
        if len(pairs) > 1:
            combined_keys = [f"K_{{{p[0]+1},{p[1]+1}}}" for p in pairs]
            items.append(f"&{' = '.join(combined_keys)} = {val_str}")
            used_pairs.update(pairs)
        else:
            i,j = pairs[0]
            if (i,j) not in used_pairs:
                items.append(f"&K_{{{i+1},{j+1}}} = {val_str}")
                used_pairs.add((i,j))
    
    return "" + " \\\\\n ".join(items) + ''""

# Write to file
with open('stiffness_matrix.txt', 'w') as f:
    f.write(dict_to_string(K.todok()))