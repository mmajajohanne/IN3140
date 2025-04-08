# -------------------
# Imports
# -------------------
from sympy import symbols, Matrix, sin, cos
from sympy.physics.vector import dynamicsymbols

# -------------------
# Define symbols
# -------------------
q1, q2, q3 = dynamicsymbols('q1 q2 q3')
L1, L2, L3 = symbols('L1 L2 L3')  # link lengths
g = symbols('g')                 # gravity
m1, m2, m3 = symbols('m1 m2 m3') # masses

# Gravity vector (in base frame)
g_vec = Matrix([0, 0, -g])

# -------------------
# Define Center of mass positions (CoM)
# -------------------

# CoM of Link 1 (vertical link)
rc1 = Matrix([0, 0, L1/2])

# CoM of Link 2 (from forward kinematics task 2 assignment 2)
rc2 = Matrix([
    L2/2 * cos(q1) * cos(q2),
    L2/2 * sin(q1) * cos(q2),
    L1 + L2/2 * sin(q2)
])

# CoM of Link 3 (from forward kinematics task 2 assignment 2)
rc3 = Matrix([
    cos(q1)*(L2*cos(q2) + L3/2*cos(q2 + q3)),
    sin(q1)*(L2*cos(q2) + L3/2*cos(q2 + q3)),
    L1 + L2*sin(q2) + L3/2*sin(q2 + q3)
])

# -------------------
# Compute potential energy
# -------------------
P = m1 * g_vec.dot(rc1) + m2 * g_vec.dot(rc2) + m3 * g_vec.dot(rc3)
P = P.simplify()

print("Potential energy P:")
print(P)