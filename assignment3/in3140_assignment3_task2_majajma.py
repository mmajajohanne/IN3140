from sympy import symbols, Matrix, sin, cos, pprint, diff
from sympy.physics.vector import dynamicsymbols

# Define symbols
q1, q2, q3 = dynamicsymbols("q1 q2 q3")
t = dynamicsymbols._t
L1, L2, L3 = symbols("L1 L2 L3")  # link lengths
g = symbols("g")  # gravity
m1, m2, m3 = symbols("m1 m2 m3")  # masses

# Generalized velocity and acceleration vectors
qdot = Matrix([q1.diff(), q2.diff(), q3.diff()])
qddot = Matrix([q1.diff(t, 2), q2.diff(t, 2), q3.diff(t, 2)])


# ------------------------------------
# a) Compute the potential energy for the manipulator P
# ------------------------------------

# Gravity vector (in base frame)
g_vec = Matrix([0, 0, -g])

# CoM of Link 1 (vertical link)
rc1 = Matrix([0, 0, L1 / 2])

# CoM of Link 2 (from forward kinematics assignment 1)
rc2 = Matrix(
    [L2 / 2 * cos(q1) * cos(q2), L2 / 2 * sin(q1) * cos(q2), L1 + L2 / 2 * sin(q2)]
)

# CoM of Link 3 (from forward kinematics assignment 1)
rc3 = Matrix(
    [
        cos(q1) * (L2 * cos(q2) + L3 / 2 * cos(q2 + q3)),
        sin(q1) * (L2 * cos(q2) + L3 / 2 * cos(q2 + q3)),
        L1 + L2 * sin(q2) + L3 / 2 * sin(q2 + q3),
    ]
)

# Compute potential energy
P = m1 * g_vec.dot(rc1) + m2 * g_vec.dot(rc2) + m3 * g_vec.dot(rc3)
P = P.simplify()

print("--------Potential energy:------")
pprint(P)


# ------------------------------------
# b) Compute the kinetic energy K
# ------------------------------------

# Jacobian matrix for the 3-DOF CrustCrawler
c1 = cos(q1)
s1 = sin(q1)
c2 = cos(q2)
s2 = sin(q2)
c23 = cos(q2 + q3)
s23 = sin(q2 + q3)

J = Matrix(
    [
        [-s1 * (L2 * c2 + L3 * c23), -c1 * (L2 * s2 + L3 * s23), -c1 * (L3 * s23)],
        [c1 * (L2 * c2 + L3 * c23), -s1 * (L2 * s2 + L3 * s23), -s1 * (L3 * s23)],
        [0, (L2 * c2 + L3 * c23), L3 * c23],
        [0, s1, s1],
        [0, -c1, -c1],
        [1, 0, 0],
    ]
)


# Linear velocity part (top 3 rows)
Jv = J[0:3, :]
# Angular velocity part (bottom 3 rows)
Jw = J[3:6, :]

# Link 1 uses only column 0
Jv1 = Jv[:, 0:1]
Jw1 = Jw[:, 0:1]

# Link 2 uses columns 0 and 1
Jv2 = Jv[:, 0:2]
Jw2 = Jw[:, 0:2]

# Link 3 uses columns 0, 1, 2
Jv3 = Jv[:, 0:3]
Jw3 = Jw[:, 0:3]


# Compute inertia matrix D(q)

# Inertia tensors: the assignment assumes rectangular solid links. The standard formula for a box:
# I = (1/12) * m * diag(b^2 + h^2, a^2 + h^2, a^2 + b^2) where a, b, h = width, depth, height of the link.
a1, b1, h1 = symbols("a1 b1 h1")
a2, b2, h2 = symbols("a2 b2 h2")
a3, b3, h3 = symbols("a3 b3 h3")

# Inertia tensors in local link frames (as diagonal matrices)
I1 = Matrix.diag(
    (1 / 12) * m1 * (b1**2 + h1**2),
    (1 / 12) * m1 * (a1**2 + h1**2),
    (1 / 12) * m1 * (a1**2 + b1**2),
)

I2 = Matrix.diag(
    (1 / 12) * m2 * (b2**2 + h2**2),
    (1 / 12) * m2 * (a2**2 + h2**2),
    (1 / 12) * m2 * (a2**2 + b2**2),
)

I3 = Matrix.diag(
    (1 / 12) * m3 * (b3**2 + h3**2),
    (1 / 12) * m3 * (a3**2 + h3**2),
    (1 / 12) * m3 * (a3**2 + b3**2),
)

# Rotation matrices from base frame to each link's center of mass
R1 = Matrix.eye(3)


def Rz(theta):  # Each link rotates about the z-axis
    return Matrix(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
    )


R2 = Rz(q1) * Rz(q2)  # From base to mass center 2
R3 = Rz(q1) * Rz(q2) * Rz(q3)  # From base to mass center 3

# Pad D1 (1x1) to 3x3
D1_block = m1 * (Jv1.T * Jv1) + (Jw1.T * R1 * I1 * R1.T * Jw1)
D1 = Matrix.zeros(3)
D1[0, 0] = D1_block[0, 0]

# Pad D2 (2x2) to 3x3
D2_block = m2 * (Jv2.T * Jv2) + (Jw2.T * R2 * I2 * R2.T * Jw2)
D2 = Matrix.zeros(3)
D2[:2, :2] = D2_block

# D3 is already 3x3
D3 = m3 * (Jv3.T * Jv3) + (Jw3.T * R3 * I3 * R3.T * Jw3)

# Full inertia matrix
D = D1 + D2 + D3


# Final kinetic energy expression

# Kinetic energy
K = (1 / 2) * qdot.T * D * qdot
K.simplify()
K_rounded = K.evalf(4)  # Round all floats to 4 decimals
print("-------- Kinetic energy:--------")
pprint(K_rounded)


# ------------------------------------
# d) Compute gravity vector g(q) and Coriolis term C(q, qdot)*qdot
# ------------------------------------

# Gravity vector from the gradient of potential energy
g1 = diff(P, q1)
g2 = diff(P, q2)
g3 = diff(P, q3)
g_vec = Matrix([g1, g2, g3])

# Initialize Christoffel symbols and C(q, qdot) matrix
C = Matrix.zeros(3, 3)

q_list = [q1, q2, q3]
for i in range(3):
    for j in range(3):
        for k in range(3):
            c_ijk = (1 / 2) * (
                diff(D[k, j], q_list[i])
                + diff(D[k, i], q_list[j])
                - diff(D[i, j], q_list[k])
            )
            C[k, j] += c_ijk * qdot[i]

# Coriolis and centrifugal forces: C(q, qdot) * qdot
Cqdot = C * qdot
Cqdot = Cqdot.evalf(4)

print("--------Gravity vector:--------")
pprint(g_vec)
print("--------Coriolis and centrifugal forces:--------")
pprint(Cqdot)


# ------------------------------------
# e) Final dynamic model: tau = D(q) * qddot + C(q, qdot) * qdot + g(q)
# ------------------------------------

# Compute full torque vector tau
tau = D * qddot + Cqdot + g_vec

# substitute in numerical mass values and g from assignment
mass_values = {m1: 0.3833, m2: 0.2724, m3: 0.1196, g: 9.81}
tau = tau.subs(mass_values)
tau_rounded = tau.evalf(4)

print("--------Final dynamic model (tau):--------")
pprint(tau_rounded)
