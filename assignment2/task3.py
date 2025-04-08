import numpy as np

def dh(a, alpha, d, theta):
        alpha = np.radians(alpha)
        theta = np.radians(theta)

        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

# a
def jacobian(joint_angles, joint_velocities):
    """
    compute the velocity of the end-effector using the Jacobian matrix.

    parameters:
        joint_angles (list): joint angles [theta1, theta2, theta3] in degrees.
        joint_velocities (list): joint angular velocities [theta1_dot, theta2_dot, theta3_dot] in rad/s.

    returns:
        cart_velocities (array): the linear velocity of the end-effector
    """

    L1 = 100.9
    L2 = 222.1
    L3 = 136.2

    # forward kinematics
    dh_params = [
        [0,    90, L1, joint_angles[0]],
        [L2,   0, 0, joint_angles[1]],
        [L3,   0, 0, joint_angles[2]]
    ]

    A1 = dh(*dh_params[0])
    A2 = dh(*dh_params[1])
    A3 = dh(*dh_params[2])

    A12 = np.dot(A1, A2)
    A123 = np.dot(A12, A3)

    # find the components of the Jacobian matrix
    z0 = np.array([0, 0, 1])
    z1 = A1[:3, 2]
    z2 = A2[:3, 2]

    o0 = np.array([0, 0, 0])
    o1 = A1[:3, 3]
    o2 = A12[:3, 3]
    o3 = A123[:3, 3]

    # Jv values
    Jv1 = np.cross(z0, o3 - o0)
    Jv2 = np.cross(z1, o3 - o1)
    Jv3 = np.cross(z2, o3 - o2)

    J = [[Jv1, Jv2, Jv3], [z0, z1, z2]]

    # compute the end-effector linear velocity
    cart_velocities = np.dot(J, joint_velocities)
    return cart_velocities

# b
joint_angles = [100,60,45]
joint_velocities = [0.1,0.05,0.05]

cart_velocities = jacobian(joint_angles, joint_velocities)
print(cart_velocities)