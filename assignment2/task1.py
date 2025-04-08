import numpy as np

# a
def forward(joint_angles):
    """
    compute forward kinematics for x sets of joint angles.

    parameters:
        joint_angles (list of lists): each inner list contains a set of joint angles [θ1, θ2, θ3].

    returns:
        np.array (array): cartesian coordinates of the end-effector for each set of joint angles.
    """

    # define link lengths
    L1 = 100.9
    L2 = 222.1
    L3 = 136.2 

    def dh(a, alpha, d, theta):
        """
        compute the DH transformation matrix
        
        parameters:
            a (float): link length
            alpha (float): link twist
            d (float): link offset
            theta (float): joint angle
            
        returns:
            np.array: DH transformation matrix
        """
        alpha = np.radians(alpha)
        theta = np.radians(theta)

        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    results = []  # store end-effector positions

    for joint_angles in joint_angles:
        # define DH parameters for the current joint angles
        dh_params = [
            [0, 90, L1, joint_angles[0]],
            [L2, 0, 0, joint_angles[1]],
            [L3, 0, 0, joint_angles[2]]
        ]

        # compute transformation matrices and overall transformation
        A1 = dh(*dh_params[0])
        A2 = dh(*dh_params[1])
        A3 = dh(*dh_params[2])

        A12 = np.dot(A1, A2)
        A123 = np.dot(A12, A3)

        # end-effector position
        cart_cord = A123[:3, 3]
        results.append(cart_cord)

    return np.array(results)


# b
def inverse(cart_cord):
    """
    compute inverse kinematics to find joint angles given the end-effector Cartesian coordinates.
    
    parameters:
        cart_cord (list): end-effector Cartesian coordinates [x, y, z].

    returns:
        list: Joint angles [θ1, θ2, θ3]
    """
    
    #link lengths
    L1 = 100.9
    L2 = 222.1
    L3 = 136.2

    x = cart_cord[0]
    y = cart_cord[1]
    z = cart_cord[2]

    r = np.sqrt(x**2 + y**2) #horizontal distance from origin to wrist
    s = z - L1 #vertical distance from origin to wrist

    #theta1
    theta1 = np.degrees(np.arctan2(y, x))

    #theta3
    D = (r**2 + s**2 - L2**2 - L3**2)/(2*L2*L3)
    D = np.clip(D, -1, 1)  # ensure D is in the valid range [-1,1]
    theta3 = np.degrees(np.arctan2(np.sqrt(1-D**2), D))

    #theta2
    theta2 = np.degrees(np.arctan2(s, r) - np.arctan2(L3*np.sin(np.radians(theta3)), L2 + L3*np.cos(np.radians(theta3))))

    return [theta1, theta2, theta3]


# c
def verify():
    """
    verifies that the inverse and forward kinematics functions are correct
    by checking if inverse(forward(joint_angles)) == joint_angles.
    """

    test_joint_angles = [
        [180,60,45]
    ]

    for i, angles in enumerate(test_joint_angles):
        print(f"test {i + 1}: {angles}")

        # Forward kinematics
        cart_cord = forward([angles])[0]
        cart_cord = np.round(cart_cord, 4)
        print(f"Forward kinematics: {cart_cord}")

        # Inverse kinematics
        joint_angles = inverse(cart_cord)
        joint_angles = np.round(joint_angles, 4)
        print(f"Inverse kinematics: {joint_angles}")

        print("")

verify()


# d
def inverse_4_solutions(cart_cord):
    """
    compute the four sets of inverse kinematics solutions given the end effector Cartesian coordinates.

    parameters:
        cart_cord (list): end-effector Cartesian coordinates [x, y, z].

    returns:
        dict: a dictionary containing the four sets of joint angles.
    """

    L1 = 100.9
    L2 = 222.1
    L3 = 136.2

    x = cart_cord[0]
    y = cart_cord[1]
    z = cart_cord[2]

    r = np.sqrt(x**2 + y**2)
    s = z - L1

    # compute two possible values for θ1 (due to symmetry around Y-axis)
    theta1_1 = np.degrees(np.arctan2(y, x)) 
    theta1_2 = np.degrees(np.arctan2(-y, -x))

    # compute θ3 (two solutions: elbow-up and elbow-down)
    D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)
    D = np.clip(D, -1, 1)

    theta3_up = np.degrees(np.arctan2(np.sqrt(1 - D**2), D))
    theta3_down = np.degrees(np.arctan2(-np.sqrt(1 - D**2), D))

    # compute θ2 (for both elbow-up and elbow-down)
    phi = np.arctan2(s, r)

    alpha_up = np.arctan2(L3 * np.sin(np.radians(theta3_up)), (L2 + L3 * np.cos(np.radians(theta3_up))))
    alpha_down = np.arctan2(L3 * np.sin(np.radians(theta3_down)), (L2 + L3 * np.cos(np.radians(theta3_down))))

    theta2_up = np.degrees(phi - alpha_up)
    theta2_down = np.degrees(phi - alpha_down)

    solutions = {
        "elbow_up_1": [float(np.round(theta1_1, 4)), float(np.round(theta2_up, 4)), float(np.round(theta3_up, 4))],
        "elbow_down_1": [float(np.round(theta1_1, 4)), float(np.round(theta2_down, 4)), float(np.round(theta3_down, 4))],
        "elbow_up_2": [float(np.round(theta1_2, 4)), float(np.round(theta2_up, 4)), float(np.round(theta3_up, 4))],
        "elbow_down_2": [float(np.round(theta1_2, 4)), float(np.round(theta2_down, 4)), float(np.round(theta3_down, 4))]
    }

    return solutions 

tcp_position = [0, -323.9033, 176.6988]  # given TCP position
print(inverse(tcp_position))

# compute inverse kinematics solutions
solutions = inverse_4_solutions([0, -323.9033, 176.6988])

# display the four possible sets of joint angles
print("Inverse Kinematics Solutions (Four Sets):")
for key, angles in solutions.items():
    print(f"{key}: {angles}")