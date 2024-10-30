import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''


#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    if alpha < 0 or beta < 0 or gamma < 0 or alpha >= 90 or beta >= 90 or gamma >= 90:
        raise ValueError("Invaid alapha, beta or gamma value")
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    rot_xyz2XYZ = np.eye(3).astype(float)
    step1z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
    step2x = np.array([[1, 0, 0],
                      [0, np.cos(beta), -np.sin(beta)],
                      [0, np.sin(beta), np.cos(beta)]])
    step3z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])
    rot_xyz2XYZ  = step1z @ step2x @ step3z
    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)
    alpha = np.radians(-alpha)
    beta = np.radians(-beta)
    gamma = np.radians(-gamma)
    step1z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
    step2x = np.array([[1, 0, 0],
                      [0, np.cos(beta), -np.sin(beta)],
                      [0, np.sin(beta), np.cos(beta)]])
    step3z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])
    
    rot_XYZ2xyz = rot_XYZ2xyz @ step3z @ step2x @ step1z
    # rot_XYZ2xyz = np.round(rot_XYZ2xyz, decimals=3)
    # rot_XYZ2xyz[rot_XYZ2xyz == -0] = 0
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''

    img_coord = np.zeros([32, 2], dtype=float)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, (4, 9), None)
    term_criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    if ret:
        criteria = (term_criteria, 30, 0.001)
        corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
        corners_left = [corners[i] for i in range(16)]
        corners_right = [corners[i] for i in range(20, len(corners))]
        selected_corners = corners_left + corners_right
        img_coord = np.vstack(selected_corners)
        cv2.drawChessboardCorners(image, (4, 9), corners, ret)
    else:
        raise ValueError("Checkerboard corners not found")
        
    return img_coord



def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)
    for index in range(32):
        row = index // 8
        col = index % 8
        world_coord[index] = [col * 10.0, row * 10.0, 0.0]

    return world_coord



def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''
    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0
    img_coord = np.transpose(img_coord.reshape([2, 32]))
    world_coord = np.transpose(world_coord.reshape([3, 32]))
    x = []
    y = []
    for i in range(32):
        x.append(img_coord[i][0])
        y.append(img_coord[i][1])
    x = np.array(x)
    y = np.array(y)
    X_hom = np.hstack((world_coord, np.ones((world_coord.shape[0], 1))))# homo_coord
    ini = np.zeros((32, 4))
    #yi⋅[Xi, Yi, Zi, 1] = 0 
    y_func = np.hstack([ini, -X_hom, y[:, np.newaxis] * X_hom])
    #xi⋅[Xi, Yi, Zi, 1] = 0 
    x_func = np.hstack([X_hom, ini, -x[:, np.newaxis] * X_hom])
    m = np.vstack([y_func, x_func])
    vt = np.linalg.svd(m)[2]
    projection_m = vt[-1].reshape((3, 4))# projection matrix
    M = projection_m[:3, :3]
    R = np.linalg.qr(M)[0]
    K = R/float(R[2,2])
    K[:, 0] = np.where(K[0, 0] < 0, -K[:, 0], K[:, 0])
    K[:, 1] = np.where(K[1, 1] < 0, -K[:, 1], K[:, 1])
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)
    
    img_coord = np.transpose(img_coord.reshape([2, 32]))
    world_coord = np.transpose(world_coord.reshape([3, 32]))
    x = []
    y = []
    for i in range(32):
        x.append(img_coord[i][0])
        y.append(img_coord[i][1])
    x = np.array(x)
    y = np.array(y)
    X_hom = np.hstack((world_coord, np.ones((world_coord.shape[0], 1))))# homo_coord
    ini = np.zeros((32, 4))
    #yi⋅[Xi, Yi, Zi, 1] = 0 
    y_func = np.hstack([ini, -X_hom, y[:, np.newaxis] * X_hom])
    #xi⋅[Xi, Yi, Zi, 1] = 0 
    x_func = np.hstack([X_hom, ini, -x[:, np.newaxis] * X_hom])
    m = np.vstack([y_func, x_func])
    vt = np.linalg.svd(m)[2]
    projection_m = vt[-1].reshape((3, 4))# projection matrix
    M = projection_m[:3, :3]
    R = np.linalg.qr(M)[0]
    Q = np.linalg.qr(M)[1]
    K = R/float(R[2,2])
    K[:, 0] = np.where(K[0, 0] < 0, -K[:, 0], K[:, 0])
    Q[0, :] = np.where(Q[0, 0] < 0, -Q[0, :], Q[0, :])
    K[:, 1] = np.where(K[1, 1] < 0, -K[:, 1], K[:, 1])
    Q[1, :] = np.where(Q[1, 1] < 0, -Q[1, :], Q[1, :])
    P_mat = np.dot(K,Q)
    P_scaled = (P_mat[0,0]*projection_m)/float(projection_m[0,0])
    T = np.dot(np.linalg.inv(K), P_scaled[:,3])
    # Your implementation

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

#Your functions for task2







#---------------------------------------------------------------------------------------------------------------------