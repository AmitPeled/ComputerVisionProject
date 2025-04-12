import cv2
from typing import List

import numpy as np
from PIL import Image


def load_image(fpath, sz=256):
    img = Image.open(fpath)
    img = img.resize((sz, sz))
    return np.asarray(img)[:, :, :3]


def spherical_to_cartesian(sph):

    theta, azimuth, radius = sph

    return np.array([
        radius * np.sin(theta) * np.cos(azimuth),
        radius * np.sin(theta) * np.sin(azimuth),
        radius * np.cos(theta),
    ])


def cartesian_to_spherical(xyz):

    xy = xyz[0]**2 + xyz[1]**2
    radius = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    azimuth = np.arctan2(xyz[1], xyz[0])

    return np.array([theta, azimuth, radius])


def elu_to_c2w(eye, lookat, up):

    if isinstance(eye, list):
        eye = np.array(eye)
    if isinstance(lookat, list):
        lookat = np.array(lookat)
    if isinstance(up, list):
        up = np.array(up)

    l = eye - lookat
    if np.linalg.norm(l) < 1e-8:
        l[-1] = 1
    l = l / np.linalg.norm(l)

    s = np.cross(l, up)
    if np.linalg.norm(s) < 1e-8:
        s[0] = 1
    s = s / np.linalg.norm(s)
    uu = np.cross(s, l)

    rot = np.eye(3)
    rot[0, :] = -s
    rot[1, :] = uu
    rot[2, :] = l
    
    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = eye

    return c2w


def c2w_to_elu(c2w):

    w2c = np.linalg.inv(c2w)
    eye = c2w[:3, 3]
    lookat_dir = -w2c[2, :3]
    lookat = eye + lookat_dir
    up = w2c[1, :3]

    return eye, lookat, up


def qvec_to_rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])


def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def recenter_cameras(c2ws):

    is_list = False
    if isinstance(c2ws, list):
        is_list = True
        c2ws = np.stack(c2ws)
  
    center = c2ws[..., :3, -1].mean(axis=0)
    c2ws[..., :3, -1] = c2ws[..., :3, -1] - center

    if is_list:
         c2ws = [ c2w for c2w in c2ws ]

    return c2ws


def rescale_cameras(c2ws, scale):

    is_list = False
    if isinstance(c2ws, list):
        is_list = True
        c2ws = np.stack(c2ws)
  
    c2ws[..., :3, -1] *= scale

    if is_list:
         c2ws = [ c2w for c2w in c2ws ]

    return c2ws


def generate_mock_c2w_values(points_file='tools/clicked_points.pkl'):
    import numpy as np

    def rotate_camera(c2w: np.ndarray, axis: str, direction: str, angle: float = np.radians(10)) -> np.ndarray:
        """
        Rotates the camera-to-world (c2w) pose matrix toward the specified axis.

        Args:
            c2w (np.ndarray): 4x4 camera-to-world transformation matrix.
            axis (str): One of ['xz', 'xy', 'yz'], indicating the plane of rotation.
            direction (str): 'left' or 'right', determining rotation direction.
            angle (float): Rotation angle in radians (default: 10 degrees).

        Returns:
            np.ndarray: Updated 4x4 camera-to-world transformation matrix.
        """
        if axis not in ['xz', 'xy', 'yz']:
            raise ValueError("axis must be one of ['xz', 'xy', 'yz']")
        if direction not in ['left', 'right']:
            raise ValueError("direction must be 'left' or 'right'")

        # Define the rotation axis
        if axis == 'xz':
            rot_axis = np.array([0, 1, 0])  # Rotate around Y-axis
        elif axis == 'xy':
            rot_axis = np.array([0, 0, 1])  # Rotate around Z-axis
        else:  # 'yz'
            rot_axis = np.array([1, 0, 0])  # Rotate around X-axis

        # Adjust angle direction
        if direction == 'right':
            angle = -angle

        # Compute the rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Apply rotation to the camera pose (rotation part only)
        c2w_new = np.copy(c2w)
        c2w_new[:3, :3] = R @ c2w[:3, :3]

        return c2w_new

    import pickle
    import numpy as np

    with open(points_file, 'rb') as f:
        clicked_points = pickle.load(f)

        # Find min and max values for normalization
    all_x = [point[0] for point in clicked_points]
    all_y = [point[1] for point in clicked_points]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Normalize the points to the range [-1, 1]
    def normalize(value, min_value, max_value):
        return 2 * (value - min_value) / (max_value - min_value) - 1

    # List to store the c2w values as ndarrays
    c2w_values = []

    for point in clicked_points:
        point_x, point_y = point

        # Normalize the point_x and point_y values
        normalized_x = normalize(point_x, min_x, max_x) * 3
        normalized_y = normalize(point_y, min_y, max_y) * 3

        # Create a 4x4 transformation matrix for each normalized point
        c2w_matrix = np.array([
            [1, 0, 0, 0],  # Translation on X axis
            [0, 1, 0, normalized_x],  # Translation on Y axis
            [0, 0, 1, normalized_y],  # No translation on Z axis
            [0, 0, 0, 1]  # Homogeneous coordinate
        ])

        # Append the c2w_matrix to the list
        c2w_values.append(rotate_camera(c2w_matrix, 'xz', 'left', np.radians(90)))

    return c2w_values


def save_nerf_files(new_image_paths, new_poses: List[np.ndarray], generated_poses: List[np.ndarray]):
    SAVE_DIR = r'.'
    images = [cv2.imread(path)[:, :, ::-1] for path in new_image_paths]  # Convert BGR to RGB
    np.save(f"{SAVE_DIR}/images.npy", np.array(images, dtype=np.uint8))
    np.save(f"{SAVE_DIR}/poses.npy", np.array(new_poses))
    np.save(f"{SAVE_DIR}/generated_poses.npy", np.array(generated_poses))
