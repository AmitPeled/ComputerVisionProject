from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def generate_circle_poses_with_slerp(center_point, c1, c2, num_poses=5):
    """
    Generate camera poses along a circular path between two given poses,
    using SLERP for smooth rotation interpolation.

    Args:
        center_point (numpy.ndarray): Center of the circular path (tx, ty, tz).
        c1 (numpy.ndarray): First Camera-to-World (C2W) matrix.
        c2 (numpy.ndarray): Second Camera-to-World (C2W) matrix.
        num_poses (int): Number of poses to generate (default is 5).

    Returns:
        list: A list of new Camera-to-World matrices.
    """
    # Extract rotation matrices and translations from c1 and c2
    # Extract translation (tx, ty, tz)
    t1 = c1[:3, 3]
    t2 = c2[:3, 3]

    # Compute radius (distance from c1 to center)
    radius = np.linalg.norm(t1 - center_point)

    # Extract rotations as quaternions
    r1 = Rotation.from_matrix(c1[:3, :3])
    r2 = Rotation.from_matrix(c2[:3, :3])

    # Interpolation parameter (evenly spaced)
    times = np.linspace(0, 1, num_poses)

    # Spherical interpolation for rotation
    slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
    interpolated_rotations = slerp(times).as_matrix()

    # Generate positions along circular arc
    interpolated_poses = []
    for i, t in enumerate(times):
        # Linear interpolation for position
        interp_pos = (1 - t) * t1 + t * t2

        # Adjust to maintain fixed radius from center
        direction = interp_pos - center_point
        direction /= np.linalg.norm(direction)  # Normalize
        interp_pos = center_point + direction * radius

        # Create new transformation matrix
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interpolated_rotations[i]  # Set rotation
        interp_pose[:3, 3] = interp_pos  # Set position

        interpolated_poses.append(interp_pose)

    return interpolated_poses


def _get_radius_and_direction(positions, center_position):
    # Calculate distances from center to first and last points
    first_point_center_distance = np.linalg.norm(positions[0] - center_position)
    last_point_center_distance = np.linalg.norm(positions[-1] - center_position)

    if first_point_center_distance <= last_point_center_distance:
        center_distance = first_point_center_distance
        poses_iterable = range(len(positions) - 1, -1, -1)
        is_first = True
    else:
        center_distance = last_point_center_distance
        poses_iterable = range(len(positions))
        is_first = False

    return center_distance, poses_iterable, is_first


# trim off the excluded
# insert the generated in the trimmed area
def apply_smoothing_algorithm(poses: List) -> Tuple:
    poses_len = len(poses)

    # Extract positions (tx, ty, tz) from poses
    positions = [c2w[:3, 3] for c2w in poses]  # Shape: (N, 3)

    # Calculate center of all camera positions
    center_position = np.mean(positions, axis=0)

    center_distance, poses_iterable, is_first = _get_radius_and_direction(positions, center_position)

    excluded_poses_indexes = []
    # modified_poses = {}

    for i in poses_iterable:
        if np.linalg.norm(positions[i] - center_position) <= center_distance:
            break
        excluded_poses_indexes.append(i)

    excluded_len = len(excluded_poses_indexes)
    if is_first:
        start_pose_idx = 0
        target_pose_idx = min(excluded_poses_indexes)
        generated_poses = generate_circle_poses_with_slerp(center_position, poses[start_pose_idx],
                                                           poses[target_pose_idx],
                                                           excluded_len + 3)
        new_poses = poses[start_pose_idx:target_pose_idx] + generated_poses[::-1]
    else:
        start_pose_idx = poses_len - 1
        target_pose_idx = max(excluded_poses_indexes)
        generated_poses = generate_circle_poses_with_slerp(center_position, poses[start_pose_idx],
                                                           poses[target_pose_idx],
                                                           excluded_len + 3)
        new_poses = poses[target_pose_idx:start_pose_idx] + generated_poses[::-1]

    # for i in remain_poses_indexes:
    #     # Generate new pose
    #     target_pose = poses[i].copy()
    #     direction = positions[i] - center_position
    #     unit_direction = direction / np.linalg.norm(direction)
    #     new_position = center_position + unit_direction * center_distance
    #     target_pose[:3, 3] = new_position
    #     # generated_poses.append(new_pose)
    #
    #     modified_poses[i] = target_pose
    excluded_poses = [poses[i] for i in excluded_poses_indexes]

    return new_poses, center_position, generated_poses, excluded_poses
