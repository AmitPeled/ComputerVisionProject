import numpy as np
from typing import Tuple, List


def apply_smoothing_algorithm(poses: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    # Extract positions (tx, ty, tz) from poses
    positions = poses[:, 4:]  # Shape: (N, 3)

    # Calculate center of all camera positions
    center_position = np.mean(positions, axis=0)

    # Calculate distances from center to first and last points
    first_point_center_distance = np.linalg.norm(positions[0] - center_position)
    last_point_center_distance = np.linalg.norm(positions[-1] - center_position)

    if first_point_center_distance <= last_point_center_distance:
        center_distance = first_point_center_distance
        poses_iterable = range(len(poses) - 1, -1, -1)
    else:
        center_distance = last_point_center_distance
        poses_iterable = range(len(poses))

    excluded_poses = []
    generated_poses = []
    for i in poses_iterable:
        if np.linalg.norm(positions[i] - center_position) <= center_distance:
            break
        excluded_poses.append(i)

        # Generate new point
        direction = positions[i] - center_position
        unit_direction = direction / np.linalg.norm(direction)
        new_position = center_position + unit_direction * center_distance
        generated_poses.append(poses[i] + new_position)

    return center_position, generated_poses, excluded_poses
