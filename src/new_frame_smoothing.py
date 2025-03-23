from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline


def find_best_original_edge_points(poses: List[np.ndarray]) -> Tuple[List, List]:
    """
    Find edge points and exclude the bad points in between.
    :param poses: C2W poses extracted from colman images.txt file
    :return: (List of the new poses with modified edge points after excluding the bad points in between,
              List of excluded poses)
    """

    # The xyz poses are in 3d space.
    # Find the best start and end poses by the following method:
    #     * Draw a ray from poses[-2] that passes through poses[-1]
    #     * Draw a ray from poses[1] that passes through poses[0]
    #     * The following calculation should take care of the distance to the two rays, and the angles between them
    #     * The goal is to iteratively (O(n^2)) continue to remove one tail or promote one head to find a value that
    #       is optimal in consideration also of the number of poses that were excluded away.
    def find_valid_trajectory(pos: np.ndarray):
        """Find the new initial and last frame indices for a smooth trajectory."""

        def compute_ray(p1, p2):
            """Compute the direction vector (dx, dy, dz) for a ray from p1 to p2."""
            return np.array(p2) - np.array(p1)

        def ray_distance(p1, d1, p2, d2):
            """Compute the shortest distance between two skewed rays in 3D."""
            n = np.cross(d1, d2)
            if np.linalg.norm(n) == 0:
                return None, None, None  # Parallel rays
            n = n / np.linalg.norm(n)

            t = np.dot(np.cross((p2 - p1), d2), n) / np.dot(np.cross(d1, d2), n)
            s = np.dot(np.cross((p1 - p2), d1), n) / np.dot(np.cross(d2, d1), n)

            closest_p1 = p1 + t * d1
            closest_p2 = p2 + s * d2
            distance = np.linalg.norm(closest_p1 - closest_p2)

            return closest_p1, closest_p2, distance

        def is_between(point, ref1, ref2, direction):
            """Check if 'point' is between 'ref1' and 'ref2' along the given direction."""
            ref1_proj = np.dot(ref1 - point, direction)
            ref2_proj = np.dot(ref2 - point, direction)
            return ref1_proj * ref2_proj <= 0  # The signs should be opposite

        distances = []  # List[Tuple(start_idx, end_idx, closest_p1, closest_p2, distance, start_ray, end_ray)]
        start_idx = 0
        while start_idx < len(pos) - 3:
            end_idx = len(pos) - 1
            while end_idx - start_idx > 3:
                start_ray = compute_ray(pos[start_idx + 1], pos[start_idx])
                end_ray = compute_ray(pos[end_idx - 1], pos[end_idx])

                closest_p1, closest_p2, distance = ray_distance(pos[start_idx + 1], start_ray, pos[end_idx - 1],
                                                                end_ray)
                if closest_p1 is not None:
                    # Ensure pos[start_idx] is between closest_p1 and pos[start_idx + 1]
                    if not is_between(pos[start_idx], closest_p1, pos[start_idx + 1], start_ray):
                        end_idx -= 1
                        continue

                    # Ensure pos[end_idx] is between closest_p2 and pos[end_idx - 1]
                    if not is_between(pos[end_idx], closest_p2, pos[end_idx - 1], end_ray):
                        end_idx -= 1
                        continue

                    distances.append((start_idx, end_idx, closest_p1, closest_p2, distance, start_ray, end_ray))
                    break  # Found a valid smooth trajectory

                end_idx -= 1  # Exclude last frame and retry
            start_idx += 1

        return distances

    xyz_poses = np.array([pose[:3, 3] for pose in poses])
    distances = find_valid_trajectory(xyz_poses)

    if distances:
        optimal_exclusion = None

        print(len(distances))
        for distance in distances:
            start_idx, end_idx, *_ = distance
            num_excluded_poses = len(poses) - 1 - end_idx + start_idx
            print(f'Num excluded camera poses: {num_excluded_poses}')
            optimal_exclusion = optimal_exclusion or (num_excluded_poses, start_idx, end_idx)
            # Take the minimum pose exclusion amount
            if (num_excluded_poses < optimal_exclusion[0] or
                # Prefer excluding poses from the end instead of from the start
               (num_excluded_poses == optimal_exclusion[0] and start_idx < optimal_exclusion[1])):
                optimal_exclusion = (num_excluded_poses, start_idx, end_idx)
        print(f'Optimal camera pose exclusion amount is {optimal_exclusion[0]} where start_idx={optimal_exclusion[1]}!')
        return poses[optimal_exclusion[1]:optimal_exclusion[2]], poses[:optimal_exclusion[1]] + poses[optimal_exclusion[2]:]

    raise ValueError('No valid rays were found.')


def generate_smooth_pose_trajectory_connection(
        poses: List[np.ndarray],
        first_poses_env: List[np.ndarray],
        last_poses_env: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate new poses to smoothly connect end poses to the start poses.
    :param poses: C2W poses, after excluding bad points between last and first points
    :param first_poses_env: Several first positions of the trajectory to be connected
    :param last_poses_env: Several last positions of the trajectory to be connected
    :return: (List of all poses in the smoothed trajectory, List of generated poses to smoothly connect last and first
              points)
    """
    # Use cubic spline the passes through first_poses_env and last_poses_env.
    # Calculate the curve's length and generate new points, in an amount such that the curve's length between each
    # two points equals the average length between two consecutive points of first and last poses.
    # Now, don't forget to consider the camera rotations as well in the generated poses - use SLERP for that.

    first_positions = np.array([pose[:3, 3] for pose in first_poses_env])
    last_positions = np.array([pose[:3, 3] for pose in last_poses_env])

    # Compute cumulative arc length
    all_positions = np.vstack((last_positions, first_positions))
    segment_lengths = np.linalg.norm(np.diff(all_positions, axis=0), axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Normalize lengths to [0, 1] for cubic spline
    normalized_lengths = cumulative_lengths / cumulative_lengths[-1]

    # Create spline based on arc length
    spline = CubicSpline(normalized_lengths, all_positions, axis=0)

    # Determine number of interpolation points based on average spacing
    avg_spacing = np.mean(segment_lengths)
    num_interp_points = 10  # int(cumulative_lengths[-1] / avg_spacing)

    # Distribute new points uniformly along arc length
    interp_arc_lengths = np.linspace(0, 1, num_interp_points)
    interp_positions = spline(interp_arc_lengths)

    # Interpolate rotations using Slerp
    first_rotations = Rotation.from_matrix([pose[:3, :3] for pose in first_poses_env])
    last_rotations = Rotation.from_matrix([pose[:3, :3] for pose in last_poses_env])

    slerp = Slerp([0, 1], Rotation.from_matrix([last_rotations.mean().as_matrix(), first_rotations.mean().as_matrix()]))
    interp_rotations = slerp(interp_arc_lengths)

    # Construct new poses
    generated_poses = []
    for pos, rot in zip(interp_positions, interp_rotations.as_matrix()):
        new_pose = np.eye(4)
        new_pose[:3, :3] = rot
        new_pose[:3, 3] = pos
        generated_poses.append(new_pose)

    return poses + generated_poses, generated_poses


def apply_smoothing_algorithm(poses: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    new_poses, excluded_poses = find_best_original_edge_points(poses)
    smoothed_poses, generated_poses = generate_smooth_pose_trajectory_connection(new_poses,
                                                                                 new_poses[0:3],
                                                                                 new_poses[-4:-1])
    return smoothed_poses, generated_poses, excluded_poses
