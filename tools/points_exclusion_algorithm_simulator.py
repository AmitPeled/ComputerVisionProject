import numpy as np
import matplotlib.pyplot as plt


def compute_ray(p1, p2):
    """Compute the direction vector (dx, dy) for a ray from p1 to p2."""
    return np.array(p2) - np.array(p1)


def ray_intersection(p1, d1, p2, d2):
    """Find intersection of two rays, if valid."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    if np.linalg.det(A) == 0:
        return None  # Parallel rays

    t, s = np.linalg.solve(A, b)

    if t < 0 or s < 0:
        return None  # Intersection is behind one of the rays

    return p1 + t * d1  # Valid intersection point


import numpy as np
import matplotlib.pyplot as plt


def compute_ray(p1, p2):
    """Compute the direction vector (dx, dy) for a ray from p1 to p2."""
    return np.array(p2) - np.array(p1)


def ray_intersection(p1, d1, p2, d2):
    """Find intersection of two rays, if valid."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    if np.linalg.det(A) == 0:
        return None  # Parallel rays

    t, s = np.linalg.solve(A, b)

    if t < 0 or s < 0:
        return None  # Intersection is behind one of the rays

    return p1 + t * d1  # Valid intersection point


def find_valid_trajectory(poses):
    """Find the new initial and last frame indices for a smooth trajectory."""
    intersections = []  # List[Tuple(start_idx, end_idx, intersection, num_excluded_points)]
    start_idx = 0
    while start_idx < len(poses) - 3:
        end_idx = len(poses) - 1
        while end_idx - start_idx > 3:
            start_ray = compute_ray(poses[start_idx + 1], poses[start_idx])
            end_ray = compute_ray(poses[end_idx - 1], poses[end_idx])

            intersection = ray_intersection(poses[start_idx + 1], start_ray, poses[end_idx - 1], end_ray)

            if intersection is not None:
                intersections.append((start_idx, end_idx, intersection, len(poses) + start_idx - end_idx - 1))
                break  # Found a valid smooth trajectory

            end_idx -= 1  # Exclude last frame and retry
        start_idx += 1

    return intersections


# Example usage
import pickle
import random

with open('clicked_points_bad.pkl', 'rb') as f:
    pos = pickle.load(f)
intersections = find_valid_trajectory(pos)

# Visualization
plt.plot(*zip(*pos), marker='o', label='Original Path')
for start, end, intersection, _ in intersections:
    color = (random.random(), random.random(), random.random(), 0.5)  # Random color with transparency
    # plt.scatter(*intersection, color=color, label='Intersection')
    plt.plot([pos[start][0], intersection[0]], [pos[start][1], intersection[1]], '-', color=color, label='Start Ray')
    plt.plot([pos[end][0], intersection[0]], [pos[end][1], intersection[1]], '-', color=color, label='End Ray')
plt.legend()
plt.show()