import numpy as np
import matplotlib.pyplot as plt
import random


def find_valid_trajectory(pos):
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

    if distances:
        print(len(distances))
        for distance in distances:
            start_idx, end_idx, *_ = distance
            print(f'Num excluded frames: {len(pos) - 1 - end_idx + start_idx}')
        return distances
    raise ValueError('No valid rays were found.')


import pickle

random.seed(5)

with open('clicked_points_bad.pkl', 'rb') as f:
    pos = pickle.load(f)
    pos = [(*p, random.random()) for p in pos]

# Convert to NumPy array
pos = np.array(pos)
distances = find_valid_trajectory(pos)

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], marker='o', label='Original Path')

index = 0


def update_plot(index):
    ax.cla()
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], marker='o', label='Original Path')

    if index < len(distances):
        start, end, closest_p1, closest_p2, distance, start_ray, end_ray = distances[index]
        color = (random.random(), random.random(), random.random(), 0.5)  # Random color with transparency
        ax.scatter(*closest_p1, color=color, label='Closest Point 1')
        ax.scatter(*closest_p2, color=color, label='Closest Point 2')
        ax.plot([closest_p1[0], closest_p2[0]], [closest_p1[1], closest_p2[1]], [closest_p1[2], closest_p2[2]], '-',
                color=color, label=f'Distance = {distance:.2f}')

        # Draw the rays
        ray_length = 5  # Extend the rays for visualization
        start_ray_end = pos[start + 1] + start_ray / np.linalg.norm(start_ray) * ray_length
        end_ray_end = pos[end - 1] + end_ray / np.linalg.norm(end_ray) * ray_length

        ax.plot([pos[start + 1][0], start_ray_end[0]], [pos[start + 1][1], start_ray_end[1]],
                [pos[start + 1][2], start_ray_end[2]], '--', color=color, label='Start Ray')
        ax.plot([pos[end - 1][0], end_ray_end[0]], [pos[end - 1][1], end_ray_end[1]],
                [pos[end - 1][2], end_ray_end[2]], '--', color=color, label='End Ray')

    plt.draw()


update_plot(index)


def on_key(event):
    global index
    if event.key == 'right':
        index = min(index + 1, len(distances) - 1)
    elif event.key == 'left':
        index = max(index - 1, 0)
    update_plot(index)


fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
