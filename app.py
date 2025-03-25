from typing import List
import os
import argparse
import re
import time
import numpy as np
import cv2

from src.visualizer import CameraVisualizer
from src.loader import load_quick, load_nerf, load_colmap
from src.utils import load_image, rescale_cameras, recenter_cameras
from src.new_frame_smoothing import apply_smoothing_algorithm

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Define the Dash app
app = dash.Dash(__name__)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--format', default='colmap', choices=['quick', 'nerf', 'colmap'])
    parser.add_argument('--type', default=None, choices=[None, 'sph', 'xyz', 'elu', 'c2w', 'w2c'])
    parser.add_argument('--no_images', action='store_true')
    parser.add_argument('--mesh_path', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--scene_size', type=int, default=5)
    parser.add_argument('--y_up', action='store_true')
    parser.add_argument('--recenter', action='store_true')
    parser.add_argument('--rescale', type=float, default=None)

    args = parser.parse_args()

    root_path = args.root

    poses = []
    legends = []
    colors = []
    images = None
    image_paths = []

    if args.format == 'quick':
        poses, legends, colors, image_paths = load_quick(root_path, args.type)

    elif args.format == 'nerf':
        poses, legends, colors, image_paths = load_nerf(root_path)

    elif args.format == 'colmap':
        poses, legends, colors, image_paths = load_colmap(root_path)

    if args.recenter:
        poses = recenter_cameras(poses)

    if args.rescale is not None:
        poses = rescale_cameras(poses, args.rescale)

    if args.y_up:
        for i in range(0, len(poses)):
            poses[i] = poses[i][[0, 2, 1, 3]]
            poses[i][1, :] *= -1

    if not args.no_images:
        images = []
        for fpath in image_paths:
            if fpath is None:
                images.append(None)
                continue

            if not os.path.exists(fpath):
                images.append(None)
                print(f'Image not found at {fpath}')
                continue

            images.append(load_image(fpath, sz=args.image_size))

    legends = [re.search(r'(\d\d)\.(\w+$)', name).group(1) for name in legends]
    poses, colors, legends = poses[:40], colors[:40], legends[:40]

    return args, poses, colors, legends, images, image_paths


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
    SAVE_DIR = r'G:\My Drive\NeRF_Training'

    images = [cv2.imread(path)[:, :, ::-1] for path in new_image_paths]  # Convert BGR to RGB
    np.save(f"{SAVE_DIR}/images.npy", np.array(images, dtype=np.uint8))

    np.save(f"{SAVE_DIR}/poses.npy", np.array(new_poses))
    np.save(f"{SAVE_DIR}/generated_poses.npy", np.array(generated_poses))


def main():
    args, poses, colors, legends, images, image_paths = get_params()
    poses = generate_mock_c2w_values()

    new_poses, new_image_paths, generated_poses, excluded_poses = apply_smoothing_algorithm(poses, image_paths)
    # new_poses, generated_poses, excluded_poses = poses, [], [] # apply_smoothing_algorithm(poses)

    save_nerf_files(new_image_paths, new_poses, generated_poses)

    viz = CameraVisualizer(poses, new_poses + generated_poses, generated_poses, excluded_poses, legends, colors,
                           images=images)

    gif_fig = viz.update_gif_figure(0)
    camera_fig = viz.update_camera_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True,
                                          show_ticklabels=True,
                                          show_background=True, y_up=args.y_up)

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='camera-visualization', figure=camera_fig),
            # dcc.Store(id='camera-store', data={}),  # Store camera updates
            # html.Pre(id='camera-output')  # Display camera settings
            html.H3("Camera Visualization")
        ], style={"width": "50%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(id='running-gif', figure=gif_fig),
            html.H3("Running GIF")
        ], style={"width": "50%", "display": "inline-block"}),
        # dcc.Interval(id='interval-update', interval=2000, n_intervals=0),
    ])

    # Callback to update the figure dynamically
    @app.callback(
        Output('camera-visualization', 'figure'),
        Output('running-gif', 'figure'),
        Input('interval-update', 'n_intervals')
    )
    def update_figure(n):
        pose_index = n % len(poses)
        new_gif_fig = viz.update_gif_figure(pose_index)
        new_camera_fig = viz.update_camera_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True,
                                                  show_ticklabels=True,
                                                  show_background=True, y_up=args.y_up, highlight_index=pose_index)

        return new_camera_fig, new_gif_fig


@app.callback(
    [Output('camera-store', 'data'),
     Output('camera-output', 'children')],
    [Input('camera-visualization', 'relayoutData')]
)
def capture_camera_view(relayout_data):
    if relayout_data and 'scene.camera' in relayout_data:
        camera = relayout_data['scene.camera']
        print(f"Current Camera View:\n{camera}", flush=True)  # Print to console
        return camera, f"Camera: {camera}"  # Store and display it
    return dash.no_update, dash.no_update  # No change if no update


# Run the Dash app
if __name__ == '__main__':
    main()
    app.run_server(debug=False)
