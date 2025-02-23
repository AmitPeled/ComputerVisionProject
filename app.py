import os
import argparse
import re
import time

from src.visualizer import CameraVisualizer
from src.loader import load_quick, load_nerf, load_colmap
from src.utils import load_image, rescale_cameras, recenter_cameras

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Define the Dash app
app = dash.Dash(__name__)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--format', default='quick', choices=['quick', 'nerf', 'colmap'])
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
    poses, colors = poses[:40], colors[:40]

    return args, poses, colors, legends, images


def main():
    args, poses, colors, legends, images = get_params()

    viz = CameraVisualizer(poses, legends, colors, images=images)
    fig = viz.update_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True, show_ticklabels=True,
                            show_background=True, y_up=args.y_up)

    app.layout = html.Div([
        dcc.Graph(id='camera-visualization', figure=fig),
        dcc.Interval(id='interval-update', interval=2000, n_intervals=0),
        # dcc.Store(id='camera-store', data={}),  # Store camera updates
        # html.Pre(id='camera-output')  # Display camera settings
    ])

    # Callback to update the figure dynamically
    @app.callback(
        Output('camera-visualization', 'figure'),
        Input('interval-update', 'n_intervals')
    )
    def update_figure(n):
        pose_index = n % len(poses)  # Use n_intervals to track updates
        new_colors = ['red'] * len(colors)
        new_colors[pose_index] = 'green'
        new_viz = CameraVisualizer(poses, legends, new_colors, images=images)
        new_fig = new_viz.update_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True,
                                        show_ticklabels=True,
                                        show_background=True, y_up=args.y_up, highlight_index=pose_index)

        return new_fig


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
