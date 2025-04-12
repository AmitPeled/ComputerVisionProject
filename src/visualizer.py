from PIL import Image
import plotly.graph_objects as go
import numpy as np


def calc_cam_cone_pts_3d(c2w, fov_deg, zoom=1.0):
    fov_rad = np.deg2rad(fov_deg)

    cam_x = c2w[0, -1]
    cam_y = c2w[1, -1]
    cam_z = c2w[2, -1]

    corn1 = [np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn2 = [-np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn3 = [-np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn4 = [np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn5 = [0, np.tan(fov_rad / 2.0), -1.0]

    corn1 = np.dot(c2w[:3, :3], corn1)
    corn2 = np.dot(c2w[:3, :3], corn2)
    corn3 = np.dot(c2w[:3, :3], corn3)
    corn4 = np.dot(c2w[:3, :3], corn4)
    corn5 = np.dot(c2w[:3, :3], corn5)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2) * zoom
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2) * zoom
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2) * zoom
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2) * zoom
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]
    corn5 = np.array(corn5) / np.linalg.norm(corn5, ord=2) * zoom
    corn_x5 = cam_x + corn5[0]
    corn_y5 = cam_y + corn5[1]
    corn_z5 = cam_z + corn5[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, corn_x5]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, corn_y5]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, corn_z5]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:
    def __init__(self, original_image_poses, new_image_poses, generated_image_poses, excluded_image_poses):
        self._fig = None
        self._camera_x = 1.0
        self._all_image_poses = new_image_poses + excluded_image_poses
        self._original_image_poses = original_image_poses
        self._new_image_poses = new_image_poses
        self._generated_image_poses = generated_image_poses
        self._excluded_image_poses = excluded_image_poses

        self._raw_images = [im.image for im in self._new_image_poses]
        self._bit_images = None
        self._image_colorscale = None

        self._bit_images = []
        self._image_colorscale = []
        for img in self._raw_images:
            bit_img, colorscale = self.encode_image(img)
            self._bit_images.append(bit_img)
            self._image_colorscale.append(colorscale)

    def encode_image(self, raw_image):
        """
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        """
        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        return bit_image, colorscale

    def update_gif_figure(self, index):
        """Update the figure to display the next image in the sequence."""
        if not self._raw_images or self._raw_images[index] is None:
            return go.Figure()  # Return empty figure if no images

        image_data = self._raw_images[index]

        fig = go.Figure()
        fig.add_trace(go.Image(z=image_data))  # Add image trace
        fig.update_layout(title="GIF Animation", margin=dict(l=10, r=10, t=30, b=10))
        return fig

    def update_camera_figure(
            self, scene_bounds,
            base_radius=0.0, fov_deg=50.,
            show_background=False, show_grid=False, show_ticklabels=False, y_up=False,
            highlight_index=None,
            scale_factor=0.9
    ):
        fig = go.Figure()

        for i in range(len(self._all_image_poses)):
            camera_lines_scale_factor = scale_factor if i != highlight_index else 1
            legend = str(i)

            pose = self._all_image_poses[i].pose
            if i == highlight_index:
                clr = 'green'
            elif any(np.array_equal(self._all_image_poses[i].pose, pose.pose) for pose in self._generated_image_poses):
                clr = 'deeppink'
            elif any(np.array_equal(self._all_image_poses[i].pose, pose.pose) for pose in self._excluded_image_poses):
                clr = 'lightgrey'
            else:
                clr = 'blue'

            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (0, 5)]
            cone = calc_cam_cone_pts_3d(pose, fov_deg) * camera_lines_scale_factor

            if i == highlight_index:  # Draw image
                image_dimensions_scale_factor = camera_lines_scale_factor / 2
                if self._bit_images and self._bit_images[i]:
                    raw_image = self._raw_images[i]

                    (H, W, C) = raw_image.shape

                    z = np.zeros((H, W)) + base_radius
                    (x, y) = np.meshgrid(
                        np.linspace(-1.0 * self._camera_x * image_dimensions_scale_factor,
                                    1.0 * self._camera_x * image_dimensions_scale_factor, W),
                        np.linspace(1.0 * image_dimensions_scale_factor, -1.0 * image_dimensions_scale_factor,
                                    H) * H / W
                    )

                    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
                    rot_xyz = np.matmul(xyz, pose[:3, :3].T) + pose[:3, -1]

            for (edge_index, edge) in enumerate(edges):
                (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                fig.add_trace(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                    line=dict(color=clr, width=3 * camera_lines_scale_factor),
                    name=legend, showlegend=(edge_index == 0)))

            if cone[0, 2] < 0:
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                    mode='text', text=legend, textposition='bottom center'))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                    mode='text', text=legend, textposition='top center'))

        # look at the center of scene
        fig.update_layout(
            height=720,
            autosize=True,
            hovermode=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor='right',
                x=0.99,
            ),
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    eye=dict(x=1, y=0, z=0),  # Camera position
                    center=dict(x=0, y=0, z=0),  # Look at the center
                    up=dict(x=0, y=0, z=1)  # Y-axis is up
                ),
                xaxis_title='X',
                yaxis_title='Z' if y_up else 'Y',
                zaxis_title='Y' if y_up else 'Z',
                xaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                yaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                zaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks='')
            )
        )

        self._fig = fig
        return fig
