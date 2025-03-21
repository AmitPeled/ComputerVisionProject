import os
import numpy as np
import json

# Define Paths
colmap_folder = r"G:\My Drive\COLMAP_Project\sparse_txt"
image_folder = r"G:\My Drive\COLMAP_Project\frames"
output_json = os.path.join(colmap_folder, "transforms.json")

# Read cameras.txt
camera_params = {}
with open(os.path.join(colmap_folder, "cameras.txt"), "r") as f:
    for line in f.readlines():
        if line.startswith("#"):
            continue
        elements = line.strip().split()
        camera_id = int(elements[0])
        model = elements[1]  # e.g., PINHOLE
        width = int(elements[2])
        height = int(elements[3])
        fx, fy, cx, cy = map(float, elements[4:8])
        camera_params[camera_id] = {"w": width, "h": height, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy}

# Read images.txt
frames = []
with open(os.path.join(colmap_folder, "images.txt"), "r") as f:
    lines = f.readlines()

    for i in range(0, len(lines), 2):
        if lines[i].startswith("#") or i + 1 >= len(lines):
            continue

        elements = lines[i].strip().split()
        image_id = int(elements[0])
        qw, qx, qy, qz, tx, ty, tz = map(float, elements[1:8])
        camera_id = int(elements[8])
        image_name = elements[9]

        # Convert COLMAP (q, t) to 4x4 Transform Matrix
        R = np.array([
            [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
        ])
        T = np.array([tx, ty, tz]).reshape((3, 1))
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = T.flatten()

        # Fix coordinate system (COLMAP -> Instant-NGP)
        transform_matrix[:3, 1:3] *= -1  # Invert Y and Z axes

        frames.append({
            "file_path": os.path.join(image_folder, image_name),
            "transform_matrix": transform_matrix.tolist()
        })

# Create JSON structure
data = {
    "fl_x": camera_params[camera_id]["fl_x"],
    "fl_y": camera_params[camera_id]["fl_y"],
    "cx": camera_params[camera_id]["cx"],
    "cy": camera_params[camera_id]["cy"],
    "w": camera_params[camera_id]["w"],
    "h": camera_params[camera_id]["h"],
    "camera_angle_x": 2 * np.arctan(camera_params[camera_id]["w"] / (2 * camera_params[camera_id]["fl_x"])),
    "frames": frames
}

# Save as transforms.json
with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"Saved to {output_json}")
