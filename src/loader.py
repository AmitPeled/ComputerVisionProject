from PIL import Image
import os
from pathlib import Path

from .config import *
from .common import *
from .utils import load_image, qvec_to_rotmat, rotmat


def get_image(image_path, image_size=256):
    return load_image(image_path, sz=image_size)


def get_images(image_paths, image_size=256):
    images = []
    for fpath in image_paths:
        if fpath is None:
            images.append(None)
            continue

        if not os.path.exists(fpath):
            images.append(None)
            print(f'Image not found at {fpath}')
            continue

        images.append(get_image(fpath, image_size))
    return images


def compress_jpeg_quality(input_path, quality=15):
    output_dir = Path(r'C:\WINDOWS\Temp')
    output_name = Path(input_path).stem + '_reduced_quality' + Path(input_path).suffix
    out_path = str(output_dir / output_name)
    os.makedirs(output_dir, exist_ok=True)
    if input_path.lower().endswith(('.jpg', '.jpeg')):
        img = Image.open(input_path)
        img.save(out_path, 'JPEG', quality=quality, optimize=True)
    return out_path


def get_new_images(new_image_paths, image_size):
    if not USE_ALG and GENERATE:
        generated = []
        for path in new_image_paths[len(new_image_paths) - len(GENERATED_IDXS):]:
            generated.append(compress_jpeg_quality(path))
        new_image_paths[len(new_image_paths) - len(GENERATED_IDXS):] = generated
    return get_images(new_image_paths, image_size)


def old_load_colmap(root_path):
    image_poses = []
    poses = []
    image_paths = []

    root_path = r'C:\Users\iritp\Downloads\Instant-NGP-for-RTX-5000\Instant-NGP-for-RTX-5000\scripts\shoes'
    pose_path = os.path.join(root_path, 'colmap_text', 'images.txt')
    print(f'Load poses from {pose_path}')
    fin = open(pose_path, 'r')
    up = np.zeros(3)

    lines = fin.readlines()
    images_lines = {'_'.join(line.strip().split(' ')[9:]): line.strip() for line in lines[4::2]}

    import collections
    sorted_images = collections.OrderedDict(sorted(images_lines.items()))
    for fname, line in sorted_images.items():
        print(fname)
        elems = line.split(' ')

        # fpath = os.path.join(root_path, fname)
        fpath = os.path.join(r'C:\Users\iritp\Downloads\resized_shoes_images', fname)

        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        rot = qvec_to_rotmat(-qvec)
        tvec = tvec.reshape(3)

        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, -1] = tvec
        c2w = np.linalg.inv(w2c)

        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1  # flip whole world upside down

        up += c2w[0:3, 1]

        poses.append(c2w)
        image_paths.append(fpath)

    fin.close()

    up = up / np.linalg.norm(up)
    up_rot = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    up_rot = np.pad(up_rot, [0, 1])
    up_rot[-1, -1] = 1

    for i in range(0, len(poses)):
        poses[i] = np.matmul(up_rot, poses[i])
        image_poses.append(ImagePose(pose=poses[i], image_path=image_paths[i], image=get_image(image_paths[i])))

    return image_poses


def load_colmap(root_path):
    from pathlib import Path
    import json
    import collections

    poses = []
    legends = []
    colors = []
    image_paths = []
    root_path = Path(r'C:\Users\iritp\Downloads\Instant-NGP-for-RTX-5000\Instant-NGP-for-RTX-5000\scripts\shoes')

    pose_path = root_path / 'transforms.json'
    transforms = json.loads(pose_path.read_text())
    f_by_img_path = {f["file_path"]: f for f in transforms["frames"]}
    for img_path, f in collections.OrderedDict(sorted(f_by_img_path.items())).items():
        print(img_path)
        poses.append(np.array(f["transform_matrix"]))
        image_paths.append(str(root_path / f["file_path"]))
        colors.append("blue")
        legends.append(Path(img_path).stem)

    return poses, legends, colors, image_paths
