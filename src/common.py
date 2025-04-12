from dataclasses import dataclass
import numpy as np

@dataclass
class ImagePose:
    pose: np.ndarray
    image_path: str
    image: np.ndarray
