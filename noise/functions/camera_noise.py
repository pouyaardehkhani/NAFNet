import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, Any

def _generate_iso_sensor_noise(
    image: np.ndarray,
    **noise_params: Any
) -> np.ndarray:
    """Generates realistic ISO/Sensor noise."""
    iso_level = noise_params.get('iso_level', 800)
    dark_noise = noise_params.get('dark_noise', 0.05)
    read_noise = noise_params.get('read_noise', 0.02)
    shot_noise_factor = noise_params.get('shot_noise_factor', 1.0)
    
    image_float = image.astype(np.float32) / 255.0
    iso_factor = np.sqrt(iso_level / 100.0)
    shot_noise = np.random.poisson(image_float * 255 * shot_noise_factor * iso_factor) / 255.0 - image_float
    shot_noise *= 0.1
    dark_current = np.random.normal(0, dark_noise * iso_factor, image.shape)
    readout_noise = np.random.normal(0, read_noise * iso_factor, image.shape)
    
    noisy_image = image_float + shot_noise + dark_current + readout_noise
    return (np.clip(noisy_image, 0, 1) * 255).astype(np.uint8)

def _generate_chromatic_noise(
    image: np.ndarray,
    **noise_params: Any
) -> np.ndarray:
    """Generates chromatic aberration noise."""
    color_shift = noise_params.get('color_shift', 1.0)
    blur_radius = noise_params.get('blur_radius', 1.0)

    image_float = image.astype(np.float32) / 255.0
    h, w, _ = image_float.shape
    center_y, center_x = h / 2, w / 2

    y, x = np.mgrid[:h, :w]
    dist_x, dist_y = x - center_x, y - center_y
    dist = np.sqrt(dist_x**2 + dist_y**2)
    dist[dist == 0] = 1
    
    shift_amount = (dist / dist.max()) * color_shift
    
    map_x_r = (x - shift_amount * (dist_x / dist)).astype(np.float32)
    map_y_r = (y - shift_amount * (dist_y / dist)).astype(np.float32)
    map_x_b = (x + shift_amount * (dist_x / dist)).astype(np.float32)
    map_y_b = (y + shift_amount * (dist_y / dist)).astype(np.float32)

    b, g, r = cv2.split(image_float)
    r_shifted = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR)
    b_shifted = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR)
    
    r_blurred = ndimage.gaussian_filter(r_shifted, sigma=blur_radius * 1.2)
    g_blurred = ndimage.gaussian_filter(g, sigma=blur_radius)
    b_blurred = ndimage.gaussian_filter(b_shifted, sigma=blur_radius * 0.8)
    
    result = cv2.merge([b_blurred, g_blurred, r_blurred])
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def apply_camera_noise(
    image: np.ndarray,
    noise_type: str = 'iso',
    noise_params: Dict[str, Any] = None
) -> np.ndarray:
    """Applies a specified camera noise to an image array."""
    if noise_params is None:
        noise_params = {}

    if noise_type == 'iso':
        return _generate_iso_sensor_noise(image, **noise_params)
    elif noise_type == 'chromatic':
        return _generate_chromatic_noise(image, **noise_params)
    else:
        print(f"Warning: Unknown camera noise type '{noise_type}'. Returning original image.")
        return image