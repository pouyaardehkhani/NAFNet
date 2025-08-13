import cv2
import numpy as np
from typing import Literal, Optional, Any

def _create_motion_kernel(length: int, angle: float) -> np.ndarray:
    """Internal helper to create a motion blur kernel."""
    kernel_size = int(length) * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    x, y = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    line_points = zip(np.round(center + dx).astype(int) for dx in np.linspace(-x*length/2, x*length/2, int(length))), \
                    zip(np.round(center + dy).astype(int) for dy in np.linspace(-y*length/2, y*length/2, int(length)))
    for (x_coord, y_coord) in zip(*line_points):
        for x_c, y_c in zip(x_coord, y_coord):
             kernel[y_c, x_c] = 1

    return kernel / kernel.sum() if kernel.sum() > 0 else kernel

def _apply_motion_blur(image: np.ndarray, **params) -> np.ndarray:
    """Internal helper to apply motion blur to an image array."""
    length = params.get('length', 10)
    angle = params.get('angle', 0)
    kernel = _create_motion_kernel(length, angle)
    return cv2.filter2D(image, -1, kernel)

def _apply_vibration_noise(image: np.ndarray, **params) -> np.ndarray:
    """Internal helper to apply vibration/shake noise."""
    intensity, frequency = params.get('intensity', 1.0), params.get('frequency', 30)
    h, w = image.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    dx = intensity * np.sin(2 * np.pi * y_coords / frequency)
    dy = intensity * np.cos(2 * np.pi * x_coords / frequency)
    map_x, map_y = (x_coords + dx).astype(np.float32), (y_coords + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_motion_effect(
    image: np.ndarray,
    effect: Literal['motion_blur', 'vibration', 'both'] = 'motion_blur',
    **effect_params: Any
) -> np.ndarray:
    """Applies motion effects to an image array."""
    noisy_image = image.copy()
    if effect in ('motion_blur', 'both'):
        noisy_image = _apply_motion_blur(noisy_image, **effect_params)
    if effect in ('vibration', 'both'):
        if 'seed' in effect_params: np.random.seed(effect_params['seed'])
        noisy_image = _apply_vibration_noise(noisy_image, **effect_params)
    return noisy_image