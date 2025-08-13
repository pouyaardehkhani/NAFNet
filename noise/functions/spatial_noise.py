import numpy as np
from typing import Literal, Dict, Any

def _generate_checkerboard_pattern(shape: tuple, **params) -> np.ndarray:
    h, w = shape[:2]
    block_size = params.get('block_size', 16)
    checker = np.kron([[1, 0], [0, 1]], np.ones((h // (2*block_size)+1, w // (2*block_size)+1)))
    pattern = np.repeat(np.repeat(checker, block_size, axis=0), block_size, axis=1)[:h, :w]
    return (pattern * 2 - 1)

def _generate_stripe_pattern(shape: tuple, **params) -> np.ndarray:
    h, w = shape[:2]
    stripe_width = params.get('stripe_width', 5)
    direction = params.get('direction', 'horizontal')
    y, x = np.mgrid[:h, :w]
    if direction == 'horizontal': pattern = (y // stripe_width % 2).astype(float)
    elif direction == 'vertical': pattern = (x // stripe_width % 2).astype(float)
    else: pattern = ((x + y) // stripe_width % 2).astype(float)
    return (pattern * 2 - 1)

def _generate_ring_pattern(shape: tuple, **params) -> np.ndarray:
    h, w = shape[:2]
    spacing = params.get('ring_spacing', 20)
    center_x, center_y = params.get('center_x', w // 2), params.get('center_y', h // 2)
    y, x = np.mgrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return np.sin(distance / spacing * 2 * np.pi)

def _generate_moire_pattern(shape: tuple, **params) -> np.ndarray:
    h, w = shape[:2]
    freq1, freq2 = params.get('freq1', 0.1), params.get('freq2', 0.12)
    angle1, angle2 = np.deg2rad(params.get('angle1', 0)), np.deg2rad(params.get('angle2', 15))
    y, x = np.mgrid[:h, :w]
    p1 = np.sin((x * np.cos(angle1) + y * np.sin(angle1)) * freq1 * 2 * np.pi)
    p2 = np.sin((x * np.cos(angle2) + y * np.sin(angle2)) * freq2 * 2 * np.pi)
    return p1 * p2

def apply_spatial_pattern(
    image: np.ndarray,
    pattern: Literal['checkerboard', 'stripe', 'ring', 'moire'],
    **pattern_params: Any
) -> np.ndarray:
    """Applies a specified spatial noise pattern to an image array."""
    pattern_generators = {
        'checkerboard': _generate_checkerboard_pattern,
        'stripe': _generate_stripe_pattern,
        'ring': _generate_ring_pattern,
        'moire': _generate_moire_pattern,
    }
    generator_func = pattern_generators.get(pattern)
    if not generator_func:
        print(f"Warning: Unknown pattern type '{pattern}'. Returning original image.")
        return image
    
    noise_pattern_2d = generator_func(image.shape, **pattern_params)
    intensity = pattern_params.get('intensity', 0.3)
    noise = noise_pattern_2d * intensity * 255.0
    
    noisy_image = image.astype(np.float32) + noise[:, :, np.newaxis]
    return np.clip(noisy_image, 0, 255).astype(np.uint8)