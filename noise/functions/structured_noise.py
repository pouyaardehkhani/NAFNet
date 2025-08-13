import numpy as np
from typing import Literal, Any

def _add_periodic_noise(image: np.ndarray, **params) -> np.ndarray:
    h, w = image.shape[:2]
    frequency, amplitude = params.get('frequency', 0.1), params.get('amplitude', 20.0)
    phase_x, phase_y = params.get('phase_x', 0.0), params.get('phase_y', 0.0)
    direction = params.get('direction', 'both')
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    noise = np.zeros((h, w), dtype=np.float32)
    if direction in ('horizontal', 'both'): noise += np.sin(2 * np.pi * frequency * x + phase_x)
    if direction in ('vertical', 'both'): noise += np.sin(2 * np.pi * frequency * y + phase_y)
    noisy_image = image.astype(np.float32) + (noise * amplitude)[:, :, np.newaxis]
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def _add_banding_noise(image: np.ndarray, **params) -> np.ndarray:
    h, w = image.shape[:2]
    band_width, intensity = params.get('band_width', 10), params.get('band_intensity', 30.0)
    direction = params.get('banding_direction', 'horizontal')
    spacing, random_offset = params.get('band_spacing', 1), params.get('random_band_offset', True)
    noise = np.zeros((h, w), dtype=np.float32)
    if direction == 'horizontal':
        for i in range(0, h, band_width + spacing):
            offset = np.random.uniform(-0.3, 0.3) if random_offset else 0
            noise[i : i + band_width, :] = intensity * (1 + offset)
    else:
        for i in range(0, w, band_width + spacing):
            offset = np.random.uniform(-0.3, 0.3) if random_offset else 0
            noise[:, i : i + band_width] = intensity * (1 + offset)
    noisy_image = image.astype(np.float32) + noise[:, :, np.newaxis]
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_structured_noise(
    image: np.ndarray,
    noise_type: Literal['periodic', 'banding', 'both'] = 'both',
    **noise_params: Any
) -> np.ndarray:
    """Applies structured noise (periodic and/or banding) to an image array."""
    noisy_image = image.copy()
    if noise_type in ('periodic', 'both'):
        noisy_image = _add_periodic_noise(noisy_image, **noise_params)
    if noise_type in ('banding', 'both'):
        noisy_image = _add_banding_noise(noisy_image, **noise_params)
    return noisy_image