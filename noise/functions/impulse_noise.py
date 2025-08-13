import numpy as np
from typing import Literal, Optional

def _add_shot_noise(
    image: np.ndarray, 
    intensity: float = 0.05, 
) -> np.ndarray:
    """Adds shot (Poisson) noise to an image."""
    normalized_img = image.astype(np.float64) / 255.0
    scaled_img = normalized_img * intensity * 255.0
    poisson_noise = np.random.poisson(scaled_img)
    noisy_img = normalized_img * 255.0 + (poisson_noise - scaled_img) / intensity
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def _add_quantization_noise(
    image: np.ndarray, 
    bits: int = 6, 
) -> np.ndarray:
    """Adds quantization noise by reducing bit depth."""
    bits = max(1, min(8, bits))
    levels = 2 ** bits
    quantized_float = np.round(image.astype(np.float64) / 255.0 * (levels - 1))
    return (quantized_float / (levels - 1) * 255.0).astype(np.uint8)

def apply_impulse_noise(
    image: np.ndarray,
    noise_type: Literal['shot', 'quantization', 'both'] = 'both',
    shot_intensity: float = 0.05,
    quant_bits: int = 6,
    seed: Optional[int] = None
) -> np.ndarray:
    """Applies impulse noise to an image array."""
    if seed is not None:
        np.random.seed(seed)
        
    noisy_image = image.copy()
    if noise_type in ('shot', 'both'):
        noisy_image = _add_shot_noise(noisy_image, shot_intensity)
    if noise_type in ('quantization', 'both'):
        noisy_image = _add_quantization_noise(noisy_image, quant_bits)
    
    return noisy_image