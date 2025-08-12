import numpy as np
from typing import Literal, Optional

def apply_speckle_noise(
    image: np.ndarray,
    variance: float = 0.1,
    distribution: Literal['normal', 'uniform', 'gamma'] = 'normal',
    intensity_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Applies multiplicative speckle noise to an image array."""
    if seed is not None:
        np.random.seed(seed)

    if distribution == 'normal':
        noise = np.random.normal(0, np.sqrt(variance), image.shape)
    elif distribution == 'uniform':
        limit = np.sqrt(3 * variance)
        noise = np.random.uniform(-limit, limit, image.shape)
    elif distribution == 'gamma':
        shape, scale = 1 / variance, variance
        noise = np.random.gamma(shape, scale, image.shape) - 1
    else:
        print(f"Warning: Unknown distribution '{distribution}'. Using 'normal'.")
        noise = np.random.normal(0, np.sqrt(variance), image.shape)
        
    image_float = image.astype(np.float64) / 255.0
    noisy_image = image_float * (1 + noise * intensity_scale)
    return (np.clip(noisy_image, 0, 1) * 255).astype(np.uint8)