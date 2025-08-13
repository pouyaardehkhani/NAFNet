import numpy as np
from skimage import util
from typing import Literal, Optional, Any

def _generate_gaussian_noise(image: np.ndarray, **params) -> np.ndarray:
    std = params.get('std', 25)
    noise = np.random.normal(0, std, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _generate_salt_pepper_noise(image: np.ndarray, **params) -> np.ndarray:
    salt_prob = params.get('salt_prob', 0.05)
    pepper_prob = params.get('pepper_prob', 0.05)
    amount = salt_prob + pepper_prob
    if amount == 0: return image
    salt_vs_pepper = salt_prob / amount
    noisy_float = util.random_noise(image, mode='s&p', salt_vs_pepper=salt_vs_pepper, amount=amount)
    return (noisy_float * 255).astype(np.uint8)

def _generate_uniform_noise(image: np.ndarray, **params) -> np.ndarray:
    low, high = params.get('low', -30), params.get('high', 30)
    noise = np.random.uniform(low, high, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_additive_noise(
    image: np.ndarray,
    noise_type: Literal['gaussian', 'salt_pepper', 'uniform', 'poisson', 'speckle'],
    seed: Optional[int] = None,
    **noise_params: Any
) -> np.ndarray:
    """Applies a specified additive noise model to an image array."""
    if seed is not None:
        np.random.seed(seed)
    
    if noise_type == 'gaussian':
        return _generate_gaussian_noise(image, **noise_params)
    if noise_type == 'salt_pepper':
        return _generate_salt_pepper_noise(image, **noise_params)
    if noise_type == 'uniform':
        return _generate_uniform_noise(image, **noise_params)
    if noise_type == 'poisson':
        return (util.random_noise(image, mode='poisson') * 255).astype(np.uint8)
    if noise_type == 'speckle':
        return (util.random_noise(image, mode='speckle', var=noise_params.get('variance', 0.1)) * 255).astype(np.uint8)

    print(f"Warning: Unknown additive noise type '{noise_type}'. Returning original image.")
    return image