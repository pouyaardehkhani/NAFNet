import numpy as np
from typing import Literal, Any, Callable

def apply_mathematical_noise(
    image: np.ndarray,
    noise_type: Literal['awgn', 'rayleigh', 'exponential', 'gamma'],
    **noise_params: Any
) -> np.ndarray:
    """Applies a specified mathematical noise model to an image array."""
    noise_map = {
        'awgn': {'func': np.random.normal, 'defaults': {'loc': 0, 'scale': 25}},
        'rayleigh': {'func': np.random.rayleigh, 'defaults': {'scale': 30}},
        'exponential': {'func': np.random.exponential, 'defaults': {'scale': 20}},
        'gamma': {'func': np.random.gamma, 'defaults': {'shape': 2, 'scale': 25}}
    }
    config = noise_map.get(noise_type)
    if not config:
        print(f"Warning: Unknown math noise type '{noise_type}'. Returning original image.")
        return image

    params = config['defaults']
    if noise_type == 'awgn' and 'sigma' in noise_params:
        params['scale'] = noise_params.pop('sigma')
    params.update(noise_params)
        
    noise_array = config['func'](**params, size=image.shape)
    noisy_image_float = image.astype(np.float64) + noise_array
    return np.clip(noisy_image_float, 0, 255).astype(np.uint8)