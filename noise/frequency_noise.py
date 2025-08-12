import cv2
import numpy as np
from typing import Literal

_NOISE_GENERATORS = {
    'white': lambda shape, intensity: np.random.normal(0, intensity, shape),
    'pink': lambda shape, intensity: _generate_filtered_noise(shape, intensity, 'pink'),
    'blue': lambda shape, intensity: _generate_filtered_noise(shape, intensity, 'blue'),
    'brown': lambda shape, intensity: _generate_filtered_noise(shape, intensity, 'brown'),
}

def _generate_filtered_noise(
    shape: tuple, 
    intensity: float, 
    noise_type: Literal['pink', 'blue', 'brown']
) -> np.ndarray:
    """Generates pink, blue, or brown noise using FFT."""
    h, w = shape[:2]
    freq_y = np.fft.fftfreq(h)[:, np.newaxis]
    freq_x = np.fft.fftfreq(w)
    freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)
    freq_magnitude[0, 0] = 1
    
    if noise_type == 'pink':    # 1/f
        spectrum_scaling = 1.0 / freq_magnitude
    elif noise_type == 'blue':  # f
        spectrum_scaling = freq_magnitude
    else: # brown (1/f^2)
        spectrum_scaling = 1.0 / (freq_magnitude**2)
    spectrum_scaling[0, 0] = 0

    noise = np.zeros(shape)
    for c in range(shape[2]) if len(shape) == 3 else [None]:
        phases = np.random.uniform(0, 2 * np.pi, (h, w))
        complex_spectrum = spectrum_scaling * np.exp(1j * phases)
        channel_noise = np.real(np.fft.ifft2(complex_spectrum))
        if c is None: noise = channel_noise
        else: noise[:, :, c] = channel_noise

    std_dev = np.std(noise)
    if std_dev > 0:
        noise = (noise - np.mean(noise)) / std_dev * intensity
    return noise

def _apply_noise_blend(
    image: np.ndarray, 
    noise: np.ndarray, 
    blend_mode: str
) -> np.ndarray:
    """Applies noise to an image using a specified blend mode."""
    image_float = image.astype(np.float32) / 255.0
    
    if blend_mode == 'multiply':
        result = image_float * (1 + noise)
    elif blend_mode == 'overlay':
        mask = image_float < 0.5
        result = np.where(mask, 2 * image_float * (noise + 0.5), 1 - 2 * (1 - image_float) * (0.5 - noise))
    else: # 'add'
        result = image_float + noise
    
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

def apply_frequency_noise(
    image: np.ndarray,
    noise_type: str = 'white',
    intensity: float = 0.1,
    blend_mode: str = 'add'
) -> np.ndarray:
    """Applies frequency-domain noise to an image array."""
    generator_func = _NOISE_GENERATORS.get(noise_type)
    if not generator_func:
        print(f"Warning: Unknown frequency noise type '{noise_type}'. Returning original image.")
        return image

    noise = generator_func(image.shape, intensity)
    return _apply_noise_blend(image, noise, blend_mode)