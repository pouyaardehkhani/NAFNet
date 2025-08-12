"""
Example usage:
python noise_generator.py --img input.jpg --output output.jpg --noise_type pink --intensity 0.3 --blend_mode add
python noise_generator.py --img input.jpg --output output.jpg --noise_type blue --intensity 0.5 --blend_mode multiply
python noise_generator.py --img input.jpg --output output.jpg --noise_type brown --intensity 0.2 --blend_mode overlay
python noise_generator.py --img input.jpg --output output.jpg --noise_type white --intensity 0.1 --blend_mode add

Package Version Check:
numpy : 1.24.3
cv2 : 4.11.0
scipy : 1.13.1
skimage : 0.24.0
"""

import numpy as np
import cv2
import argparse
from scipy import fftpack
import os

def generate_white_noise(shape, intensity=0.1):
    """Generate white noise with flat power spectrum"""
    noise = np.random.normal(0, intensity, shape)
    return noise

def generate_pink_noise(shape, intensity=0.1):
    """Generate pink noise with 1/f power spectrum"""
    h, w = shape[:2]
    
    # Create frequency coordinates
    freq_y = np.fft.fftfreq(h)[:, np.newaxis]
    freq_x = np.fft.fftfreq(w)[np.newaxis, :]
    
    # Calculate distance from center in frequency domain
    freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)
    
    # Avoid division by zero
    freq_magnitude[0, 0] = 1
    
    # Create 1/f spectrum
    spectrum = 1.0 / freq_magnitude
    spectrum[0, 0] = 0  # Remove DC component
    
    # Generate random phases
    phases = np.random.uniform(0, 2*np.pi, (h, w))
    
    # Create complex spectrum
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # Generate noise for each channel
    if len(shape) == 3:
        noise = np.zeros(shape)
        for c in range(shape[2]):
            noise[:, :, c] = np.real(np.fft.ifft2(complex_spectrum))
    else:
        noise = np.real(np.fft.ifft2(complex_spectrum))
    
    # Normalize and scale by intensity
    noise = noise - np.mean(noise)
    noise = noise / np.std(noise) * intensity
    
    return noise

def generate_blue_noise(shape, intensity=0.1):
    """Generate blue noise with high-frequency emphasis"""
    h, w = shape[:2]
    
    # Create frequency coordinates
    freq_y = np.fft.fftfreq(h)[:, np.newaxis]
    freq_x = np.fft.fftfreq(w)[np.newaxis, :]
    
    # Calculate distance from center in frequency domain
    freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)
    
    # Create f spectrum (emphasizes high frequencies)
    spectrum = freq_magnitude
    spectrum[0, 0] = 0  # Remove DC component
    
    # Generate random phases
    phases = np.random.uniform(0, 2*np.pi, (h, w))
    
    # Create complex spectrum
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # Generate noise for each channel
    if len(shape) == 3:
        noise = np.zeros(shape)
        for c in range(shape[2]):
            noise[:, :, c] = np.real(np.fft.ifft2(complex_spectrum))
    else:
        noise = np.real(np.fft.ifft2(complex_spectrum))
    
    # Normalize and scale by intensity
    noise = noise - np.mean(noise)
    if np.std(noise) > 0:
        noise = noise / np.std(noise) * intensity
    
    return noise

def generate_brown_noise(shape, intensity=0.1):
    """Generate brown noise with 1/f² power spectrum"""
    h, w = shape[:2]
    
    # Create frequency coordinates
    freq_y = np.fft.fftfreq(h)[:, np.newaxis]
    freq_x = np.fft.fftfreq(w)[np.newaxis, :]
    
    # Calculate distance from center in frequency domain
    freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)
    
    # Avoid division by zero
    freq_magnitude[0, 0] = 1
    
    # Create 1/f² spectrum
    spectrum = 1.0 / (freq_magnitude**2)
    spectrum[0, 0] = 0  # Remove DC component
    
    # Generate random phases
    phases = np.random.uniform(0, 2*np.pi, (h, w))
    
    # Create complex spectrum
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # Generate noise for each channel
    if len(shape) == 3:
        noise = np.zeros(shape)
        for c in range(shape[2]):
            noise[:, :, c] = np.real(np.fft.ifft2(complex_spectrum))
    else:
        noise = np.real(np.fft.ifft2(complex_spectrum))
    
    # Normalize and scale by intensity
    noise = noise - np.mean(noise)
    noise = noise / np.std(noise) * intensity
    
    return noise

def apply_noise_to_image(image, noise, blend_mode='add'):
    """Apply noise to image using different blending modes"""
    image_float = image.astype(np.float32) / 255.0
    
    if blend_mode == 'add':
        result = image_float + noise
    elif blend_mode == 'multiply':
        result = image_float * (1 + noise)
    elif blend_mode == 'overlay':
        # Simple overlay implementation
        mask = image_float < 0.5
        result = np.where(mask, 
                         2 * image_float * (noise + 0.5),
                         1 - 2 * (1 - image_float) * (0.5 - noise))
    else:
        result = image_float + noise
    
    # Clamp values to [0, 1] and convert back to uint8
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

def add_frequency_noise(img_path, output_path, noise_type='white', intensity=0.1, blend_mode='add'):
    """
    Add frequency domain noise to an image
    
    Args:
        img_path (str): Path to input image
        output_path (str): Path to save output image
        noise_type (str): Type of noise ('white', 'pink', 'blue', 'brown')
        intensity (float): Noise intensity (0.0 to 1.0)
        blend_mode (str): Blending mode ('add', 'multiply', 'overlay')
    """
    # Read image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")
    
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate noise based on type
    noise_generators = {
        'white': generate_white_noise,
        'pink': generate_pink_noise,
        'blue': generate_blue_noise,
        'brown': generate_brown_noise
    }
    
    if noise_type not in noise_generators:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from: {list(noise_generators.keys())}")
    
    print(f"Generating {noise_type} noise with intensity {intensity}...")
    noise = noise_generators[noise_type](image.shape, intensity)
    
    # Apply noise to image
    print(f"Applying noise using {blend_mode} blend mode...")
    noisy_image = apply_noise_to_image(image, noise, blend_mode)
    
    # Convert back to BGR for OpenCV
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save image
    success = cv2.imwrite(output_path, noisy_image)
    if success:
        print(f"Noisy image saved to: {output_path}")
        print(f"Original size: {image.shape}")
        print(f"Output size: {noisy_image.shape}")
    else:
        raise RuntimeError(f"Failed to save image to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add frequency domain noise to images')
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--noise_type', choices=['white', 'pink', 'blue', 'brown'], 
                       default='white', help='Type of noise to generate')
    parser.add_argument('--intensity', type=float, default=0.1, 
                       help='Noise intensity (0.0 to 1.0, default: 0.1)')
    parser.add_argument('--blend_mode', choices=['add', 'multiply', 'overlay'], 
                       default='add', help='Blending mode for applying noise')
    
    args = parser.parse_args()
    
    # Validate intensity
    if not 0.0 <= args.intensity <= 1.0:
        print("Warning: Intensity should be between 0.0 and 1.0")
    
    try:
        add_frequency_noise(
            args.img, 
            args.output, 
            args.noise_type, 
            args.intensity, 
            args.blend_mode
        )
        print(f"\nSuccess! {args.noise_type.title()} noise applied successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())