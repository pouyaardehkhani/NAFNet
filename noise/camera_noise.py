"""
Example usage:
python camera_noise.py --img input.jpg --output output.jpg --noise_type iso --iso_level 800 --intensity 0.3
python camera_noise.py --img input.jpg --output output.jpg --noise_type chromatic --fringe_strength 0.2 --blur_radius 1.5
python camera_noise.py --img input.jpg --output output.jpg --noise_type iso --iso_level 1600 --dark_noise 0.1 --read_noise 0.05
python camera_noise.py --img input.jpg --output output.jpg --noise_type chromatic --fringe_strength 0.3 --color_shift 2.0

Package Version Check:
numpy : 1.24.3
cv2 : 4.11.0
scipy : 1.13.1
skimage : 0.24.0
"""

import numpy as np
import cv2
import argparse
from scipy import ndimage
import os

def generate_iso_sensor_noise(image, iso_level=800, dark_noise=0.05, read_noise=0.02, shot_noise_factor=1.0):
    """
    Generate realistic ISO/Sensor noise including:
    - Shot noise (Poisson noise from photon counting)
    - Dark current noise (thermal noise)
    - Read noise (electronic noise)
    
    Args:
        image: Input image array (0-255)
        iso_level: ISO sensitivity level (100, 200, 400, 800, 1600, etc.)
        dark_noise: Dark current noise intensity
        read_noise: Read noise intensity
        shot_noise_factor: Shot noise scaling factor
    """
    # Convert to float and normalize
    image_float = image.astype(np.float32) / 255.0
    
    # Calculate noise scaling based on ISO
    iso_factor = np.sqrt(iso_level / 100.0)  # ISO 100 as baseline
    
    # Shot noise (Poisson noise - signal dependent)
    # Higher for brighter pixels, follows Poisson distribution
    shot_noise = np.random.poisson(image_float * 255 * shot_noise_factor * iso_factor) / 255.0 - image_float
    shot_noise = shot_noise * 0.1  # Scale down the effect
    
    # Dark current noise (thermal noise - independent of signal)
    # Gaussian noise that's more prominent in shadows
    dark_current = np.random.normal(0, dark_noise * iso_factor, image.shape)
    
    # Read noise (electronic noise from sensor readout)
    # Uniform across the image, Gaussian distribution
    readout_noise = np.random.normal(0, read_noise * iso_factor, image.shape)
    
    # Pattern noise (fixed pattern noise from sensor manufacturing)
    # Create subtle grid pattern
    h, w = image.shape[:2]
    pattern_x = np.sin(np.linspace(0, 2*np.pi*10, w)) * 0.002 * iso_factor
    pattern_y = np.sin(np.linspace(0, 2*np.pi*8, h)) * 0.002 * iso_factor
    pattern_noise = np.outer(pattern_y, pattern_x)
    if len(image.shape) == 3:
        pattern_noise = np.stack([pattern_noise] * image.shape[2], axis=2)
    
    # Combine all noise types
    total_noise = shot_noise + dark_current + readout_noise + pattern_noise
    
    # Apply noise
    noisy_image = image_float + total_noise
    
    # Clamp and convert back
    noisy_image = np.clip(noisy_image, 0, 1)
    return (noisy_image * 255).astype(np.uint8)

def generate_chromatic_noise(image, fringe_strength=0.2, color_shift=1.0, blur_radius=1.0):
    """
    Generate chromatic aberration noise including:
    - Lateral chromatic aberration (color fringing)
    - Longitudinal chromatic aberration (color blur differences)
    - Purple fringing effects
    
    Args:
        image: Input image array (0-255)
        fringe_strength: Strength of chromatic fringing (0.0 to 1.0)
        color_shift: Amount of color channel shifting in pixels
        blur_radius: Differential blur between color channels
    """
    if len(image.shape) != 3:
        # Convert grayscale to RGB for chromatic effects
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    image_float = image.astype(np.float32) / 255.0
    h, w, c = image_float.shape
    
    # Create coordinate grids for distortion
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Distance from center (for radial effects)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_r = np.sqrt(center_x**2 + center_y**2)
    r_norm = r / max_r
    
    # Split into color channels
    red = image_float[:, :, 0]
    green = image_float[:, :, 1] 
    blue = image_float[:, :, 2]
    
    # Lateral chromatic aberration - shift different channels
    # Red channel shifted outward, blue inward
    red_shift = color_shift * r_norm * 0.5
    blue_shift = -color_shift * r_norm * 0.5
    
    # Apply shifts using scipy's shift function
    red_shifted = ndimage.shift(red, [0, red_shift.mean()], order=1, mode='reflect')
    blue_shifted = ndimage.shift(blue, [0, blue_shift.mean()], order=1, mode='reflect')
    
    # Longitudinal chromatic aberration - different blur for each channel
    red_blurred = ndimage.gaussian_filter(red_shifted, blur_radius * 1.2)
    green_blurred = ndimage.gaussian_filter(green, blur_radius)
    blue_blurred = ndimage.gaussian_filter(blue_shifted, blur_radius * 0.8)
    
    # Purple fringing (common in high contrast edges)
    # Find high contrast edges
    gray = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150) / 255.0
    edges = ndimage.gaussian_filter(edges, 2.0)  # Blur edges for smooth transition
    
    # Create purple fringing effect
    purple_r = 0.5 + 0.3 * fringe_strength
    purple_g = 0.2 * fringe_strength  
    purple_b = 0.8 + 0.2 * fringe_strength
    
    # Apply purple fringing to edges
    fringe_mask = edges * fringe_strength
    red_blurred = red_blurred + fringe_mask * purple_r * 0.1
    green_blurred = green_blurred + fringe_mask * purple_g * 0.1
    blue_blurred = blue_blurred + fringe_mask * purple_b * 0.1
    
    # Add some color noise (chroma noise)
    chroma_noise_strength = fringe_strength * 0.05
    red_blurred += np.random.normal(0, chroma_noise_strength, red_blurred.shape)
    green_blurred += np.random.normal(0, chroma_noise_strength, green_blurred.shape)
    blue_blurred += np.random.normal(0, chroma_noise_strength, blue_blurred.shape)
    
    # Recombine channels
    result = np.stack([red_blurred, green_blurred, blue_blurred], axis=2)
    
    # Clamp and convert back
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

def add_camera_noise(img_path, output_path, noise_type='iso', **kwargs):
    """
    Add camera-related noise to an image
    
    Args:
        img_path (str): Path to input image
        output_path (str): Path to save output image
        noise_type (str): Type of noise ('iso', 'chromatic')
        **kwargs: Additional parameters for specific noise types
    """
    # Read image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")
    
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape
    
    # Apply noise based on type
    if noise_type == 'iso':
        print(f"Generating ISO/Sensor noise...")
        iso_level = kwargs.get('iso_level', 800)
        dark_noise = kwargs.get('dark_noise', 0.05)
        read_noise = kwargs.get('read_noise', 0.02)
        shot_noise_factor = kwargs.get('shot_noise_factor', 1.0)
        
        print(f"  ISO Level: {iso_level}")
        print(f"  Dark Noise: {dark_noise}")
        print(f"  Read Noise: {read_noise}")
        
        noisy_image = generate_iso_sensor_noise(
            image, iso_level, dark_noise, read_noise, shot_noise_factor
        )
        
    elif noise_type == 'chromatic':
        print(f"Generating Chromatic Aberration noise...")
        fringe_strength = kwargs.get('fringe_strength', 0.2)
        color_shift = kwargs.get('color_shift', 1.0)
        blur_radius = kwargs.get('blur_radius', 1.0)
        
        print(f"  Fringe Strength: {fringe_strength}")
        print(f"  Color Shift: {color_shift}")
        print(f"  Blur Radius: {blur_radius}")
        
        noisy_image = generate_chromatic_noise(
            image, fringe_strength, color_shift, blur_radius
        )
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from: ['iso', 'chromatic']")
    
    # Ensure output has same dimensions as input
    assert noisy_image.shape == original_shape, f"Size mismatch: {noisy_image.shape} != {original_shape}"
    
    # Convert back to BGR for OpenCV
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save image
    success = cv2.imwrite(output_path, noisy_image)
    if success:
        print(f"Processed image saved to: {output_path}")
        print(f"Original size: {original_shape}")
        print(f"Output size: {noisy_image.shape}")
    else:
        raise RuntimeError(f"Failed to save image to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add camera-related noise to images')
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--noise_type', choices=['iso', 'chromatic'], 
                       default='iso', help='Type of noise to generate')
    
    # ISO/Sensor noise parameters
    parser.add_argument('--iso_level', type=int, default=800, 
                       help='ISO sensitivity level (100, 200, 400, 800, 1600, etc.)')
    parser.add_argument('--dark_noise', type=float, default=0.05, 
                       help='Dark current noise intensity (0.0 to 0.2)')
    parser.add_argument('--read_noise', type=float, default=0.02, 
                       help='Read noise intensity (0.0 to 0.1)')
    parser.add_argument('--shot_noise_factor', type=float, default=1.0, 
                       help='Shot noise scaling factor (0.5 to 2.0)')
    
    # Chromatic noise parameters
    parser.add_argument('--fringe_strength', type=float, default=0.2, 
                       help='Chromatic fringing strength (0.0 to 1.0)')
    parser.add_argument('--color_shift', type=float, default=1.0, 
                       help='Color channel shift amount in pixels (0.0 to 5.0)')
    parser.add_argument('--blur_radius', type=float, default=1.0, 
                       help='Differential blur radius (0.5 to 3.0)')
    
    args = parser.parse_args()
    
    # Prepare kwargs based on noise type
    if args.noise_type == 'iso':
        kwargs = {
            'iso_level': args.iso_level,
            'dark_noise': args.dark_noise,
            'read_noise': args.read_noise,
            'shot_noise_factor': args.shot_noise_factor
        }
    else:  # chromatic
        kwargs = {
            'fringe_strength': args.fringe_strength,
            'color_shift': args.color_shift,
            'blur_radius': args.blur_radius
        }
    
    try:
        add_camera_noise(args.img, args.output, args.noise_type, **kwargs)
        print(f"\nSuccess! {args.noise_type.title()} noise applied successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())