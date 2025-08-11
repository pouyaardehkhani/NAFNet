"""
Motion Noise Effects Script
===========================

This script applies motion-related noise effects to images:
- Motion blur noise: Simulates camera/subject movement blur
- Vibration noise: Adds high-frequency shake artifacts

Package Requirements:
- numpy : 1.24.3
- cv2 : 4.11.0
- scipy : 1.13.1
- skimage : 0.24.0

Usage Examples:
python motion_noise.py --img input.jpg --output output.jpg --effect motion_blur --length 15 --angle 45
python motion_noise.py --img input.jpg --output output.jpg --effect vibration --intensity 2.0 --frequency 50
python motion_noise.py --img input.jpg --output output.jpg --effect both --length 10 --angle 30 --intensity 1.5 --frequency 30
"""

import argparse
import cv2
import numpy as np
from scipy import ndimage
from skimage import util
import os


def create_motion_kernel(length, angle):
    """
    Create a motion blur kernel for simulating camera/subject movement.
    
    Args:
        length (int): Length of the motion blur in pixels
        angle (float): Angle of motion in degrees (0-360)
    
    Returns:
        numpy.ndarray: Motion blur kernel
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Create kernel size (should be odd)
    kernel_size = int(length * 2 + 1)
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Calculate the line coordinates
    center = kernel_size // 2
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Draw the line representing motion direction
    for i in range(-length, length + 1):
        x = int(center + i * cos_a)
        y = int(center + i * sin_a)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
    
    return kernel


def apply_motion_blur(image, length=10, angle=0):
    """
    Apply motion blur noise to simulate camera/subject movement.
    
    Args:
        image (numpy.ndarray): Input RGB image
        length (int): Motion blur length in pixels (default: 10)
        angle (float): Motion direction angle in degrees (default: 0)
    
    Returns:
        numpy.ndarray: Image with motion blur applied
    """
    # Create motion blur kernel
    kernel = create_motion_kernel(length, angle)
    
    # Apply blur to each channel separately for RGB images
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for channel in range(image.shape[2]):
            blurred[:, :, channel] = cv2.filter2D(image[:, :, channel], -1, kernel)
    else:
        blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred.astype(np.uint8)


def generate_vibration_pattern(shape, intensity=1.0, frequency=30):
    """
    Generate vibration noise pattern for high-frequency shake artifacts.
    
    Args:
        shape (tuple): Image shape (height, width) or (height, width, channels)
        intensity (float): Vibration intensity multiplier (default: 1.0)
        frequency (int): Vibration frequency parameter (default: 30)
    
    Returns:
        numpy.ndarray: Vibration displacement maps (dy, dx)
    """
    h, w = shape[:2]
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Generate high-frequency vibration patterns
    # Use multiple sine waves with different frequencies and phases
    vibration_y = np.zeros((h, w))
    vibration_x = np.zeros((h, w))
    
    for freq_mult in [1, 1.5, 2, 2.5]:
        phase_y = np.random.uniform(0, 2*np.pi)
        phase_x = np.random.uniform(0, 2*np.pi)
        
        vibration_y += np.sin(frequency * freq_mult * x / w * 2 * np.pi + phase_y)
        vibration_x += np.cos(frequency * freq_mult * y / h * 2 * np.pi + phase_x)
    
    # Normalize and scale by intensity
    vibration_y = intensity * vibration_y / 4.0
    vibration_x = intensity * vibration_x / 4.0
    
    return vibration_y, vibration_x


def apply_vibration_noise(image, intensity=1.0, frequency=30, random_seed=None):
    """
    Apply vibration noise to simulate high-frequency shake artifacts.
    
    Args:
        image (numpy.ndarray): Input RGB image
        intensity (float): Vibration intensity (default: 1.0)
        frequency (int): Vibration frequency parameter (default: 30)
        random_seed (int): Random seed for reproducible results (default: None)
    
    Returns:
        numpy.ndarray: Image with vibration noise applied
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    h, w = image.shape[:2]
    
    # Generate vibration displacement patterns
    dy, dx = generate_vibration_pattern(image.shape, intensity, frequency)
    
    # Create coordinate grids for remapping
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Apply vibration displacement
    map_y = (y_coords + dy).astype(np.float32)
    map_x = (x_coords + dx).astype(np.float32)
    
    # Remap the image using the vibration pattern
    vibrated = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return vibrated


def load_image(image_path):
    """
    Load image from path and convert to RGB.
    
    Args:
        image_path (str): Path to input image
    
    Returns:
        numpy.ndarray: RGB image array
    """
    # Load image using OpenCV (BGR format)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image, output_path):
    """
    Save RGB image to file.
    
    Args:
        image (numpy.ndarray): RGB image array
        output_path (str): Path to save the image
    """
    # Convert RGB to BGR for OpenCV saving
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, image_bgr)


def main():
    parser = argparse.ArgumentParser(description='Apply motion-related noise effects to images')
    
    # Required arguments
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    
    # Effect selection
    parser.add_argument('--effect', choices=['motion_blur', 'vibration', 'both'], 
                       default='motion_blur', help='Type of noise effect to apply')
    
    # Motion blur parameters
    parser.add_argument('--length', type=int, default=10, 
                       help='Motion blur length in pixels (default: 10)')
    parser.add_argument('--angle', type=float, default=0, 
                       help='Motion blur angle in degrees (default: 0)')
    
    # Vibration noise parameters
    parser.add_argument('--intensity', type=float, default=1.0, 
                       help='Vibration intensity multiplier (default: 1.0)')
    parser.add_argument('--frequency', type=int, default=30, 
                       help='Vibration frequency parameter (default: 30)')
    
    # Optional parameters
    parser.add_argument('--random_seed', type=int, default=None, 
                       help='Random seed for reproducible vibration (default: None)')
    
    args = parser.parse_args()
    
    try:
        # Load input image
        print(f"Loading image from: {args.img}")
        image = load_image(args.img)
        original_shape = image.shape
        print(f"Original image shape: {original_shape}")
        
        # Apply selected effects
        if args.effect == 'motion_blur':
            print(f"Applying motion blur (length={args.length}, angle={args.angle})")
            result = apply_motion_blur(image, length=args.length, angle=args.angle)
            
        elif args.effect == 'vibration':
            print(f"Applying vibration noise (intensity={args.intensity}, frequency={args.frequency})")
            result = apply_vibration_noise(image, intensity=args.intensity, 
                                         frequency=args.frequency, random_seed=args.random_seed)
            
        elif args.effect == 'both':
            print(f"Applying motion blur (length={args.length}, angle={args.angle})")
            result = apply_motion_blur(image, length=args.length, angle=args.angle)
            print(f"Applying vibration noise (intensity={args.intensity}, frequency={args.frequency})")
            result = apply_vibration_noise(result, intensity=args.intensity, 
                                         frequency=args.frequency, random_seed=args.random_seed)
        
        # Verify output shape matches input
        assert result.shape == original_shape, f"Output shape {result.shape} doesn't match input {original_shape}"
        
        # Save result
        print(f"Saving result to: {args.output}")
        save_image(result, args.output)
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())