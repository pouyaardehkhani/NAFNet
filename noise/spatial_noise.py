"""
Spatial Noise Pattern Generator for Images

This script adds various spatial noise patterns to images including:
- Checkerboard noise: Alternating pattern of noise blocks
- Stripe noise: Horizontal or vertical bands
- Ring artifacts: Circular patterns (common in CT/MRI)
- Moiré patterns: Interference between regular structures

Package Requirements:
numpy : 1.24.3
cv2 : 4.11.0
scipy : 1.13.1
skimage : 0.24.0

Usage Examples:
python spatial_noise.py --img input.jpg --output output.jpg --pattern checkerboard --intensity 0.3 --block_size 16
python spatial_noise.py --img input.jpg --output output.jpg --pattern stripe --intensity 0.4 --stripe_width 5 --direction horizontal
python spatial_noise.py --img input.jpg --output output.jpg --pattern ring --intensity 0.25 --ring_spacing 20 --center_x 256 --center_y 256
python spatial_noise.py --img input.jpg --output output.jpg --pattern moire --intensity 0.2 --freq1 0.1 --freq2 0.12 --angle1 0 --angle2 15
"""

import numpy as np
import cv2
import argparse
import sys
from scipy import ndimage
from skimage import util
import os

def add_checkerboard_noise(image, intensity=0.3, block_size=16):
    """
    Add checkerboard pattern noise to image
    
    Args:
        image: Input RGB image (H, W, 3)
        intensity: Noise intensity (0.0 to 1.0)
        block_size: Size of each checkerboard square
    
    Returns:
        Noisy image with same dimensions as input
    """
    h, w, c = image.shape
    
    # Create checkerboard pattern
    y_blocks = (h + block_size - 1) // block_size
    x_blocks = (w + block_size - 1) // block_size
    
    # Create basic checkerboard
    checker = np.zeros((y_blocks, x_blocks))
    checker[::2, ::2] = 1
    checker[1::2, 1::2] = 1
    
    # Resize to match image dimensions
    checker = np.repeat(checker, block_size, axis=0)[:h]
    checker = np.repeat(checker, block_size, axis=1)[:w]
    
    # Convert to noise pattern (-1 to 1)
    noise_pattern = (checker * 2 - 1) * intensity
    
    # Apply to all channels
    noisy_image = image.copy().astype(np.float32)
    for ch in range(c):
        noisy_image[:, :, ch] += noise_pattern * 255
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_stripe_noise(image, intensity=0.4, stripe_width=5, direction='horizontal'):
    """
    Add stripe pattern noise to image
    
    Args:
        image: Input RGB image (H, W, 3)
        intensity: Noise intensity (0.0 to 1.0)
        stripe_width: Width of each stripe in pixels
        direction: 'horizontal', 'vertical', or 'diagonal'
    
    Returns:
        Noisy image with same dimensions as input
    """
    h, w, c = image.shape
    
    if direction == 'horizontal':
        # Create horizontal stripes
        stripe_pattern = np.zeros((h, w))
        for i in range(0, h, stripe_width * 2):
            stripe_pattern[i:i+stripe_width, :] = 1
    
    elif direction == 'vertical':
        # Create vertical stripes
        stripe_pattern = np.zeros((h, w))
        for i in range(0, w, stripe_width * 2):
            stripe_pattern[:, i:i+stripe_width] = 1
    
    elif direction == 'diagonal':
        # Create diagonal stripes
        stripe_pattern = np.zeros((h, w))
        y, x = np.ogrid[:h, :w]
        diagonal_coord = (x + y) // stripe_width
        stripe_pattern = (diagonal_coord % 2).astype(float)
    
    else:
        raise ValueError("Direction must be 'horizontal', 'vertical', or 'diagonal'")
    
    # Convert to noise pattern (-1 to 1)
    noise_pattern = (stripe_pattern * 2 - 1) * intensity
    
    # Apply to all channels
    noisy_image = image.copy().astype(np.float32)
    for ch in range(c):
        noisy_image[:, :, ch] += noise_pattern * 255
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_ring_artifacts(image, intensity=0.25, ring_spacing=20, center_x=None, center_y=None):
    """
    Add ring artifact noise to image (common in CT/MRI)
    
    Args:
        image: Input RGB image (H, W, 3)
        intensity: Noise intensity (0.0 to 1.0)
        ring_spacing: Spacing between rings in pixels
        center_x: X coordinate of ring center (default: image center)
        center_y: Y coordinate of ring center (default: image center)
    
    Returns:
        Noisy image with same dimensions as input
    """
    h, w, c = image.shape
    
    # Default to image center
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create ring pattern
    ring_pattern = np.sin(2 * np.pi * distance / ring_spacing)
    
    # Apply intensity
    noise_pattern = ring_pattern * intensity
    
    # Apply to all channels
    noisy_image = image.copy().astype(np.float32)
    for ch in range(c):
        noisy_image[:, :, ch] += noise_pattern * 255
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_moire_patterns(image, intensity=0.2, freq1=0.1, freq2=0.12, angle1=0, angle2=15):
    """
    Add Moiré pattern noise (interference between regular structures)
    
    Args:
        image: Input RGB image (H, W, 3)
        intensity: Noise intensity (0.0 to 1.0)
        freq1: Frequency of first pattern
        freq2: Frequency of second pattern
        angle1: Angle of first pattern in degrees
        angle2: Angle of second pattern in degrees
    
    Returns:
        Noisy image with same dimensions as input
    """
    h, w, c = image.shape
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Convert angles to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    
    # Create first pattern
    x1 = x * np.cos(angle1_rad) + y * np.sin(angle1_rad)
    pattern1 = np.sin(2 * np.pi * freq1 * x1)
    
    # Create second pattern
    x2 = x * np.cos(angle2_rad) + y * np.sin(angle2_rad)
    pattern2 = np.sin(2 * np.pi * freq2 * x2)
    
    # Create Moiré interference
    moire_pattern = pattern1 * pattern2
    
    # Apply intensity
    noise_pattern = moire_pattern * intensity
    
    # Apply to all channels
    noisy_image = image.copy().astype(np.float32)
    for ch in range(c):
        noisy_image[:, :, ch] += noise_pattern * 255
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Add spatial noise patterns to images')
    
    # Required arguments
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--pattern', required=True, 
                       choices=['checkerboard', 'stripe', 'ring', 'moire'],
                       help='Type of spatial noise pattern')
    
    # Common parameters
    parser.add_argument('--intensity', type=float, default=0.3,
                       help='Noise intensity (0.0 to 1.0, default: 0.3)')
    
    # Checkerboard parameters
    parser.add_argument('--block_size', type=int, default=16,
                       help='Checkerboard block size (default: 16)')
    
    # Stripe parameters
    parser.add_argument('--stripe_width', type=int, default=5,
                       help='Stripe width in pixels (default: 5)')
    parser.add_argument('--direction', choices=['horizontal', 'vertical', 'diagonal'],
                       default='horizontal', help='Stripe direction (default: horizontal)')
    
    # Ring parameters
    parser.add_argument('--ring_spacing', type=int, default=20,
                       help='Ring spacing in pixels (default: 20)')
    parser.add_argument('--center_x', type=int, default=None,
                       help='X coordinate of ring center (default: image center)')
    parser.add_argument('--center_y', type=int, default=None,
                       help='Y coordinate of ring center (default: image center)')
    
    # Moiré parameters
    parser.add_argument('--freq1', type=float, default=0.1,
                       help='Frequency of first Moiré pattern (default: 0.1)')
    parser.add_argument('--freq2', type=float, default=0.12,
                       help='Frequency of second Moiré pattern (default: 0.12)')
    parser.add_argument('--angle1', type=float, default=0,
                       help='Angle of first Moiré pattern in degrees (default: 0)')
    parser.add_argument('--angle2', type=float, default=15,
                       help='Angle of second Moiré pattern in degrees (default: 15)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.img):
        print(f"Error: Input file '{args.img}' not found.")
        sys.exit(1)
    
    # Load image
    try:
        image = cv2.imread(args.img)
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {image.shape}")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Apply selected noise pattern
    try:
        if args.pattern == 'checkerboard':
            noisy_image = add_checkerboard_noise(image, args.intensity, args.block_size)
            print(f"Applied checkerboard noise: intensity={args.intensity}, block_size={args.block_size}")
        
        elif args.pattern == 'stripe':
            noisy_image = add_stripe_noise(image, args.intensity, args.stripe_width, args.direction)
            print(f"Applied stripe noise: intensity={args.intensity}, width={args.stripe_width}, direction={args.direction}")
        
        elif args.pattern == 'ring':
            noisy_image = add_ring_artifacts(image, args.intensity, args.ring_spacing, 
                                           args.center_x, args.center_y)
            print(f"Applied ring artifacts: intensity={args.intensity}, spacing={args.ring_spacing}")
        
        elif args.pattern == 'moire':
            noisy_image = add_moire_patterns(image, args.intensity, args.freq1, args.freq2, 
                                           args.angle1, args.angle2)
            print(f"Applied Moiré patterns: intensity={args.intensity}, freq1={args.freq1}, freq2={args.freq2}")
        
    except Exception as e:
        print(f"Error applying noise pattern: {e}")
        sys.exit(1)
    
    # Save result
    try:
        # Convert RGB back to BGR for OpenCV
        output_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save image
        success = cv2.imwrite(args.output, output_image)
        if not success:
            raise ValueError("Failed to save image")
        
        print(f"Saved noisy image to: {args.output}")
        print(f"Output image size: {noisy_image.shape}")
        
    except Exception as e:
        print(f"Error saving image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()