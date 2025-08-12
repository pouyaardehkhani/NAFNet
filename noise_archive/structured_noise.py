#!/usr/bin/env python3
"""
Structured Noise Generator
Adds periodic noise and banding effects to images.

Usage Examples:
    Basic usage:
    python structured_noise.py --img input.jpg --output noisy_output.jpg

    Apply only periodic noise:
    python structured_noise.py --img input.jpg --output periodic_noise.jpg --noise_type periodic --periodic_frequency 0.05 --periodic_amplitude 25

    Apply only horizontal banding:
    python structured_noise.py --img input.jpg --output banding.jpg --noise_type banding --band_width 5 --band_intensity 40 --banding_direction horizontal

    Custom periodic noise with both directions:
    python structured_noise.py --img input.jpg --output complex_noise.jpg --periodic_frequency 0.15 --periodic_amplitude 30 --periodic_direction both --periodic_phase_x 1.57

    Complex structured noise with custom parameters:
    python structured_noise.py --img input.jpg --output structured.jpg --noise_type both --periodic_frequency 0.08 --periodic_amplitude 15 --band_width 8 --band_intensity 25 --banding_direction vertical
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional


def add_periodic_noise(image: np.ndarray, 
                      frequency: float = 0.1, 
                      amplitude: float = 20.0, 
                      phase_x: float = 0.0, 
                      phase_y: float = 0.0,
                      direction: str = 'both') -> np.ndarray:
    """
    Add periodic noise to an image (simulates electrical interference).
    
    Args:
        image: Input image as numpy array
        frequency: Noise frequency (0.01-1.0, higher = more oscillations)
        amplitude: Noise strength (0-100, higher = more visible)
        phase_x: Phase shift in x direction (0-2π)
        phase_y: Phase shift in y direction (0-2π)
        direction: 'horizontal', 'vertical', or 'both'
    
    Returns:
        Image with periodic noise added
    """
    h, w = image.shape[:2]
    
    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Generate periodic patterns
    noise = np.zeros((h, w))
    
    if direction in ['horizontal', 'both']:
        # Horizontal periodic pattern
        noise += np.sin(2 * np.pi * frequency * X + phase_x)
    
    if direction in ['vertical', 'both']:
        # Vertical periodic pattern
        noise += np.sin(2 * np.pi * frequency * Y + phase_y)
    
    # Scale noise by amplitude
    noise = (noise * amplitude).astype(np.float32)
    
    # Add noise to image
    if len(image.shape) == 3:
        # Color image - add noise to all channels
        noisy_image = image.astype(np.float32)
        for i in range(image.shape[2]):
            noisy_image[:, :, i] += noise
    else:
        # Grayscale image
        noisy_image = image.astype(np.float32) + noise
    
    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)


def add_banding_noise(image: np.ndarray,
                     band_width: int = 10,
                     band_intensity: float = 30.0,
                     direction: str = 'horizontal',
                     spacing: int = 1,
                     random_offset: bool = True) -> np.ndarray:
    """
    Add banding noise to an image (horizontal/vertical stripes).
    
    Args:
        image: Input image as numpy array
        band_width: Width of each band in pixels (1-50)
        band_intensity: Intensity variation of bands (-100 to 100)
        direction: 'horizontal' or 'vertical'
        spacing: Spacing between bands (1-10)
        random_offset: Whether to add random variations to band positions
    
    Returns:
        Image with banding noise added
    """
    h, w = image.shape[:2]
    
    # Create banding pattern
    if direction == 'horizontal':
        # Horizontal bands
        pattern = np.zeros(h)
        for i in range(0, h, band_width + spacing):
            end_idx = min(i + band_width, h)
            intensity = band_intensity
            if random_offset:
                intensity += np.random.uniform(-band_intensity * 0.3, band_intensity * 0.3)
            pattern[i:end_idx] = intensity
        
        # Expand pattern to full image dimensions
        noise = np.tile(pattern.reshape(-1, 1), (1, w))
        
    else:  # vertical
        # Vertical bands
        pattern = np.zeros(w)
        for i in range(0, w, band_width + spacing):
            end_idx = min(i + band_width, w)
            intensity = band_intensity
            if random_offset:
                intensity += np.random.uniform(-band_intensity * 0.3, band_intensity * 0.3)
            pattern[i:end_idx] = intensity
        
        # Expand pattern to full image dimensions
        noise = np.tile(pattern.reshape(1, -1), (h, 1))
    
    # Add noise to image
    if len(image.shape) == 3:
        # Color image - add noise to all channels
        noisy_image = image.astype(np.float32)
        for i in range(image.shape[2]):
            noisy_image[:, :, i] += noise
    else:
        # Grayscale image
        noisy_image = image.astype(np.float32) + noise
    
    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)


def apply_structured_noise(image_path: str, 
                          output_path: str,
                          noise_type: str = 'both',
                          # Periodic noise parameters
                          periodic_frequency: float = 0.1,
                          periodic_amplitude: float = 20.0,
                          periodic_phase_x: float = 0.0,
                          periodic_phase_y: float = 0.0,
                          periodic_direction: str = 'both',
                          # Banding noise parameters
                          band_width: int = 10,
                          band_intensity: float = 30.0,
                          banding_direction: str = 'horizontal',
                          band_spacing: int = 1,
                          random_band_offset: bool = True) -> bool:
    """
    Apply structured noise to an image and save the result.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        noise_type: 'periodic', 'banding', or 'both'
        (other parameters as documented in individual functions)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
        
        # Convert BGR to RGB for processing
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Processing image: {image.shape}")
        
        # Apply noise based on type
        if noise_type in ['periodic', 'both']:
            image = add_periodic_noise(
                image, 
                frequency=periodic_frequency,
                amplitude=periodic_amplitude,
                phase_x=periodic_phase_x,
                phase_y=periodic_phase_y,
                direction=periodic_direction
            )
            print(f"Applied periodic noise (freq={periodic_frequency}, amp={periodic_amplitude})")
        
        if noise_type in ['banding', 'both']:
            image = add_banding_noise(
                image,
                band_width=band_width,
                band_intensity=band_intensity,
                direction=banding_direction,
                spacing=band_spacing,
                random_offset=random_band_offset
            )
            print(f"Applied banding noise (width={band_width}, intensity={band_intensity})")
        
        # Convert back to BGR for saving
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save result
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Saved noisy image to: {output_path}")
            return True
        else:
            print(f"Error: Could not save image to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Add structured noise to images')
    
    # Required arguments
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    
    # Noise type selection
    parser.add_argument('--noise_type', default='both', 
                       choices=['periodic', 'banding', 'both'],
                       help='Type of noise to apply (default: both)')
    
    # Periodic noise parameters
    parser.add_argument('--periodic_frequency', type=float, default=0.1,
                       help='Periodic noise frequency (0.01-1.0, default: 0.1)')
    parser.add_argument('--periodic_amplitude', type=float, default=20.0,
                       help='Periodic noise amplitude (0-100, default: 20)')
    parser.add_argument('--periodic_phase_x', type=float, default=0.0,
                       help='Phase shift in X direction (default: 0)')
    parser.add_argument('--periodic_phase_y', type=float, default=0.0,
                       help='Phase shift in Y direction (default: 0)')
    parser.add_argument('--periodic_direction', default='both',
                       choices=['horizontal', 'vertical', 'both'],
                       help='Direction of periodic noise (default: both)')
    
    # Banding noise parameters
    parser.add_argument('--band_width', type=int, default=10,
                       help='Width of bands in pixels (1-50, default: 10)')
    parser.add_argument('--band_intensity', type=float, default=30.0,
                       help='Band intensity variation (-100 to 100, default: 30)')
    parser.add_argument('--banding_direction', default='horizontal',
                       choices=['horizontal', 'vertical'],
                       help='Direction of banding (default: horizontal)')
    parser.add_argument('--band_spacing', type=int, default=1,
                       help='Spacing between bands (1-10, default: 1)')
    parser.add_argument('--no_random_band_offset', action='store_true',
                       help='Disable random band position variations')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.img):
        print(f"Error: Input image '{args.img}' does not exist")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply structured noise
    success = apply_structured_noise(
        image_path=args.img,
        output_path=args.output,
        noise_type=args.noise_type,
        periodic_frequency=args.periodic_frequency,
        periodic_amplitude=args.periodic_amplitude,
        periodic_phase_x=args.periodic_phase_x,
        periodic_phase_y=args.periodic_phase_y,
        periodic_direction=args.periodic_direction,
        band_width=args.band_width,
        band_intensity=args.band_intensity,
        banding_direction=args.banding_direction,
        band_spacing=args.band_spacing,
        random_band_offset=not args.no_random_band_offset
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())