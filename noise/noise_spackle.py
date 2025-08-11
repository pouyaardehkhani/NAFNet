#!/usr/bin/env python3
"""
Speckle Noise Image Processing Script

This script adds speckle (multiplicative) noise to images.
Speckle noise affects signal intensity and is commonly found in ultrasound and radar imaging.

Usage:
    python noise_spackle.py --img input_image.jpg --output output_image.jpg --variance 0.1
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path


def add_speckle_noise_opencv(image_path, output_path, variance=0.1, seed=None):
    """
    Add speckle noise to an image using OpenCV and NumPy.
    
    Speckle noise is multiplicative noise: I_noisy = I_original * (1 + noise)
    where noise follows a normal distribution.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save the noisy image
        variance (float): Variance of the multiplicative noise (default: 0.1)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
            
        # Convert to float for processing
        image_float = image.astype(np.float64) / 255.0
        
        # Generate multiplicative noise
        # Speckle noise: I_noisy = I_original * (1 + noise)
        # where noise ~ N(0, variance)
        noise = np.random.normal(0, np.sqrt(variance), image_float.shape)
        
        # Apply multiplicative noise
        noisy_image = image_float * (1 + noise)
        
        # Clip values to valid range [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        
        # Convert back to uint8
        noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        success = cv2.imwrite(output_path, noisy_image_uint8)
        
        if success:
            print(f"Speckle noise added successfully. Image saved to: {output_path}")
            print(f"Noise variance: {variance}")
            return True
        else:
            print(f"Error: Could not save image to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False


def add_speckle_noise_advanced(image_path, output_path, variance=0.1, distribution='normal', 
                             intensity_scale=1.0, seed=None):
    """
    Advanced speckle noise function with additional parameters.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save the noisy image
        variance (float): Variance of the multiplicative noise (default: 0.1)
        distribution (str): Noise distribution type ('normal', 'uniform', 'gamma')
        intensity_scale (float): Scale factor for noise intensity (default: 1.0)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
            
        # Convert to float for processing
        image_float = image.astype(np.float64) / 255.0
        
        # Generate noise based on distribution type
        if distribution.lower() == 'normal':
            noise = np.random.normal(0, np.sqrt(variance), image_float.shape)
        elif distribution.lower() == 'uniform':
            # For uniform distribution, map variance to appropriate range
            range_val = np.sqrt(3 * variance)  # For uniform dist: var = (b-a)²/12
            noise = np.random.uniform(-range_val, range_val, image_float.shape)
        elif distribution.lower() == 'gamma':
            # Gamma distribution for speckle (more realistic for ultrasound)
            # Adjust shape and scale to achieve desired variance
            shape = 1 / variance
            scale = variance
            noise = np.random.gamma(shape, scale, image_float.shape) - shape * scale
        else:
            print(f"Warning: Unknown distribution '{distribution}'. Using normal distribution.")
            noise = np.random.normal(0, np.sqrt(variance), image_float.shape)
        
        # Scale the noise intensity
        noise *= intensity_scale
        
        # Apply multiplicative noise
        noisy_image = image_float * (1 + noise)
        
        # Clip values to valid range [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        
        # Convert back to uint8
        noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        success = cv2.imwrite(output_path, noisy_image_uint8)
        
        if success:
            print(f"Advanced speckle noise added successfully. Image saved to: {output_path}")
            print(f"Noise variance: {variance}, Distribution: {distribution}, "
                  f"Intensity scale: {intensity_scale}")
            return True
        else:
            print(f"Error: Could not save image to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False


def get_image_info(image_path):
    """Get basic information about the image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width, channels = image.shape
        return {
            'height': height,
            'width': width,
            'channels': channels,
            'size': (width, height)
        }
    except Exception as e:
        print(f"Error reading image info: {str(e)}")
        return None


def validate_paths(image_path, output_path):
    """Validate input and output paths."""
    # Check if input image exists
    if not os.path.exists(image_path):
        print(f"Error: Input image '{image_path}' does not exist.")
        return False
    
    # Check if input is a file
    if not os.path.isfile(image_path):
        print(f"Error: '{image_path}' is not a file.")
        return False
    
    # Check output directory
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {str(e)}")
            return False
    
    return True


def main():
    """Main function to handle command line arguments and execute noise addition."""
    parser = argparse.ArgumentParser(
        description="Add speckle noise to images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python speckle_noise.py --img input.jpg --output output.jpg
    python speckle_noise.py --img input.jpg --output output.jpg --variance 0.2
    python speckle_noise.py --img input.jpg --output output.jpg --variance 0.1 --distribution gamma
    python speckle_noise.py --img input.jpg --output output.jpg --advanced --intensity-scale 1.5
        """
    )
    
    # Required arguments
    parser.add_argument('--img', required=True, 
                       help='Path to input image')
    parser.add_argument('--output', required=False,
                       help='Path to output image (default: adds _speckle to input filename)')
    
    # Noise parameters
    parser.add_argument('--variance', type=float, default=0.1,
                       help='Variance of speckle noise (default: 0.1)')
    
    # Advanced options
    parser.add_argument('--advanced', action='store_true',
                       help='Use advanced speckle noise function')
    parser.add_argument('--distribution', choices=['normal', 'uniform', 'gamma'], 
                       default='normal',
                       help='Noise distribution type (default: normal)')
    parser.add_argument('--intensity-scale', type=float, default=1.0,
                       help='Scale factor for noise intensity (default: 1.0)')
    
    # Utility arguments
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--info', action='store_true',
                       help='Display image information')
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.img)
        args.output = str(input_path.parent / f"{input_path.stem}_speckle{input_path.suffix}")
    
    # Validate paths
    if not validate_paths(args.img, args.output):
        sys.exit(1)
    
    # Display image info if requested
    if args.info:
        info = get_image_info(args.img)
        if info:
            print(f"Image Information:")
            print(f"  Size: {info['width']} x {info['height']}")
            print(f"  Channels: {info['channels']}")
            print()
    
    # Validate parameters
    if args.variance < 0:
        print("Error: Variance must be non-negative")
        sys.exit(1)
    
    if args.intensity_scale <= 0:
        print("Error: Intensity scale must be positive")
        sys.exit(1)
    
    # Process the image
    print(f"Processing image: {args.img}")
    print(f"Output path: {args.output}")
    
    if args.advanced:
        success = add_speckle_noise_advanced(
            args.img, 
            args.output, 
            variance=args.variance,
            distribution=args.distribution,
            intensity_scale=args.intensity_scale,
            seed=args.seed
        )
    else:
        success = add_speckle_noise_opencv(
            args.img, 
            args.output, 
            variance=args.variance,
            seed=args.seed
        )
    
    if success:
        print("Processing completed successfully!")
        
        # Display final image info
        info = get_image_info(args.output)
        if info:
            print(f"Output image size: {info['width']} x {info['height']}")
    else:
        print("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()