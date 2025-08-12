#!/usr/bin/env python3
"""
Impulse Noise Generator for Images
Adds shot noise and quantization noise to input images.

Examples:
  python impulse_noise.py --img input.jpg output_shot.jpg --type shot --shot_intensity 0.1
  python impulse_noise.py --img input.jpg output_quant.jpg --type quantization --quant_bits 4
  python impulse_noise.py --img input.jpg output_both.jpg --type both --shot_intensity 0.05 --quant_bits 6
  python impulse_noise.py --img input.jpg output.jpg --seed 42
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys


def add_shot_noise(image, intensity=0.05, seed=None):
    """
    Add shot noise (Poisson noise) to an image.
    
    Args:
        image (numpy.ndarray): Input image (RGB format)
        intensity (float): Noise intensity factor (0.0 to 1.0)
        seed (int): Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Image with shot noise added
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Normalize image to [0, 1] range
    normalized_img = image.astype(np.float64) / 255.0
    
    # Scale the image by intensity factor to control noise level
    scaled_img = normalized_img * intensity * 255.0
    
    # Apply Poisson noise (shot noise is modeled as Poisson process)
    # Poisson noise has variance equal to the signal intensity
    noisy_img = np.random.poisson(scaled_img).astype(np.float64)
    
    # Scale back and add to original
    noisy_img = normalized_img * 255.0 + (noisy_img - scaled_img) * (1.0 / intensity)
    
    # Clip values to valid range and convert back to uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img


def add_quantization_noise(image, bits=6, seed=None):
    """
    Add quantization noise to an image by reducing bit depth.
    
    Args:
        image (numpy.ndarray): Input image (RGB format)
        bits (int): Number of bits for quantization (1-8)
        seed (int): Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Image with quantization noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure bits is within valid range
    bits = max(1, min(8, bits))
    
    # Calculate quantization levels
    levels = 2 ** bits
    
    # Quantize the image
    # Scale to [0, levels-1], round, then scale back to [0, 255]
    quantized = np.round(image.astype(np.float64) / 255.0 * (levels - 1))
    quantized = (quantized / (levels - 1) * 255.0).astype(np.uint8)
    
    return quantized


def add_impulse_noise(image_path, output_path, noise_type='both', 
                     shot_intensity=0.05, quant_bits=6, seed=None):
    """
    Add impulse noise to an image and save the result.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        noise_type (str): Type of noise ('shot', 'quantization', 'both')
        shot_intensity (float): Shot noise intensity
        quant_bits (int): Quantization bits
        seed (int): Random seed for reproducibility
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply noise based on type
    if noise_type == 'shot':
        noisy_image = add_shot_noise(image_rgb, shot_intensity, seed)
    elif noise_type == 'quantization':
        noisy_image = add_quantization_noise(image_rgb, quant_bits, seed)
    elif noise_type == 'both':
        # Apply both types of noise sequentially
        temp_image = add_shot_noise(image_rgb, shot_intensity, seed)
        noisy_image = add_quantization_noise(temp_image, quant_bits, seed)
    else:
        raise ValueError("noise_type must be 'shot', 'quantization', or 'both'")
    
    # Convert back to BGR for saving
    noisy_image_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the noisy image (same size as original)
    success = cv2.imwrite(str(output_path), noisy_image_bgr)
    if not success:
        raise ValueError(f"Could not save image to {output_path}")
    
    print(f"Successfully added {noise_type} noise to image")
    print(f"Original size: {image_rgb.shape}")
    print(f"Output size: {noisy_image.shape}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add impulse noise (shot noise and/or quantization noise) to images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python impulse_noise.py --img input.jpg output_shot.jpg --type shot --shot_intensity 0.1
  python impulse_noise.py --img input.jpg output_quant.jpg --type quantization --quant_bits 4
  python impulse_noise.py --img input.jpg output_both.jpg --type both --shot_intensity 0.05 --quant_bits 6
        """
    )
    
    parser.add_argument('--img', required=True, 
                       help='Input image path')
    parser.add_argument('output_path', 
                       help='Output path to save the noisy image')
    
    # Noise type selection
    parser.add_argument('--type', choices=['shot', 'quantization', 'both'], 
                       default='both',
                       help='Type of impulse noise to add (default: both)')
    
    # Shot noise parameters
    parser.add_argument('--shot_intensity', type=float, default=0.05,
                       help='Shot noise intensity factor (0.0-1.0, default: 0.05)')
    
    # Quantization noise parameters
    parser.add_argument('--quant_bits', type=int, default=6,
                       help='Number of bits for quantization (1-8, default: 6)')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not (0.0 <= args.shot_intensity <= 1.0):
        print("Error: shot_intensity must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not (1 <= args.quant_bits <= 8):
        print("Error: quant_bits must be between 1 and 8")
        sys.exit(1)
    
    try:
        add_impulse_noise(
            image_path=args.img,
            output_path=args.output_path,
            noise_type=args.type,
            shot_intensity=args.shot_intensity,
            quant_bits=args.quant_bits,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()