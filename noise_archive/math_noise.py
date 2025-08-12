"""
Noise Addition Functions for Image Processing

This script provides functions to add various types of mathematical noise models to images:
- Additive White Gaussian Noise (AWGN)
- Rayleigh Noise
- Exponential Noise  
- Gamma Noise

Package Versions:
numpy : 1.24.3
cv2 : 4.11.0
scipy : 1.13.1
skimage : 0.24.0

Usage Examples:
python math_noise.py --img input.jpg --output output_awgn.jpg --noise awgn --sigma 25
python math_noise.py --img input.jpg --output output_rayleigh.jpg --noise rayleigh --scale 30
python math_noise.py --img input.jpg --output output_exponential.jpg --noise exponential --scale 20
python math_noise.py --img input.jpg --output output_gamma.jpg --noise gamma --shape 2 --scale 25
"""

import numpy as np
import cv2
import argparse
from scipy import stats
import os

def add_awgn_noise(image_path, output_path, sigma=25):
    """
    Add Additive White Gaussian Noise (AWGN) to an image.
    
    Mathematical Model: I(x,y) + N(0,σ²)
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        sigma (float): Standard deviation of Gaussian noise
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to float for processing
    img_float = img.astype(np.float64)
    
    # Generate Gaussian noise with mean=0 and std=sigma
    noise = np.random.normal(0, sigma, img_float.shape)
    
    # Add noise to image
    noisy_img = img_float + noise
    
    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # Convert back to uint8
    noisy_img = noisy_img.astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, noisy_img)
    print(f"AWGN noise added. Saved to: {output_path}")

def add_rayleigh_noise(image_path, output_path, scale=30):
    """
    Add Rayleigh noise to an image (common in radar imaging).
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        scale (float): Scale parameter for Rayleigh distribution
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to float for processing
    img_float = img.astype(np.float64)
    
    # Generate Rayleigh noise
    noise = np.random.rayleigh(scale, img_float.shape)
    
    # Add noise to image
    noisy_img = img_float + noise
    
    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # Convert back to uint8
    noisy_img = noisy_img.astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, noisy_img)
    print(f"Rayleigh noise added. Saved to: {output_path}")

def add_exponential_noise(image_path, output_path, scale=20):
    """
    Add Exponential noise to an image (heavy-tailed distribution).
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        scale (float): Scale parameter (1/rate) for exponential distribution
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to float for processing
    img_float = img.astype(np.float64)
    
    # Generate Exponential noise
    noise = np.random.exponential(scale, img_float.shape)
    
    # Add noise to image
    noisy_img = img_float + noise
    
    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # Convert back to uint8
    noisy_img = noisy_img.astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, noisy_img)
    print(f"Exponential noise added. Saved to: {output_path}")

def add_gamma_noise(image_path, output_path, shape=2, scale=25):
    """
    Add Gamma noise to an image (models multiplicative effects).
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        shape (float): Shape parameter (alpha) for gamma distribution
        scale (float): Scale parameter (beta) for gamma distribution
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to float for processing
    img_float = img.astype(np.float64)
    
    # Generate Gamma noise
    noise = np.random.gamma(shape, scale, img_float.shape)
    
    # Add noise to image
    noisy_img = img_float + noise
    
    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # Convert back to uint8
    noisy_img = noisy_img.astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, noisy_img)
    print(f"Gamma noise added. Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add various types of noise to images')
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--noise', required=True, choices=['awgn', 'rayleigh', 'exponential', 'gamma'],
                       help='Type of noise to add')
    
    # AWGN parameters
    parser.add_argument('--sigma', type=float, default=25,
                       help='Standard deviation for AWGN (default: 25)')
    
    # Rayleigh parameters
    parser.add_argument('--rayleigh_scale', type=float, default=30,
                       help='Scale parameter for Rayleigh noise (default: 30)')
    
    # Exponential parameters
    parser.add_argument('--exp_scale', type=float, default=20,
                       help='Scale parameter for Exponential noise (default: 20)')
    
    # Gamma parameters
    parser.add_argument('--gamma_shape', type=float, default=2,
                       help='Shape parameter for Gamma noise (default: 2)')
    parser.add_argument('--gamma_scale', type=float, default=25,
                       help='Scale parameter for Gamma noise (default: 25)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.img):
        print(f"Error: Input image '{args.img}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply selected noise
    try:
        if args.noise == 'awgn':
            add_awgn_noise(args.img, args.output, args.sigma)
        elif args.noise == 'rayleigh':
            add_rayleigh_noise(args.img, args.output, args.rayleigh_scale)
        elif args.noise == 'exponential':
            add_exponential_noise(args.img, args.output, args.exp_scale)
        elif args.noise == 'gamma':
            add_gamma_noise(args.img, args.output, args.gamma_shape, args.gamma_scale)
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()