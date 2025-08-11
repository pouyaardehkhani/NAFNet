#!/usr/bin/env python3
"""
Image Noise Addition Script
Adds various types of noise to images while maintaining original dimensions.

Usage:
    # Basic usage
    python noise_additive.py --img input.jpg --output output.jpg --noise gaussian
    python noise_additive.py --img input.jpg --output output.jpg --noise salt_pepper
    python noise_additive.py --img input.jpg --output output.jpg --noise uniform
    python noise_additive.py --img input.jpg --output output.jpg --noise poisson
    
    # With custom parameters
    python noise_additive.py --img input.jpg --output output.jpg --noise gaussian --gaussian_mean 0 --gaussian_std 50
    python noise_additive.py --img input.jpg --output output.jpg --noise salt_pepper --salt_prob 0.1 --pepper_prob 0.08
    python noise_additive.py --img input.jpg --output output.jpg --noise uniform --uniform_low -50 --uniform_high 50
    python noise_additive.py --img input.jpg --output output.jpg --noise poisson --poisson_scale 0.8
"""

import argparse
import numpy as np
import cv2
import os
from skimage import util

def add_gaussian_noise(image_path, output_path, mean=0, std=25):
    """
    Add Gaussian noise to an image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        mean (float): Mean of Gaussian distribution (default: 0)
        std (float): Standard deviation of Gaussian distribution (default: 25)
    """
    # Load image in BGR format (OpenCV default)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    # Convert BGR to RGB for consistent processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype(np.float32)
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, std, img_array.shape)
    
    # Add noise to image
    noisy_img = img_array + noise
    
    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for saving
    noisy_img_bgr = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, noisy_img_bgr)
    print(f"Gaussian noise added and saved to {output_path}")

def add_salt_pepper_noise(image_path, output_path, salt_prob=0.05, pepper_prob=0.05):
    """
    Add salt-and-pepper noise to an image using scikit-image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        salt_prob (float): Probability of salt noise (white pixels)
        pepper_prob (float): Probability of pepper noise (black pixels)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] for scikit-image
    img_normalized = img_rgb.astype(np.float64) / 255.0
    
    # Add salt and pepper noise using scikit-image
    noisy_img = util.random_noise(img_normalized, mode='s&p', 
                                 salt_vs_pepper=salt_prob/(salt_prob + pepper_prob),
                                 amount=salt_prob + pepper_prob)
    
    # Convert back to [0, 255] range
    noisy_img = (noisy_img * 255).astype(np.uint8)
    
    # Convert back to BGR for saving
    noisy_img_bgr = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, noisy_img_bgr)
    print(f"Salt-and-pepper noise added and saved to {output_path}")

def add_uniform_noise(image_path, output_path, low=-30, high=30):
    """
    Add uniform noise to an image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        low (float): Lower bound of uniform distribution (default: -30)
        high (float): Upper bound of uniform distribution (default: 30)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype(np.float32)
    
    # Generate uniform noise
    noise = np.random.uniform(low, high, img_array.shape)
    
    # Add noise to image
    noisy_img = img_array + noise
    
    # Clip values to valid range [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for saving
    noisy_img_bgr = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, noisy_img_bgr)
    print(f"Uniform noise added and saved to {output_path}")

def add_poisson_noise(image_path, output_path, scale=1.0):
    """
    Add Poisson noise to an image (shot noise) using scikit-image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        scale (float): Scaling factor for Poisson noise intensity (default: 1.0)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] for scikit-image
    img_normalized = img_rgb.astype(np.float64) / 255.0
    
    # Scale the image to control noise intensity
    img_scaled = img_normalized * scale
    
    # Add Poisson noise using scikit-image
    noisy_img = util.random_noise(img_scaled, mode='poisson')
    
    # Scale back and convert to [0, 255] range
    noisy_img = (noisy_img / scale * 255).astype(np.uint8)
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # Convert back to BGR for saving
    noisy_img_bgr = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, noisy_img_bgr)
    print(f"Poisson noise added and saved to {output_path}")

def add_speckle_noise(image_path, output_path, variance=0.1):
    """
    Add speckle noise to an image using scikit-image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save noisy image
        variance (float): Variance of the speckle noise (default: 0.1)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] for scikit-image
    img_normalized = img_rgb.astype(np.float64) / 255.0
    
    # Add speckle noise using scikit-image
    noisy_img = util.random_noise(img_normalized, mode='speckle', var=variance)
    
    # Convert back to [0, 255] range
    noisy_img = (noisy_img * 255).astype(np.uint8)
    
    # Convert back to BGR for saving
    noisy_img_bgr = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, noisy_img_bgr)
    print(f"Speckle noise added and saved to {output_path}")

def print_image_info(image_path):
    """Print information about the input image."""
    img = cv2.imread(image_path)
    if img is not None:
        height, width, channels = img.shape
        print(f"Image info: {width}x{height}, {channels} channels")
        print(f"Data type: {img.dtype}, Min: {img.min()}, Max: {img.max()}")

def main():
    parser = argparse.ArgumentParser(description='Add various types of noise to images')
    parser.add_argument('--img', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path to save output image')
    parser.add_argument('--noise', required=True, 
                       choices=['gaussian', 'salt_pepper', 'uniform', 'poisson', 'speckle'],
                       help='Type of noise to add')
    
    # Gaussian noise parameters
    parser.add_argument('--gaussian_mean', type=float, default=0,
                       help='Mean of Gaussian noise distribution (default: 0)')
    parser.add_argument('--gaussian_std', type=float, default=25, 
                       help='Standard deviation for Gaussian noise (default: 25)')
    
    # Salt-and-pepper noise parameters
    parser.add_argument('--salt_prob', type=float, default=0.05,
                       help='Probability of salt noise (white pixels) (default: 0.05)')
    parser.add_argument('--pepper_prob', type=float, default=0.05,
                       help='Probability of pepper noise (black pixels) (default: 0.05)')
    
    # Uniform noise parameters
    parser.add_argument('--uniform_low', type=float, default=-30,
                       help='Lower bound for uniform noise distribution (default: -30)')
    parser.add_argument('--uniform_high', type=float, default=30,
                       help='Upper bound for uniform noise distribution (default: 30)')
    
    # Poisson noise parameters
    parser.add_argument('--poisson_scale', type=float, default=1.0,
                       help='Scale factor for Poisson noise intensity (default: 1.0)')
    
    # Speckle noise parameters
    parser.add_argument('--speckle_variance', type=float, default=0.1,
                       help='Variance of speckle noise (default: 0.1)')
    
    # Additional options
    parser.add_argument('--info', action='store_true',
                       help='Print image information before processing')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Check if input image exists
    if not os.path.exists(args.img):
        print(f"Error: Input image '{args.img}' not found!")
        return
    
    # Print image info if requested
    if args.info:
        print_image_info(args.img)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Apply the selected noise type with custom parameters
    if args.noise == 'gaussian':
        add_gaussian_noise(args.img, args.output, 
                         mean=args.gaussian_mean, 
                         std=args.gaussian_std)
    elif args.noise == 'salt_pepper':
        add_salt_pepper_noise(args.img, args.output, 
                            salt_prob=args.salt_prob, 
                            pepper_prob=args.pepper_prob)
    elif args.noise == 'uniform':
        add_uniform_noise(args.img, args.output, 
                         low=args.uniform_low, 
                         high=args.uniform_high)
    elif args.noise == 'poisson':
        add_poisson_noise(args.img, args.output, scale=args.poisson_scale)
    elif args.noise == 'speckle':
        add_speckle_noise(args.img, args.output, variance=args.speckle_variance)

if __name__ == "__main__":
    main()
    
    