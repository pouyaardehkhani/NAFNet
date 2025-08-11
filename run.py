import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import IQA
from IQA import ImageQualityAssessment
from datetime import datetime
import os


def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1) 
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('NAFNet output', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)


def single_image_inference(model, img, save_path):
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      sr_img = tensor2img([visuals['result']])
      imwrite(sr_img, save_path)


def log_iqa_results(original, output, original_image_name, log_dir="logs"):
    """
    Perform image quality assessment and log results to txt file
    
    Args:
        original: Original image array
        output: output/test image array
        original_image_name: Name of the original image
        log_dir: Directory to save log files
    """
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create IQA object
    iqa = ImageQualityAssessment()
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{original_image_name}_{timestamp}_iqa_results.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    # Calculate all metrics
    try:
        psnr_val = iqa.psnr(original, output)
        ssim_val = iqa.ssim(original, output)
        lpips_val = iqa.lpips(original, output)
        vif_val = iqa.vif(original, output)
        fsim_val = iqa.fsim(original, output)
        gmsd_val = iqa.gmsd(original, output)
        
        # No-reference metrics
        niqe_val = iqa.niqe(output)
        piqe_val = iqa.piqe(output)
        brisque_val = iqa.brisque(output)
        
        # Write results to log file
        with open(log_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("IMAGE QUALITY ASSESSMENT RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Original Image: {original_image_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Shape: {original.shape}\n")
            f.write(f"output Shape: {output.shape}\n")
            f.write("-" * 50 + "\n")
            f.write("FULL-REFERENCE METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"PSNR: {psnr_val:.2f} dB\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            f.write(f"LPIPS (approx): {lpips_val:.4f}\n")
            f.write(f"VIF: {vif_val:.4f}\n")
            f.write(f"FSIM: {fsim_val:.4f}\n")
            f.write(f"GMSD: {gmsd_val:.4f}\n")
            f.write("-" * 50 + "\n")
            f.write("NO-REFERENCE METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"NIQE: {niqe_val:.4f}\n")
            f.write(f"PIQE: {piqe_val:.4f}\n")
            f.write(f"BRISQUE: {brisque_val:.4f}\n")
            f.write("=" * 50 + "\n")
            
        print(f"Results logged to: {log_path}")
        print("\nImage Quality Assessment Results:")
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        print(f"LPIPS (approx): {lpips_val:.4f}")
        print(f"VIF: {vif_val:.4f}")
        print(f"FSIM: {fsim_val:.4f}")
        print(f"GMSD: {gmsd_val:.4f}")
        print(f"\nNo-Reference Metrics:")
        print(f"NIQE: {niqe_val:.4f}")
        print(f"PIQE: {piqe_val:.4f}")
        print(f"BRISQUE: {brisque_val:.4f}")
        
    except Exception as e:
        error_msg = f"Error calculating metrics: {str(e)}"
        print(error_msg)
        with open(log_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("IMAGE QUALITY ASSESSMENT - ERROR\n")
            f.write("=" * 50 + "\n")
            f.write(f"Original Image: {original_image_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {error_msg}\n")
            f.write("=" * 50 + "\n")


def save_comparison_plots(test_img, gt_img, output_dir):
    """Save comparison plots of the images"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display ground truth image
    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth Image')
    axes[0].axis('off')
    
    # Display test image
    axes[1].imshow(test_img)
    axes[1].set_title('Test Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f"comparison_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {plot_path}")
    

def main():
    parser = argparse.ArgumentParser(description='Process two image files and run IQA metrics')
    parser.add_argument('--img1', required=True, help='Path to first image (test image)')
    parser.add_argument('--img2-gt', required=True, help='Path to ground truth image')
    parser.add_argument('--output-dir', default='./results', help='Output directory for logs and results')
    parser.add_argument('--save-plots', action='store_true', help='Save comparison plots')
    
    args = parser.parse_args()
    
    # Validate file paths
    img1_path = Path(args.img1)
    img2_gt_path = Path(getattr(args, 'img2_gt'))
    output_dir = Path(args.output_dir)
    
    if not img1_path.exists():
        print(f"Error: Image file '{args.img1}' not found", file=sys.stderr)
        sys.exit(1)
    
    if not img2_gt_path.exists():
        print(f"Error: Ground truth image file '{getattr(args, 'img2_gt')}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Test Image: {img1_path}")
    print(f"Ground Truth Image: {img2_gt_path}")
    print(f"Output Directory: {output_dir}")
    
    #run pipeline, run metrics, save logs
    opt_path = 'options/test/SIDD/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)
    
    img_1_input = imread(img1_path)
    img_2_input = imread(img2_gt_path)
    inp = img2tensor(img_1_input)
    single_image_inference(NAFNet, inp, output_dir)
    
    try:
        # Initialize IQA class
        print("Initializing Image Quality Assessment...")
        iqa = ImageQualityAssessment()
        
        # Run IQA pipeline and log results
        print("Running Image Quality Assessment pipeline...")
        log_iqa_results(img_2_input, img_1_input, img1_path.stem, str(output_dir))
        
        # Save comparison plots if requested
        if args.save_plots:
            save_comparison_plots(img_1_input, img_2_input, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Error in pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    

if __name__ == "__main__":
    main()
    
