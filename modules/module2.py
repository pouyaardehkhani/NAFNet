"""
Module 2: impulse denoiser
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from scipy import ndimage
from typing import Dict, Tuple, Optional, List
import warnings

class StatisticalJumpDetector:
    """Advanced statistical jump detection using multi-scale analysis and robust estimators."""
    
    def __init__(self, sensitivity: float = 2.5, multi_scale: bool = True):
        """
        Initialize the jump detector.
        
        Args:
            sensitivity: Threshold for jump detection (lower = more sensitive)
            multi_scale: Whether to use multi-scale detection
        """
        self.sensitivity = sensitivity
        self.multi_scale = multi_scale
    
    def detect_jumps_multiscale(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-scale jump detection using statistical methods.
        
        Args:
            image: Input RGB image (H, W, 3)
            
        Returns:
            Binary mask of detected jumps (H, W)
        """
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        detection_maps = []
        
        # Multiple scales for robustness
        scales = [1, 2, 4] if self.multi_scale else [1]
        
        for scale in scales:
            if scale > 1:
                # Downsample for coarse detection
                h, w = gray_image.shape
                scaled_img = cv2.resize(gray_image, (w//scale, h//scale))
            else:
                scaled_img = gray_image
            
            # Statistical change-point detection
            jump_map = self.statistical_changepoint_detection(scaled_img)
            
            if scale > 1:
                # Upsample back to original resolution
                jump_map = cv2.resize(jump_map, (gray_image.shape[1], gray_image.shape[0]))
            
            detection_maps.append(jump_map)
        
        # Combine multi-scale detections
        final_detection = np.maximum.reduce(detection_maps)
        
        return final_detection > self.sensitivity
    
    def statistical_changepoint_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Statistical change-point detection using robust estimators.
        
        Args:
            image: Grayscale image
            
        Returns:
            Jump strength map
        """
        # Ensure image is float64 for precision
        image = image.astype(np.float64)
        
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Second-order derivatives for curvature
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Apply robust estimation
        median_grad = np.median(grad_magnitude)
        grad_residuals = grad_magnitude - median_grad
        grad_weights = self.tukey_biweight(grad_residuals)
        
        median_laplace = np.median(laplacian)
        laplace_residuals = laplacian - median_laplace
        laplace_weights = self.tukey_biweight(laplace_residuals)
        
        # Combine gradient and curvature information
        jump_strength = grad_magnitude * (1 - grad_weights) + \
                       np.abs(laplacian) * (1 - laplace_weights)
        
        return jump_strength
    
    def tukey_biweight(self, residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
        """
        Tukey's biweight function for robust estimation.
        
        Args:
            residuals: Residual values
            c: Tukey's constant
            
        Returns:
            Weight values
        """
        # Avoid division by zero
        mad = np.median(np.abs(residuals))
        if mad == 0:
            mad = 1e-8
        
        normalized_residuals = residuals / mad
        weights = np.where(np.abs(normalized_residuals) <= c,
                         (1 - (normalized_residuals/c)**2)**2, 0)
        return weights


class MultiMethodInpainter:
    """Multi-method inpainting system with various algorithms."""
    
    def __init__(self):
        """Initialize the inpainter with available methods."""
        self.methods = {
            'fast_marching': self.fast_marching_inpainting,
            'navier_stokes': self.navier_stokes_inpainting,
            'exemplar_based': self.exemplar_based_inpainting
        }
    
    def inpaint_all_methods(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all inpainting methods.
        
        Args:
            image: Input RGB image
            mask: Binary mask (1 = inpaint, 0 = keep)
            
        Returns:
            Dictionary of inpainted results
        """
        results = {}
        
        for method_name, method_func in self.methods.items():
            try:
                results[method_name] = method_func(image, mask)
            except Exception as e:
                print(f"Warning: Method {method_name} failed: {str(e)}")
                results[method_name] = image.copy()  # Fallback to original
        
        return results
    
    def fast_marching_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fast Marching Method for inpainting."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            # For color images, process each channel separately
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = cv2.inpaint(
                    image[:, :, c].astype(np.uint8), 
                    mask_uint8, 
                    3, 
                    cv2.INPAINT_TELEA
                )
            return result
        else:
            return cv2.inpaint(image.astype(np.uint8), mask_uint8, 3, cv2.INPAINT_TELEA)
    
    def navier_stokes_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Navier-Stokes equation based inpainting."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            # For color images, process each channel separately
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = cv2.inpaint(
                    image[:, :, c].astype(np.uint8), 
                    mask_uint8, 
                    3, 
                    cv2.INPAINT_NS
                )
            return result
        else:
            return cv2.inpaint(image.astype(np.uint8), mask_uint8, 3, cv2.INPAINT_NS)
    
    def exemplar_based_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Exemplar-based texture synthesis inpainting."""
        # Convert to grayscale for processing if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Priority computation
        confidence_map = self.compute_confidence(gray_image, mask)
        data_term = self.compute_data_term(gray_image, mask)
        
        inpainted = image.copy().astype(np.float64)
        current_mask = mask.copy().astype(np.bool_)
        
        # Iterative exemplar-based filling
        patch_size = 9
        max_iterations = 50  # Reduced for performance
        
        for iteration in range(max_iterations):
            if np.sum(current_mask) == 0:
                break
            
            # Find boundary pixels
            boundary_mask = self.get_boundary_mask(current_mask)
            if np.sum(boundary_mask) == 0:
                break
            
            # Compute priority for boundary pixels
            priority_map = confidence_map * data_term
            priority_boundary = priority_map * boundary_mask
            
            if np.max(priority_boundary) == 0:
                break
            
            max_priority_pos = np.unravel_index(np.argmax(priority_boundary), 
                                              priority_boundary.shape)
            
            # Find best matching patch
            source_patch_pos = self.find_best_match(
                inpainted, current_mask, max_priority_pos, patch_size
            )
            
            if source_patch_pos is not None:
                # Copy source patch to target
                self.copy_patch(inpainted, current_mask, confidence_map,
                              max_priority_pos, source_patch_pos, patch_size)
        
        return np.clip(inpainted, 0, 255).astype(np.uint8)
    
    def compute_confidence(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute confidence values for exemplar-based inpainting."""
        confidence = np.ones_like(mask, dtype=np.float64)
        confidence[mask > 0] = 0
        return confidence
    
    def compute_data_term(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute data term (edge strength) for priority."""
        # Compute gradients
        grad_x = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge strength
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        if np.max(edge_strength) > 0:
            edge_strength = edge_strength / np.max(edge_strength)
        
        return edge_strength
    
    def get_boundary_mask(self, mask: np.ndarray) -> np.ndarray:
        """Get boundary pixels of the mask."""
        # Dilate and subtract original to get boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated.astype(bool) & (~mask)
        return boundary
    
    def find_best_match(self, image: np.ndarray, mask: np.ndarray, 
                       target_pos: Tuple[int, int], patch_size: int) -> Optional[Tuple[int, int]]:
        """Find the best matching patch for exemplar-based inpainting."""
        ty, tx = target_pos
        half_patch = patch_size // 2
        
        # Extract target patch
        y1, y2 = max(0, ty - half_patch), min(image.shape[0], ty + half_patch + 1)
        x1, x2 = max(0, tx - half_patch), min(image.shape[1], tx + half_patch + 1)
        
        if len(image.shape) == 3:
            target_patch = image[y1:y2, x1:x2, :].copy()
            target_mask = mask[y1:y2, x1:x2].copy()
        else:
            target_patch = image[y1:y2, x1:x2].copy()
            target_mask = mask[y1:y2, x1:x2].copy()
        
        best_match_pos = None
        best_ssd = float('inf')
        
        # Search for best match in known regions
        search_step = 2  # Reduce search density for performance
        
        for sy in range(half_patch, image.shape[0] - half_patch, search_step):
            for sx in range(half_patch, image.shape[1] - half_patch, search_step):
                # Skip if source patch contains unknown pixels
                source_mask_region = mask[sy - half_patch:sy + half_patch + 1,
                                        sx - half_patch:sx + half_patch + 1]
                if np.any(source_mask_region):
                    continue
                
                # Extract source patch
                if len(image.shape) == 3:
                    source_patch = image[sy - half_patch:sy + half_patch + 1,
                                       sx - half_patch:sx + half_patch + 1, :]
                else:
                    source_patch = image[sy - half_patch:sy + half_patch + 1,
                                       sx - half_patch:sx + half_patch + 1]
                
                # Compute SSD only for known pixels in target
                if source_patch.shape != target_patch.shape:
                    continue
                
                known_pixels = ~target_mask
                if np.sum(known_pixels) == 0:
                    continue
                
                if len(image.shape) == 3:
                    diff = (source_patch - target_patch) ** 2
                    ssd = np.sum(diff[known_pixels, :])
                else:
                    diff = (source_patch - target_patch) ** 2
                    ssd = np.sum(diff[known_pixels])
                
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_match_pos = (sy, sx)
        
        return best_match_pos
    
    def copy_patch(self, image: np.ndarray, mask: np.ndarray, confidence: np.ndarray,
                   target_pos: Tuple[int, int], source_pos: Tuple[int, int], patch_size: int):
        """Copy patch from source to target position."""
        ty, tx = target_pos
        sy, sx = source_pos
        half_patch = patch_size // 2
        
        # Define patch boundaries
        ty1, ty2 = max(0, ty - half_patch), min(image.shape[0], ty + half_patch + 1)
        tx1, tx2 = max(0, tx - half_patch), min(image.shape[1], tx + half_patch + 1)
        
        sy1, sy2 = max(0, sy - half_patch), min(image.shape[0], sy + half_patch + 1)
        sx1, sx2 = max(0, sx - half_patch), min(image.shape[1], sx + half_patch + 1)
        
        # Get the actual patch sizes (might be different at boundaries)
        target_h, target_w = ty2 - ty1, tx2 - tx1
        source_h, source_w = sy2 - sy1, sx2 - sx1
        
        # Use minimum dimensions to avoid size mismatch
        patch_h = min(target_h, source_h)
        patch_w = min(target_w, source_w)
        
        if patch_h <= 0 or patch_w <= 0:
            return
        
        # Adjust boundaries
        ty2, tx2 = ty1 + patch_h, tx1 + patch_w
        sy2, sx2 = sy1 + patch_h, sx1 + patch_w
        
        # Copy only pixels that need inpainting
        patch_mask = mask[ty1:ty2, tx1:tx2]
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                image[ty1:ty2, tx1:tx2, c][patch_mask] = \
                    image[sy1:sy2, sx1:sx2, c][patch_mask]
        else:
            image[ty1:ty2, tx1:tx2][patch_mask] = \
                image[sy1:sy2, sx1:sx2][patch_mask]
        
        # Update mask and confidence
        mask[ty1:ty2, tx1:tx2] = mask[ty1:ty2, tx1:tx2] & (~patch_mask)
        confidence[ty1:ty2, tx1:tx2][patch_mask] = 1.0


class QualityAssessment:
    """Quality assessment and iterative refinement for inpainting results."""
    
    def __init__(self):
        """Initialize quality assessment."""
        self.metrics = {}
    
    def compute_ssim_local(self, img1: np.ndarray, img2: np.ndarray, 
                          window_size: int = 11) -> np.ndarray:
        """
        Compute local SSIM for quality assessment.
        
        Args:
            img1, img2: Images to compare
            window_size: Size of the sliding window
            
        Returns:
            Local SSIM map
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Ensure same data type
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        ssim_map = np.zeros_like(img1, dtype=np.float64)
        half_window = window_size // 2
        
        for i in range(half_window, img1.shape[0] - half_window):
            for j in range(half_window, img1.shape[1] - half_window):
                window1 = img1[i-half_window:i+half_window+1, 
                             j-half_window:j+half_window+1]
                window2 = img2[i-half_window:i+half_window+1,
                             j-half_window:j+half_window+1]
                
                try:
                    ssim_local = structural_similarity(window1, window2, 
                                                     data_range=255)
                    ssim_map[i, j] = ssim_local
                except:
                    ssim_map[i, j] = 0
        
        return ssim_map
    
    def assess_inpainting_quality(self, inpainted: np.ndarray, mask: np.ndarray) -> float:
        """
        Assess the quality of inpainting in the masked region.
        
        Args:
            inpainted: Inpainted image
            mask: Original mask
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Convert to grayscale for analysis
        if len(inpainted.shape) == 3:
            gray_inpainted = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
        else:
            gray_inpainted = inpainted.copy()
        
        # Compute smoothness in inpainted regions
        grad_x = cv2.Sobel(gray_inpainted.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_inpainted.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient in inpainted regions (lower is smoother)
        inpainted_grad = grad_magnitude[mask > 0]
        if len(inpainted_grad) > 0:
            smoothness_score = 1.0 / (1.0 + np.mean(inpainted_grad) / 100.0)
        else:
            smoothness_score = 1.0
        
        # Boundary continuity assessment
        boundary_mask = self.get_boundary_pixels(mask)
        if np.sum(boundary_mask) > 0:
            boundary_grad = grad_magnitude[boundary_mask]
            continuity_score = 1.0 / (1.0 + np.std(boundary_grad) / 50.0)
        else:
            continuity_score = 1.0
        
        # Combined quality score
        quality_score = 0.6 * smoothness_score + 0.4 * continuity_score
        
        return np.clip(quality_score, 0, 1)
    
    def get_boundary_pixels(self, mask: np.ndarray) -> np.ndarray:
        """Get pixels at the boundary between inpainted and original regions."""
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = (dilated - eroded) > 0
        return boundary
    
    def iterative_refinement(self, inpainted_results: Dict[str, np.ndarray], 
                           original_mask: np.ndarray, 
                           quality_threshold: float = 0.8) -> np.ndarray:
        """
        Iterative refinement based on quality assessment.
        
        Args:
            inpainted_results: Dictionary of inpainting results
            original_mask: Original mask
            quality_threshold: Minimum quality threshold
            
        Returns:
            Best refined result
        """
        best_result = None
        best_quality = 0
        best_method = None
        
        # Evaluate all methods
        for method_name, result in inpainted_results.items():
            quality_score = self.assess_inpainting_quality(result, original_mask)
            
            if quality_score > best_quality:
                best_quality = quality_score
                best_result = result.copy()
                best_method = method_name
        
        print(f"Best method: {best_method} (quality: {best_quality:.3f})")
        
        # If quality is below threshold, apply additional refinement
        if best_quality < quality_threshold:
            print("Applying additional refinement...")
            best_result = self.additional_refinement(best_result, original_mask)
        
        return best_result
    
    def additional_refinement(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply additional refinement techniques."""
        # Apply Gaussian smoothing to inpainted regions
        refined = image.copy()
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                channel = refined[:, :, c].astype(np.float64)
                smoothed = cv2.GaussianBlur(channel, (5, 5), 1.0)
                # Apply smoothing only to inpainted regions
                refined[:, :, c][mask > 0] = smoothed[mask > 0]
        else:
            smoothed = cv2.GaussianBlur(refined.astype(np.float64), (5, 5), 1.0)
            refined[mask > 0] = smoothed[mask > 0]
        
        return refined.astype(np.uint8)


class StatisticalJumpInpaintingSystem:
    """Complete system for statistical jump detection and inpainting."""
    
    def __init__(self, sensitivity: float = 2.5, multi_scale: bool = True):
        """
        Initialize the complete system.
        
        Args:
            sensitivity: Jump detection sensitivity
            multi_scale: Whether to use multi-scale detection
        """
        self.detector = StatisticalJumpDetector(sensitivity, multi_scale)
        self.inpainter = MultiMethodInpainter()
        self.quality_assessor = QualityAssessment()
    
    def process_image(self, image: np.ndarray, 
                     quality_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete processing pipeline: detect jumps and inpaint.
        
        Args:
            image: Input RGB image
            quality_threshold: Quality threshold for refinement
            
        Returns:
            Tuple of (final_result, jump_mask, inpainting_results)
        """
        print("Detecting statistical jumps...")
        jump_mask = self.detector.detect_jumps_multiscale(image)
        
        if np.sum(jump_mask) == 0:
            print("No jumps detected.")
            return image, jump_mask, {}
        
        print(f"Detected {np.sum(jump_mask)} jump pixels.")
        
        print("Applying inpainting methods...")
        inpainting_results = self.inpainter.inpaint_all_methods(image, jump_mask)
        
        print("Assessing quality and selecting best result...")
        final_result = self.quality_assessor.iterative_refinement(
            inpainting_results, jump_mask, quality_threshold
        )
        
        return final_result, jump_mask, inpainting_results


# # Example usage
# if __name__ == "__main__":
#     # Example with synthetic data
#     def create_test_image():
#         """Create a test image with artificial jumps."""
#         image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
#         # Add some artificial jumps
#         image[50:60, 50:150] = 255  # White stripe
#         image[100:110, 30:170] = 0   # Black stripe
        
#         return image
    
#     # Create test image
#     test_image = create_test_image()
    
#     # Initialize system
#     system = StatisticalJumpInpaintingSystem(sensitivity=2.0, multi_scale=True)
    
#     # Process image
#     result, mask, all_results = system.process_image(test_image)
    
#     print("Processing complete!")
#     print(f"Original image shape: {test_image.shape}")
#     print(f"Jump mask shape: {mask.shape}")
#     print(f"Result image shape: {result.shape}")
#     print(f"Available methods: {list(all_results.keys())}")
    
    