"""
Module 4: structured + spatial denoiser

multi-component is better
"""

import numpy as np
import cv2
import pywt
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class VariationalDecomposer:
    """
    Main class for variational pattern decomposition using Mumford-Shah functional
    and Rudin-Osher-Fatemi model adaptation.
    """
    
    def __init__(self, lambda_tv: float = 0.1, lambda_structure: float = 0.05, 
                 max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize the Variational Decomposer.
        
        Args:
            lambda_tv: Total variation regularization parameter
            lambda_structure: Structure preservation parameter
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.lambda_tv = lambda_tv
        self.lambda_structure = lambda_structure
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def mumford_shah_functional(self, image: np.ndarray, u: np.ndarray, 
                              v: np.ndarray, gamma: float = 1.0) -> float:
        """
        Compute Mumford-Shah functional for image decomposition.
        
        Args:
            image: Input image
            u: Cartoon component
            v: Texture component
            gamma: Oscillatory penalty weight
            
        Returns:
            Functional value
        """
        # Data fidelity term
        data_term = 0.5 * np.sum((image - u - v)**2)
        
        # Total variation of u (cartoon part)
        grad_u_x, grad_u_y = np.gradient(u)
        tv_term = self.lambda_tv * np.sum(np.sqrt(grad_u_x**2 + grad_u_y**2 + 1e-8))
        
        # Oscillatory penalty for v (texture part)
        laplacian_v = cv2.Laplacian(v.astype(np.float32), cv2.CV_64F)
        oscillatory_term = gamma * np.sum(v**2) - self.lambda_structure * np.sum(laplacian_v**2)
        
        return data_term + tv_term + oscillatory_term
    
    def rudin_osher_fatemi_adaptation(self, image: np.ndarray, 
                                    lambda_param: float = 0.1) -> np.ndarray:
        """
        ROF model with structure preservation for single channel.
        
        Args:
            image: Input image (single channel)
            lambda_param: Regularization parameter
            
        Returns:
            Denoised image
        """
        u = image.astype(np.float64)  # Initialize with input
        p = np.zeros((2, *image.shape), dtype=np.float64)  # Dual variable
        
        # Parameters
        tau = 0.02  # Primal step size
        sigma = 0.02  # Dual step size
        theta = 1.0  # Over-relaxation parameter
        
        for iteration in range(self.max_iterations):
            u_prev = u.copy()
            
            # Dual update
            grad_u = self.compute_gradient(u)
            p_new = p + sigma * grad_u
            p_new = self.project_dual_constraint(p_new)
            
            # Primal update
            div_p = self.compute_divergence(p_new)
            u_new = (u + tau * (image + lambda_param * div_p)) / (1 + tau)
            
            # Over-relaxation
            u_bar = u_new + theta * (u_new - u)
            
            # Check convergence
            if np.linalg.norm(u_new - u_prev) < self.tolerance:
                break
            
            u = u_new.copy()
            p = p_new.copy()
        
        return u
    
    def compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """Compute gradient using forward differences."""
        grad_x = np.diff(image, axis=1, append=image[:, -1:])
        grad_y = np.diff(image, axis=0, append=image[-1:, :])
        return np.stack([grad_x, grad_y])
    
    def compute_divergence(self, p: np.ndarray) -> np.ndarray:
        """Compute divergence using backward differences."""
        div_x = np.diff(p[0], axis=1, prepend=p[0][:, :1])
        div_y = np.diff(p[1], axis=0, prepend=p[1][:1, :])
        return div_x + div_y
    
    def project_dual_constraint(self, p: np.ndarray) -> np.ndarray:
        """Project onto dual constraint set."""
        norm_p = np.sqrt(p[0]**2 + p[1]**2)
        norm_p = np.maximum(norm_p, 1.0)
        return p / norm_p[None, ...]
    
    def decompose_rgb(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ROF decomposition to RGB image.
        
        Args:
            rgb_image: Input RGB image
            
        Returns:
            Tuple of (structure, texture) components
        """
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Input must be RGB image with shape (H, W, 3)")
        
        structure = np.zeros_like(rgb_image, dtype=np.float64)
        
        for channel in range(3):
            structure[:, :, channel] = self.rudin_osher_fatemi_adaptation(
                rgb_image[:, :, channel]
            )
        
        texture = rgb_image.astype(np.float64) - structure
        
        return structure, texture


class MultiComponentDecomposer:
    """
    Multi-component decomposer for structure-texture-noise separation.
    """
    
    def __init__(self, n_components: int = 3, max_iterations: int = 50):
        """
        Initialize the multi-component decomposer.
        
        Args:
            n_components: Number of components to decompose into
            max_iterations: Maximum iterations for decomposition
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.components = {}
    
    def structure_texture_noise_decomposition(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose image into structure, texture, and noise components.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (structure, texture, noise) components
        """
        if len(image.shape) == 3:
            # Handle RGB image
            return self._decompose_rgb_image(image)
        else:
            # Handle grayscale image
            return self._decompose_single_channel(image)
    
    def _decompose_rgb_image(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose RGB image channel by channel."""
        structure = np.zeros_like(rgb_image, dtype=np.float64)
        texture = np.zeros_like(rgb_image, dtype=np.float64)
        noise = np.zeros_like(rgb_image, dtype=np.float64)
        
        for channel in range(3):
            s, t, n = self._decompose_single_channel(rgb_image[:, :, channel])
            structure[:, :, channel] = s
            texture[:, :, channel] = t
            noise[:, :, channel] = n
        
        return structure, texture, noise
    
    def _decompose_single_channel(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose single channel image."""
        # Initialize components
        structure = np.zeros_like(image, dtype=np.float64)
        texture = np.zeros_like(image, dtype=np.float64)
        noise = np.zeros_like(image, dtype=np.float64)
        
        image_float = image.astype(np.float64)
        
        # Alternating minimization
        for iteration in range(self.max_iterations):
            # Update structure (cartoon part)
            structure = self.solve_structure_subproblem(image_float, texture, noise)
            
            # Update texture (oscillatory part)
            texture = self.solve_texture_subproblem(image_float, structure, noise)
            
            # Update noise (residual)
            noise = image_float - structure - texture
            
            # Apply noise constraint (sparsity in wavelet domain)
            noise = self.apply_noise_constraint(noise)
        
        return structure, texture, noise
    
    def solve_structure_subproblem(self, image: np.ndarray, texture: np.ndarray, 
                                 noise: np.ndarray, lambda_s: float = 0.1) -> np.ndarray:
        """
        Solve for structure component using TV regularization.
        
        Args:
            image: Input image
            texture: Current texture estimate
            noise: Current noise estimate
            lambda_s: Regularization parameter
            
        Returns:
            Updated structure component
        """
        data_target = image - texture - noise
        structure = self.tv_denoising(data_target, lambda_s)
        return structure
    
    def solve_texture_subproblem(self, image: np.ndarray, structure: np.ndarray, 
                               noise: np.ndarray, lambda_t: float = 0.05) -> np.ndarray:
        """
        Solve for texture component using G-norm.
        
        Args:
            image: Input image
            structure: Current structure estimate
            noise: Current noise estimate
            lambda_t: Regularization parameter
            
        Returns:
            Updated texture component
        """
        data_target = image - structure - noise
        texture = self.g_norm_denoising(data_target, lambda_t)
        return texture
    
    def tv_denoising(self, image: np.ndarray, lambda_param: float) -> np.ndarray:
        """Total variation denoising."""
        # Simple gradient descent for TV denoising
        u = image.copy()
        dt = 0.1
        
        for _ in range(20):
            # Compute gradients
            grad_x = np.diff(u, axis=1, append=u[:, -1:])
            grad_y = np.diff(u, axis=0, append=u[-1:, :])
            
            # Compute divergence of normalized gradient
            grad_norm = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            div_x = np.diff(grad_x / grad_norm, axis=1, prepend=(grad_x / grad_norm)[:, :1])
            div_y = np.diff(grad_y / grad_norm, axis=0, prepend=(grad_y / grad_norm)[:1, :])
            
            # Update
            u = u + dt * (image - u + lambda_param * (div_x + div_y))
        
        return u
    
    def g_norm_denoising(self, image: np.ndarray, lambda_param: float) -> np.ndarray:
        """G-norm denoising for texture extraction."""
        # Ensure image is float32 for DCT
        image_float = image.astype(np.float32)
        
        # DCT-based G-norm approximation
        dct_coeffs = cv2.dct(image_float)
        
        # Soft thresholding in DCT domain
        threshold = lambda_param * np.median(np.abs(dct_coeffs))
        
        # Apply soft thresholding
        dct_denoised = np.sign(dct_coeffs) * np.maximum(
            np.abs(dct_coeffs) - threshold, 0
        )
        
        # Inverse DCT
        result = cv2.idct(dct_denoised)
        
        return result.astype(np.float64)
    
    def apply_noise_constraint(self, noise: np.ndarray, sparsity_param: float = 0.01) -> np.ndarray:
        """Apply sparsity constraint to noise component."""
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec2(noise, 'db8', levels=4)
            
            # Soft thresholding
            threshold = sparsity_param * np.std(noise)
            coeffs_thresh = []
            
            for i, coeff in enumerate(coeffs):
                if i == 0:  # Approximation coefficients
                    coeffs_thresh.append(coeff)
                else:  # Detail coefficients
                    if isinstance(coeff, tuple):
                        coeff_thresh = tuple([
                            pywt.threshold(c, threshold, mode='soft') 
                            for c in coeff
                        ])
                    else:
                        coeff_thresh = pywt.threshold(coeff, threshold, mode='soft')
                    coeffs_thresh.append(coeff_thresh)
            
            # Reconstruction
            noise_constrained = pywt.waverec2(coeffs_thresh, 'db8')
            
            return noise_constrained
        except:
            # Fallback to simple thresholding if wavelet fails
            threshold = sparsity_param * np.std(noise)
            return np.sign(noise) * np.maximum(np.abs(noise) - threshold, 0)


class AdvancedOptimizer:
    """
    Advanced optimization algorithms for variational methods.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize the advanced optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.optimization_methods = {}
    
    def split_bregman_method(self, image: np.ndarray, lambda_param: float = 0.1, 
                           mu: float = 1.0) -> np.ndarray:
        """
        Split Bregman method for TV denoising.
        
        Args:
            image: Input image
            lambda_param: Regularization parameter
            mu: Penalty parameter
            
        Returns:
            Denoised image
        """
        # Initialize variables
        u = image.astype(np.float64)
        d = np.zeros((2, *image.shape))  # Auxiliary variable
        b = np.zeros((2, *image.shape))  # Bregman variable
        
        # Iteration parameters
        max_inner_iter = 5
        max_outer_iter = 20
        
        for outer_iter in range(max_outer_iter):
            # u-subproblem (Gaussian denoising)
            for inner_iter in range(max_inner_iter):
                grad_u = self.compute_gradient(u)
                div_db = self.compute_divergence(d - b)
                
                u_new = (image + mu * div_db) / (1 + mu * lambda_param)
                u = u_new
            
            # d-subproblem (shrinkage)
            grad_u_b = self.compute_gradient(u) + b
            d = self.shrinkage(grad_u_b, 1.0 / mu)
            
            # Bregman update
            b = b + self.compute_gradient(u) - d
        
        return u
    
    def primal_dual_algorithm(self, image: np.ndarray, lambda_param: float = 0.1) -> np.ndarray:
        """
        Primal-dual algorithm (Chambolle-Pock).
        
        Args:
            image: Input image
            lambda_param: Regularization parameter
            
        Returns:
            Optimized image
        """
        # Initialize
        u = image.astype(np.float64)
        p = np.zeros((2, *image.shape))
        
        # Step sizes
        tau = 0.02
        sigma = 0.02
        theta = 1.0
        
        u_bar = u.copy()
        
        for iteration in range(self.max_iterations):
            u_prev = u.copy()
            
            # Update dual variable
            p_new = p + sigma * self.compute_gradient(u_bar)
            p_new = self.project_dual_constraint(p_new)
            
            # Update primal variable
            div_p = self.compute_divergence(p_new)
            u_new = (u + tau * (image + lambda_param * div_p)) / (1 + tau)
            
            # Over-relaxation
            u_bar = u_new + theta * (u_new - u)
            
            # Check convergence
            if np.linalg.norm(u_new - u_prev) < self.tolerance:
                break
            
            u, p = u_new, p_new
        
        return u
    
    def compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """Compute gradient using forward differences."""
        grad_x = np.diff(image, axis=1, append=image[:, -1:])
        grad_y = np.diff(image, axis=0, append=image[-1:, :])
        return np.stack([grad_x, grad_y])
    
    def compute_divergence(self, p: np.ndarray) -> np.ndarray:
        """Compute divergence using backward differences."""
        div_x = np.diff(p[0], axis=1, prepend=p[0][:, :1])
        div_y = np.diff(p[1], axis=0, prepend=p[1][:1, :])
        return div_x + div_y
    
    def project_dual_constraint(self, p: np.ndarray) -> np.ndarray:
        """Project onto dual constraint set."""
        norm_p = np.sqrt(p[0]**2 + p[1]**2)
        norm_p = np.maximum(norm_p, 1.0)
        return p / norm_p[None, ...]
    
    def shrinkage(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft shrinkage operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


class VariationalPatternDecomposer:
    """
    Main wrapper class that combines all decomposition methods.
    """
    
    def __init__(self, method: str = 'mumford_shah', **kwargs):
        """
        Initialize the pattern decomposer.
        
        Args:
            method: Decomposition method ('mumford_shah', 'multi_component', 'advanced_opt')
            **kwargs: Additional parameters for the chosen method
        """
        self.method = method
        self.variational = VariationalDecomposer(**kwargs)
        self.multi_component = MultiComponentDecomposer(**kwargs)
        self.optimizer = AdvancedOptimizer(**kwargs)
    
    def decompose(self, rgb_image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Main decomposition method that handles RGB images.
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing decomposed components
        """
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Input must be RGB image with shape (H, W, 3)")
        
        # Normalize image to [0, 1] range
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image.astype(np.float64) / 255.0
        
        if self.method == 'mumford_shah':
            structure, texture = self.variational.decompose_rgb(rgb_image)
            return {
                'structure': structure,
                'texture': texture,
                'original': rgb_image
            }
        
        elif self.method == 'multi_component':
            structure, texture, noise = self.multi_component.structure_texture_noise_decomposition(rgb_image)
            return {
                'structure': structure,
                'texture': texture,
                'noise': noise,
                'original': rgb_image
            }
        
        elif self.method == 'advanced_opt':
            # Apply advanced optimization to each channel
            optimized = np.zeros_like(rgb_image)
            for channel in range(3):
                if kwargs.get('optimization_method') == 'split_bregman':
                    optimized[:, :, channel] = self.optimizer.split_bregman_method(
                        rgb_image[:, :, channel], **kwargs
                    )
                else:
                    optimized[:, :, channel] = self.optimizer.primal_dual_algorithm(
                        rgb_image[:, :, channel], **kwargs
                    )
            
            texture = rgb_image - optimized
            return {
                'structure': optimized,
                'texture': texture,
                'original': rgb_image
            }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def visualize_components(self, components: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prepare components for visualization by clipping values.
        
        Args:
            components: Dictionary of image components
            
        Returns:
            Dictionary of clipped components ready for display
        """
        visualized = {}
        
        for key, component in components.items():
            # Clip values to [0, 1] range
            vis_component = np.clip(component, 0, 1)
            visualized[key] = vis_component
        
        return visualized


# Example usage function
def example_usage():
    """Example of how to use the Variational Pattern Decomposer."""
    
    # Create a simple test RGB image
    test_image = np.random.rand(100, 100, 3)
    
    # Method 1: Mumford-Shah decomposition
    decomposer_ms = VariationalPatternDecomposer(
        method='mumford_shah',
        lambda_tv=0.1,
        lambda_structure=0.05
    )
    
    components_ms = decomposer_ms.decompose(test_image)
    print("Mumford-Shah components:", components_ms.keys())
    
    # Method 2: Multi-component decomposition
    decomposer_mc = VariationalPatternDecomposer(
        method='multi_component',
        n_components=3
    )
    
    components_mc = decomposer_mc.decompose(test_image)
    print("Multi-component components:", components_mc.keys())
    
    # Method 3: Advanced optimization
    decomposer_ao = VariationalPatternDecomposer(
        method='advanced_opt'
    )
    
    components_ao = decomposer_ao.decompose(
        test_image, 
        optimization_method='split_bregman',
        lambda_param=0.1
    )
    print("Advanced optimization components:", components_ao.keys())
    
    return components_ms, components_mc, components_ao


# if __name__ == "__main__":
#     example_usage()