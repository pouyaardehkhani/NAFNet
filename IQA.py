import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import entropy
from skimage import filters
from skimage.feature import local_binary_pattern
from skimage.util import img_as_float
import warnings

class ImageQualityAssessment:
    """
    A comprehensive class for image quality assessment metrics.
    Supports both full-reference (FR) and no-reference (NR) metrics.
    """
    
    def __init__(self):
        pass
    
    def psnr(self, original, compressed):
        """
        Peak Signal-to-Noise Ratio (PSNR) - Full Reference
        
        Args:
            original: Reference image (numpy array)
            compressed: Test image (numpy array)
            
        Returns:
            float: PSNR value in dB
        """
        original = img_as_float(original)
        compressed = img_as_float(compressed)
        
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 1.0
        psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr_val
    
    def ssim(self, original, compressed, win_size=11, data_range=1.0):
        """
        Structural Similarity Index (SSIM) - Full Reference
        
        Args:
            original: Reference image (numpy array)
            compressed: Test image (numpy array)
            win_size: Window size for local computations
            data_range: Data range of the input image
            
        Returns:
            float: SSIM value [0, 1]
        """
        original = img_as_float(original)
        compressed = img_as_float(compressed)
        
        if len(original.shape) > 2:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) > 2:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
        
        # Constants for stability
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Gaussian kernel
        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = ndimage.convolve(original, window)[5:-5, 5:-5]
        mu2 = ndimage.convolve(compressed, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ndimage.convolve(original ** 2, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = ndimage.convolve(compressed ** 2, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = ndimage.convolve(original * compressed, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    def lpips(self, original, compressed):
        """
        Learned Perceptual Image Patch Similarity (LPIPS) - Full Reference
        Note: This is a simplified version. For full LPIPS, use the official implementation.
        
        Args:
            original: Reference image (numpy array)
            compressed: Test image (numpy array)
            
        Returns:
            float: Approximate LPIPS score (lower is better)
        """
        # Simplified LPIPS using gradient-based features
        original = img_as_float(original)
        compressed = img_as_float(compressed)
        
        if len(original.shape) > 2:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) > 2:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        grad_x_orig = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_orig = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
        grad_x_comp = cv2.Sobel(compressed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_comp = cv2.Sobel(compressed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute difference in gradient magnitudes
        grad_mag_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)
        grad_mag_comp = np.sqrt(grad_x_comp**2 + grad_y_comp**2)
        
        lpips_approx = np.mean(np.abs(grad_mag_orig - grad_mag_comp))
        return lpips_approx
    
    def niqe(self, image):
        """
        Natural Image Quality Evaluator (NIQE) - No Reference
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            float: NIQE score (lower is better)
        """
        image = img_as_float(image)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute local mean and variance
        mu = ndimage.gaussian_filter(image, sigma=7/6)
        mu_sq = ndimage.gaussian_filter(image**2, sigma=7/6)
        sigma = np.sqrt(np.maximum(mu_sq - mu**2, 0))
        
        # Normalize
        structdis = (image - mu) / (sigma + 1e-10)
        
        # Compute features
        features = []
        
        # Shape parameter of generalized Gaussian distribution
        alpha = self._estimate_aggd_param(structdis.flatten())
        features.append(alpha)
        
        # Left and right variance
        lsigma_best, rsigma_best = self._estimate_ggd_param(structdis.flatten())
        features.extend([lsigma_best, rsigma_best])
        
        # Pairwise products
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for shift in shifts:
            shifted_structdis = np.roll(np.roll(structdis, shift[0], axis=0), shift[1], axis=1)
            pair_product = structdis * shifted_structdis
            alpha = self._estimate_aggd_param(pair_product.flatten())
            mean_param, l_var, r_var = self._estimate_aggd_param_pair(pair_product.flatten())
            features.extend([alpha, mean_param, l_var, r_var])
        
        # Simplified NIQE score (normally requires training on pristine images)
        niqe_score = np.mean(np.array(features))
        return niqe_score
    
    def piqe(self, image):
        """
        Perception-based Image Quality Evaluator (PIQE) - No Reference
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            float: PIQE score (lower is better)
        """
        image = img_as_float(image)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Block-wise analysis
        block_size = 16
        h, w = image.shape
        
        # Pad image if necessary
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
        
        blocks = []
        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = padded[i:i+block_size, j:j+block_size]
                blocks.append(block)
        
        # Compute block features
        variances = [np.var(block) for block in blocks]
        
        # Classify blocks as distorted/undistorted based on variance
        threshold = np.percentile(variances, 10)
        distorted_blocks = [var for var in variances if var < threshold]
        
        if len(distorted_blocks) == 0:
            return 0.0
        
        # Compute PIQE score
        piqe_score = np.mean(distorted_blocks) + 0.1 * np.std(distorted_blocks)
        return piqe_score
    
    def brisque(self, image):
        """
        Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) - No Reference
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            float: BRISQUE score (lower is better for quality)
        """
        image = img_as_float(image)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute local mean and variance
        mu = ndimage.gaussian_filter(image, sigma=7/6)
        mu_sq = ndimage.gaussian_filter(image**2, sigma=7/6)
        sigma = np.sqrt(np.maximum(mu_sq - mu**2, 0))
        
        # Mean subtracted contrast normalized coefficients
        structdis = (image - mu) / (sigma + 1e-10)
        
        # Compute BRISQUE features
        features = []
        
        # AGGD parameters for MSCN coefficients
        alpha, sigma_l, sigma_r = self._compute_brisque_features(structdis)
        features.extend([alpha, (sigma_l + sigma_r) / 2])
        
        # Pairwise products
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for shift in shifts:
            shifted = np.roll(np.roll(structdis, shift[0], axis=0), shift[1], axis=1)
            pair_product = structdis * shifted
            alpha, sigma_l, sigma_r = self._compute_brisque_features(pair_product)
            features.extend([alpha, (sigma_l + sigma_r) / 2])
        
        # Simplified BRISQUE score
        brisque_score = np.sum(features)
        return brisque_score
    
    def vif(self, original, compressed):
        """
        Visual Information Fidelity (VIF) - Full Reference
        
        Args:
            original: Reference image (numpy array)
            compressed: Test image (numpy array)
            
        Returns:
            float: VIF score (higher is better)
        """
        original = img_as_float(original)
        compressed = img_as_float(compressed)
        
        if len(original.shape) > 2:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) > 2:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale analysis
        scales = 4
        vif_val = 0
        
        for scale in range(scales):
            # Gaussian filtering and downsampling
            if scale > 0:
                original = ndimage.gaussian_filter(original, sigma=1)
                compressed = ndimage.gaussian_filter(compressed, sigma=1)
                original = original[::2, ::2]
                compressed = compressed[::2, ::2]
            
            # Compute local statistics
            mu1 = ndimage.gaussian_filter(original, sigma=1.5)
            mu2 = ndimage.gaussian_filter(compressed, sigma=1.5)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = ndimage.gaussian_filter(original ** 2, sigma=1.5) - mu1_sq
            sigma2_sq = ndimage.gaussian_filter(compressed ** 2, sigma=1.5) - mu2_sq
            sigma12 = ndimage.gaussian_filter(original * compressed, sigma=1.5) - mu1_mu2
            
            # Compute VIF at this scale
            sigma1_sq = np.maximum(sigma1_sq, 0)
            sigma2_sq = np.maximum(sigma2_sq, 0)
            
            g = sigma12 / (sigma1_sq + 1e-10)
            sv_sq = sigma2_sq - g * sigma12
            
            g = np.maximum(g, 0)
            sv_sq = np.maximum(sv_sq, 0)
            
            vif_scale = np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + 1e-10)))
            vif_scale /= np.sum(np.log10(1 + sigma1_sq / 1e-10))
            
            vif_val += vif_scale
        
        return vif_val
    
    def fsim(self, original, compressed):
        """
        Feature Similarity Index (FSIM) - Full Reference
        
        Args:
            original: Reference image (numpy array)
            compressed: Test image (numpy array)
            
        Returns:
            float: FSIM score [0, 1] (higher is better)
        """
        original = img_as_float(original)
        compressed = img_as_float(compressed)
        
        if len(original.shape) > 2:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) > 2:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        grad_x1 = ndimage.sobel(original, axis=1)
        grad_y1 = ndimage.sobel(original, axis=0)
        grad_x2 = ndimage.sobel(compressed, axis=1)
        grad_y2 = ndimage.sobel(compressed, axis=0)
        
        # Gradient magnitude
        grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
        grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
        
        # Gradient similarity
        T1 = 0.85  # Threshold for gradient magnitude
        grad_sim = (2 * grad_mag1 * grad_mag2 + T1) / (grad_mag1**2 + grad_mag2**2 + T1)
        
        # Phase congruency (simplified using Laplacian)
        pc1 = np.abs(ndimage.laplace(original))
        pc2 = np.abs(ndimage.laplace(compressed))
        
        T2 = 0.85  # Threshold for phase congruency
        pc_sim = (2 * pc1 * pc2 + T2) / (pc1**2 + pc2**2 + T2)
        
        # Combine similarities
        sim_map = grad_sim * pc_sim
        pc_max = np.maximum(pc1, pc2)
        
        fsim_val = np.sum(sim_map * pc_max) / np.sum(pc_max)
        return fsim_val
    
    def gmsd(self, original, compressed):
        """
        Gradient Magnitude Similarity Deviation (GMSD) - Full Reference
        
        Args:
            original: Reference image (numpy array)
            compressed: Test image (numpy array)
            
        Returns:
            float: GMSD score (lower is better)
        """
        original = img_as_float(original)
        compressed = img_as_float(compressed)
        
        if len(original.shape) > 2:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) > 2:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
        
        # Prewitt filters for gradient computation
        h1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
        h2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3.0
        
        # Compute gradients
        grad_x1 = ndimage.convolve(original, h1)
        grad_y1 = ndimage.convolve(original, h2)
        grad_x2 = ndimage.convolve(compressed, h1)
        grad_y2 = ndimage.convolve(compressed, h2)
        
        # Gradient magnitudes
        grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
        grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
        
        # Gradient magnitude similarity
        c = 0.0026  # Constant for numerical stability
        gms_map = (2 * grad_mag1 * grad_mag2 + c) / (grad_mag1**2 + grad_mag2**2 + c)
        
        # GMSD is standard deviation of GMS map
        gmsd_val = np.std(gms_map)
        return gmsd_val
    
    # Helper methods
    def _estimate_aggd_param(self, vec):
        """Estimate AGGD parameters"""
        gam = np.arange(0.2, 10.001, 0.001)
        r_gam = (gamma_func(2.0/gam) / gamma_func(1.0/gam))**2 / gamma_func(4.0/gam) * gamma_func(1.0/gam)
        
        sigma_sq = np.mean(vec**2)
        E = np.mean(np.abs(vec))
        rho = sigma_sq / E**2
        
        differences = np.abs(rho - r_gam)
        alpha = gam[np.argmin(differences)]
        return alpha
    
    def _estimate_ggd_param(self, vec):
        """Estimate GGD parameters"""
        vec_left = vec[vec < 0]
        vec_right = vec[vec > 0]
        
        if len(vec_left) > 0:
            lsigma_best = np.sqrt(np.mean(vec_left**2))
        else:
            lsigma_best = 0
            
        if len(vec_right) > 0:
            rsigma_best = np.sqrt(np.mean(vec_right**2))
        else:
            rsigma_best = 0
            
        return lsigma_best, rsigma_best
    
    def _estimate_aggd_param_pair(self, vec):
        """Estimate AGGD parameters for paired products"""
        alpha = self._estimate_aggd_param(vec)
        mean_param = np.mean(vec)
        left_var = np.mean((vec[vec < mean_param] - mean_param)**2)
        right_var = np.mean((vec[vec > mean_param] - mean_param)**2)
        return mean_param, left_var, right_var
    
    def _compute_brisque_features(self, coeffs):
        """Compute BRISQUE features from coefficients"""
        alpha = self._estimate_aggd_param(coeffs.flatten())
        
        # Estimate left and right variance
        coeffs_flat = coeffs.flatten()
        left_coeffs = coeffs_flat[coeffs_flat < 0]
        right_coeffs = coeffs_flat[coeffs_flat > 0]
        
        if len(left_coeffs) > 0:
            sigma_l = np.sqrt(np.mean(left_coeffs**2))
        else:
            sigma_l = 0
            
        if len(right_coeffs) > 0:
            sigma_r = np.sqrt(np.mean(right_coeffs**2))
        else:
            sigma_r = 0
        
        return alpha, sigma_l, sigma_r


def gamma_func(x):
    """Simplified gamma function approximation"""
    return np.exp(-x) * (x**(x-0.5)) * np.sqrt(2*np.pi/x)

