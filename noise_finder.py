"""
detector = create_detector('ultra_fast', 'huge')
results = detector.detect_noise_type_fast('0010_NOISY_SRGB_010.PNG')
detector.print_results(results)
"""

import numpy as np
import cv2
from scipy import stats, signal, ndimage
from skimage import filters, measure, feature, segmentation, restoration
from skimage.filters import rank
from skimage.morphology import disk
from skimage.restoration import estimate_sigma
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

class FastBlindNoiseDetector:
    """
    Optimized blind noise detection system for high-resolution images.
    Uses multiple speed optimization techniques:
    - Multi-scale analysis
    - ROI sampling
    - Parallel processing
    - Efficient algorithms
    """
    
    def __init__(self, max_size=1024, num_samples=50, num_threads=4, fast_mode=True):
        """
        Initialize the fast detector
        
        Args:
            max_size: Maximum image dimension for processing (larger images will be downsampled)
            num_samples: Number of sample regions to analyze
            num_threads: Number of threads for parallel processing
            fast_mode: If True, use fastest algorithms; if False, use more thorough analysis
        """
        self.max_size = max_size
        self.num_samples = num_samples
        self.num_threads = num_threads
        self.fast_mode = fast_mode
        
        self.noise_types = {
            'gaussian': 'Additive White Gaussian Noise (AWGN)',
            'salt_pepper': 'Salt and Pepper Noise',
            'poisson': 'Poisson Noise',
            'uniform': 'Uniform Noise',
            'speckle': 'Speckle Noise',
            'pink': 'Pink Noise (1/f)',
            'blue': 'Blue Noise',
            'brown': 'Brown Noise',
            'white': 'White Noise',
            'shot': 'Shot Noise',
            'quantization': 'Quantization Noise',
            'iso_sensor': 'ISO/Sensor Noise',
            'chromatic': 'Chromatic Noise',
            'motion_blur': 'Motion Blur',
            'vibration': 'Vibration Noise',
            'checkerboard': 'Checkerboard Pattern',
            'stripe': 'Stripe Pattern',
            'ring': 'Ring Artifacts',
            'banding': 'Banding Noise',
            'compression': 'Compression Artifacts',
            'low_light': 'Low Light Noise'
        }
    
    def load_and_preprocess_image(self, image_path):
        """Load and efficiently preprocess high-resolution image"""
        # Load image - no time limit, let it take as long as needed
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_shape = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Multi-scale approach: create different resolution versions
        scales = self._create_multi_scale_versions(image)
        
        return {
            'original_shape': original_shape,
            'scales': scales,
            'full_image': image if max(image.shape[:2]) <= self.max_size else None
        }
    
    def _create_multi_scale_versions(self, image):
        """Create multiple resolution versions of the image"""
        scales = {}
        h, w = image.shape[:2]
        
        # Full resolution (if small enough)
        if max(h, w) <= self.max_size:
            scales['full'] = image
        
        # Medium resolution
        if max(h, w) > 512:
            scale_factor = 512 / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            scales['medium'] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Low resolution for quick analysis
        scale_factor = 256 / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scales['low'] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return scales
    
    def _sample_regions(self, image, num_regions=None):
        """Efficiently sample regions from high-resolution image"""
        if num_regions is None:
            num_regions = self.num_samples
        
        h, w = image.shape[:2]
        
        if max(h, w) <= self.max_size:
            # Image is small enough, use regular sampling
            return self._extract_noise_patches_fast(image)
        
        # For large images, sample random regions
        patch_size = 64
        regions = []
        
        # Generate random coordinates
        np.random.seed(42)  # For reproducibility
        for _ in range(num_regions):
            y = np.random.randint(0, max(1, h - patch_size))
            x = np.random.randint(0, max(1, w - patch_size))
            
            patch = image[y:y+patch_size, x:x+patch_size]
            if len(patch.shape) == 3:
                patch_gray = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                patch_gray = patch.astype(np.uint8)
            
            # Only use patches with some variation (avoid pure backgrounds)
            if np.std(patch_gray) > 5:
                # Extract high-frequency components (noise)
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                noise_patch = cv2.filter2D(patch_gray.astype(np.float32), -1, kernel)
                regions.extend(noise_patch.flatten())
        
        return np.array(regions)
    
    def _extract_noise_patches_fast(self, image, patch_size=32):
        """Fast noise extraction using stride sampling"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        noise_patches = []
        
        # Use larger stride for speed
        stride = patch_size // 2
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = gray[i:i+patch_size, j:j+patch_size]
                
                # Quick homogeneity check (variance threshold)
                if np.var(patch) < np.var(gray) * 0.2:
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                    noise_patch = cv2.filter2D(patch.astype(np.float32), -1, kernel)
                    noise_patches.extend(noise_patch[::2, ::2].flatten())  # Subsample for speed
                
                # Limit number of patches for speed
                if len(noise_patches) > self.num_samples * 100:
                    break
            
            if len(noise_patches) > self.num_samples * 100:
                break
        
        return np.array(noise_patches[:self.num_samples * 100]) if noise_patches else np.array([])
    
    def fast_noise_estimation(self, image):
        """Ultra-fast noise estimation using optimized algorithm"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Use fast approximation method
        h, w = gray.shape
        
        # Sample random locations for speed
        if h * w > 256 * 256:
            # Large image: sample 10000 random pixels
            indices = np.random.choice(h * w, size=min(10000, h * w), replace=False)
            sampled_pixels = gray.flatten()[indices].reshape(-1, 1)
            
            # Apply Laplacian kernel to samples (approximate)
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            
            # For speed, use simpler edge detection on samples
            edges = cv2.Canny(gray[::4, ::4], 50, 150)  # Downsample for speed
            if np.any(edges):
                sigma = np.std(edges) * 0.6745
            else:
                sigma = np.std(gray) * 0.1  # Fallback
        else:
            # Small image: use standard method
            try:
                sigma = restoration.estimate_sigma(gray, average_sigmas=True, multichannel=False)
            except:
                sigma = np.std(gray) * 0.1
        
        return sigma
    
    def detect_additive_noise_fast(self, image_data):
        """Fast additive noise detection"""
        # Use the smallest scale for speed
        scale_key = 'low' if 'low' in image_data['scales'] else list(image_data['scales'].keys())[0]
        image = image_data['scales'][scale_key]
        
        noise_patches = self._sample_regions(image, self.num_samples)
        
        if len(noise_patches) == 0:
            return {}
        
        results = {}
        
        # Fast statistical tests
        if self.fast_mode:
            # Use faster, approximate tests
            
            # Gaussian test (simplified)
            skewness = stats.skew(noise_patches)
            kurtosis = stats.kurtosis(noise_patches)
            gaussian_score = 1 / (1 + abs(skewness) + abs(kurtosis - 3))
            
            results['gaussian'] = {
                'probability': gaussian_score,
                'evidence': f"Skew: {skewness:.3f}, Kurt: {kurtosis:.3f}"
            }
            
            # Salt and pepper (histogram-based)
            hist, _ = np.histogram(noise_patches, bins=20)
            extreme_ratio = (hist[0] + hist[-1]) / np.sum(hist)
            
            results['salt_pepper'] = {
                'probability': min(extreme_ratio * 5, 1.0),
                'evidence': f"Extreme ratio: {extreme_ratio:.4f}"
            }
            
            # Uniform test (range-based approximation)
            data_range = np.max(noise_patches) - np.min(noise_patches)
            expected_std = data_range / np.sqrt(12)  # Uniform distribution std
            actual_std = np.std(noise_patches)
            uniform_score = 1 - abs(actual_std - expected_std) / expected_std
            
            results['uniform'] = {
                'probability': max(0, uniform_score),
                'evidence': f"Std ratio: {actual_std/expected_std:.3f}"
            }
            
        else:
            # Use full statistical tests (slower but more accurate)
            _, p_gaussian = stats.normaltest(noise_patches)
            results['gaussian'] = {
                'probability': 1 - p_gaussian if p_gaussian < 0.05 else 0.0,
                'evidence': f"Normality p-value: {p_gaussian:.6f}"
            }
            
            # Other tests...
        
        return results
    
    def analyze_frequency_fast(self, image_data):
        """Fast frequency domain analysis"""
        # Use medium scale if available, otherwise low scale
        scale_key = 'medium' if 'medium' in image_data['scales'] else 'low'
        image = image_data['scales'][scale_key]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Fast high-pass filtering
        kernel = np.array([[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]]) / 4
        high_freq = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Downsampled FFT for speed
        if high_freq.shape[0] > 256 or high_freq.shape[1] > 256:
            high_freq = cv2.resize(high_freq, (256, 256))
        
        fft = np.fft.fft2(high_freq)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Fast radial analysis
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radial coordinates (vectorized)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        # Fast radial binning
        max_r = min(center_x, center_y, 50)  # Limit for speed
        radial_power = np.zeros(max_r)
        
        for radius in range(1, max_r):
            mask = (r == radius)
            if np.any(mask):
                radial_power[radius] = np.mean(magnitude[mask])
        
        # Remove zeros and fit
        nonzero_indices = radial_power > 0
        if np.sum(nonzero_indices) > 5:
            freqs = np.arange(max_r)[nonzero_indices]
            powers = radial_power[nonzero_indices]
            
            log_freqs = np.log(freqs + 1)
            log_powers = np.log(powers + 1e-10)
            
            try:
                slope, _, r_value, _, _ = stats.linregress(log_freqs, log_powers)
                
                results = {}
                if abs(r_value) > 0.5:  # Only if correlation is significant
                    results['pink'] = {
                        'probability': max(0, 1 - abs(slope + 1)),
                        'evidence': f"Slope: {slope:.3f}"
                    }
                    results['blue'] = {
                        'probability': max(0, 1 - abs(slope - 1)),
                        'evidence': f"Slope: {slope:.3f}"
                    }
                    results['brown'] = {
                        'probability': max(0, 1 - abs(slope + 2)),
                        'evidence': f"Slope: {slope:.3f}"
                    }
                else:
                    results = {'pink': {'probability': 0, 'evidence': 'Low correlation'}}
                
                # White noise (spectral flatness)
                spectral_flatness = np.var(powers) / (np.mean(powers) + 1e-10)
                results['white'] = {
                    'probability': max(0, 1 - spectral_flatness),
                    'evidence': f"Flatness: {spectral_flatness:.3f}"
                }
                
                return results
                
            except:
                return {'pink': {'probability': 0, 'evidence': 'Regression failed'}}
        
        return {}
    
    def detect_patterns_fast(self, image_data):
        """Fast spatial pattern detection"""
        scale_key = 'low'  # Use lowest resolution for pattern detection
        image = image_data['scales'][scale_key]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        results = {}
        
        # Fast checkerboard detection
        kernel_check = np.array([[-1, 1], [1, -1]])
        checkerboard_response = cv2.filter2D(gray.astype(np.float32), -1, kernel_check)
        checkerboard_score = np.std(checkerboard_response) / (np.std(gray) + 1e-10)
        
        results['checkerboard'] = {
            'probability': min(checkerboard_score / 2, 1.0),
            'evidence': f"Response: {checkerboard_score:.3f}"
        }
        
        # Fast stripe detection using 1D profiles
        h_profile = np.mean(gray, axis=1)
        v_profile = np.mean(gray, axis=0)
        
        # Simple periodicity check using autocorrelation peak
        def quick_periodicity(profile):
            if len(profile) < 10:
                return 0.0
            detrended = signal.detrend(profile)
            if np.std(detrended) == 0:
                return 0.0
            autocorr = np.correlate(detrended, detrended, mode='same')
            autocorr = autocorr[len(autocorr)//2:]
            if len(autocorr) > 3 and autocorr[0] != 0:
                return np.max(autocorr[1:]) / autocorr[0]
            return 0.0
        
        h_period = quick_periodicity(h_profile)
        v_period = quick_periodicity(v_profile)
        
        results['stripe'] = {
            'probability': max(h_period, v_period),
            'evidence': f"H: {h_period:.3f}, V: {v_period:.3f}"
        }
        
        # Fast banding detection
        h_var = np.var(gray, axis=1)
        v_var = np.var(gray, axis=0)
        
        h_banding = quick_periodicity(h_var) * 0.7
        v_banding = quick_periodicity(v_var) * 0.7
        
        results['banding'] = {
            'probability': max(h_banding, v_banding),
            'evidence': f"H: {h_banding:.3f}, V: {v_banding:.3f}"
        }
        
        return results
    
    def detect_motion_blur_fast(self, image_data):
        """Fast motion blur detection"""
        scale_key = 'medium' if 'medium' in image_data['scales'] else 'low'
        image = image_data['scales'][scale_key]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Fast edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Fast gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Motion blur reduces edge sharpness
        avg_gradient = np.mean(gradient_mag[edges > 0]) if np.any(edges > 0) else 0
        
        # Estimate blur based on edge density and gradient strength
        blur_score = 1 - min(edge_density * 10, 1.0) * min(avg_gradient / 50, 1.0)
        
        return {
            'motion_blur': {
                'probability': max(0, blur_score),
                'evidence': f"Edge density: {edge_density:.4f}, Avg gradient: {avg_gradient:.2f}"
            }
        }
    
    def parallel_analysis(self, image_data):
        """Run multiple analyses in parallel - no time limits"""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit tasks
            futures = {
                'additive': executor.submit(self.detect_additive_noise_fast, image_data),
                'frequency': executor.submit(self.analyze_frequency_fast, image_data),
                'patterns': executor.submit(self.detect_patterns_fast, image_data),
                'motion': executor.submit(self.detect_motion_blur_fast, image_data),
            }
            
            # Collect results - removed timeout, let each task complete naturally
            all_results = {}
            for task_name, future in futures.items():
                try:
                    result = future.result()  # No timeout - wait as long as needed
                    all_results.update(result)
                except Exception as e:
                    print(f"Task {task_name} failed: {e}")
                    continue
        
        return all_results
    
    def detect_noise_type_fast(self, image_path, verbose=False):
        """
        Fast noise detection for high-resolution images - no time limits
        
        Args:
            image_path: Path to the image
            verbose: Print timing information
            
        Returns:
            Dictionary with detected noise types and their probabilities
        """
        start_time = time.time()
        
        # Load and preprocess - no time limit
        if verbose:
            print(f"Loading and preprocessing image...")
        load_start = time.time()
        image_data = self.load_and_preprocess_image(image_path)
        load_time = time.time() - load_start
        
        # Quick noise estimation
        if verbose:
            print(f"Estimating noise level...")
        noise_start = time.time()
        noise_level = self.fast_noise_estimation(list(image_data['scales'].values())[0])
        noise_time = time.time() - noise_start
        
        # Parallel analysis - no time limits
        if verbose:
            print(f"Running parallel analysis...")
        analysis_start = time.time()
        
        if self.num_threads > 1:
            all_results = self.parallel_analysis(image_data)
        else:
            # Sequential analysis
            all_results = {}
            all_results.update(self.detect_additive_noise_fast(image_data))
            all_results.update(self.analyze_frequency_fast(image_data))
            all_results.update(self.detect_patterns_fast(image_data))
            all_results.update(self.detect_motion_blur_fast(image_data))
        
        analysis_time = time.time() - analysis_start
        
        # Quick ISO/sensor noise detection
        iso_score = min(noise_level / 30, 1.0)  # Simple heuristic
        all_results['iso_sensor'] = {
            'probability': iso_score,
            'evidence': f"Noise level: {noise_level:.2f}"
        }
        
        # Sort results
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTiming breakdown:")
            print(f"  Loading: {load_time:.2f}s")
            print(f"  Noise estimation: {noise_time:.2f}s")
            print(f"  Analysis: {analysis_time:.2f}s")
            print(f"  Total: {total_time:.2f}s")
            print(f"  Image size: {image_data['original_shape']}")
        
        return {
            'detected_noise_types': sorted_results,
            'most_likely': sorted_results[0] if sorted_results else None,
            'estimated_noise_level': noise_level,
            'processing_time': total_time,
            'image_size': image_data['original_shape']
        }
    
    def print_results(self, results):
        """Print formatted results with timing info"""
        print("=" * 70)
        print("FAST BLIND NOISE DETECTION RESULTS")
        print("=" * 70)
        
        print(f"Image size: {results['image_size']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Estimated noise level (σ): {results['estimated_noise_level']:.2f}")
        
        if results['most_likely']:
            print(f"\nMost likely noise type: {self.noise_types.get(results['most_likely'][0], results['most_likely'][0])}")
            print(f"Confidence: {results['most_likely'][1]['probability']:.3f}")
            print(f"Evidence: {results['most_likely'][1]['evidence']}")
        
        print("\nTop detected noise types:")
        print("-" * 50)
        
        for noise_type, data in results['detected_noise_types'][:5]:  # Show top 5
            if data['probability'] > 0.05:
                print(f"{self.noise_types.get(noise_type, noise_type):.<35} {data['probability']:.3f}")
                print(f"  {data['evidence']}")

# Convenience function for different speed modes
def create_detector(speed_mode='fast', image_size='large'):
    """
    Create detector with preset configurations
    
    Args:
        speed_mode: 'ultra_fast', 'fast', 'balanced', 'thorough'
        image_size: 'small', 'medium', 'large', 'huge'
    """
    
    configs = {
        'ultra_fast': {'max_size': 256, 'num_samples': 20, 'num_threads': 4, 'fast_mode': True},
        'fast': {'max_size': 512, 'num_samples': 30, 'num_threads': 4, 'fast_mode': True},
        'balanced': {'max_size': 1024, 'num_samples': 50, 'num_threads': 2, 'fast_mode': True},
        'thorough': {'max_size': 2048, 'num_samples': 100, 'num_threads': 2, 'fast_mode': False}
    }
    
    size_adjustments = {
        'small': {'max_size': lambda x: x * 2},      # Images < 1MP
        'medium': {'max_size': lambda x: x},         # Images 1-4MP  
        'large': {'max_size': lambda x: x},          # Images 4-16MP
        'huge': {'max_size': lambda x: x // 2}       # Images > 16MP
    }
    
    config = configs.get(speed_mode, configs['fast']).copy()
    
    # Adjust for image size
    if image_size in size_adjustments:
        for key, adjustment in size_adjustments[image_size].items():
            if key in config:
                config[key] = adjustment(config[key])
    
    return FastBlindNoiseDetector(**config)
            
            