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
    
    
    def create_processing_order(self, results, aggressive_processing=False):
        """
        Create intelligent processing order based on detection results
            
        Args:
            results: Results from detect_noise_type_fast()
            aggressive_processing: If True, process more noise types with lower confidence
            
        Returns:
            Processing plan with optimal order and parameters
        """
        order_generator = ProcessingOrderGenerator()
        processing_plan = order_generator.generate_processing_order(results, aggressive_processing)
        return processing_plan
        
    def print_processing_plan(self, plan):
        """Print formatted processing plan"""
        order_generator = ProcessingOrderGenerator()
        order_generator.print_processing_plan(plan)
            

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

class ProcessingOrderGenerator:
    """
    Determines optimal processing order and parameters for multi-noise denoising
    """
    
    def __init__(self):
        # Define noise type to module mapping
        self.noise_to_module = {
            # Module 1: Mathematical/Additive/Speckle
            'gaussian': 1,
            'uniform': 1,
            'speckle': 1,
            'white': 1,
            'poisson': 1,
            
            # Module 2: Impulse
            'salt_pepper': 2,
            'shot': 2,
            
            # Module 3: Frequency Domain
            'pink': 3,
            'blue': 3,
            'brown': 3,
            'quantization': 3,
            'compression': 3,
            
            # Module 4: Structured/Spatial
            'checkerboard': 4,
            'stripe': 4,
            'ring': 4,
            'banding': 4,
            'low_light': 4,
            
            # Module 5: Motion
            'motion_blur': 5,
            'vibration': 5,
            
            # Module 6: Camera Pipeline
            'iso_sensor': 6,
            'chromatic': 6
        }
        
        # Define processing priorities (lower number = process first)
        self.processing_priority = {
            2: 1,  # Impulse noise (can corrupt other analyses)
            6: 2,  # Camera pipeline (systematic corrections)
            4: 3,  # Structured noise (spatial patterns)
            5: 4,  # Motion blur (temporal effects)
            3: 5,  # Frequency domain (spectral cleaning)
            1: 6   # Additive noise (final cleanup)
        }
        
        # Define interference patterns
        self.interference_matrix = {
            # (module1, module2): interference_factor (0-1, higher = more interference)
            (1, 2): 0.3,  # Additive interferes moderately with impulse
            (1, 3): 0.2,  # Additive interferes slightly with frequency
            (1, 4): 0.4,  # Additive interferes with structured
            (1, 5): 0.1,  # Additive barely interferes with motion
            (1, 6): 0.2,  # Additive interferes slightly with camera
            
            (2, 3): 0.8,  # Impulse strongly interferes with frequency
            (2, 4): 0.6,  # Impulse moderately interferes with structured
            (2, 5): 0.4,  # Impulse interferes with motion
            (2, 6): 0.3,  # Impulse interferes with camera
            
            (3, 4): 0.5,  # Frequency moderately interferes with structured
            (3, 5): 0.3,  # Frequency interferes with motion
            (3, 6): 0.2,  # Frequency slightly interferes with camera
            
            (4, 5): 0.7,  # Structured strongly interferes with motion
            (4, 6): 0.4,  # Structured interferes with camera
            
            (5, 6): 0.3   # Motion interferes with camera
        }
        
        # Confidence thresholds for processing
        self.confidence_thresholds = {
            1: 0.05,  # Additive (always some additive noise)
            2: 0.20,  # Impulse (clear threshold needed)
            3: 0.10,  # Frequency (moderate threshold)
            4: 0.30,  # Structured (clear patterns needed)
            5: 0.35,  # Motion (clear motion needed)
            6: 0.08   # Camera (often present)
        }
    
    def generate_processing_order(self, detection_results, aggressive_processing=False):
        """
        Generate optimal processing order based on detected noise types
        
        Args:
            detection_results: Results from detect_noise_type_fast()
            aggressive_processing: If True, process more noise types with lower confidence
        
        Returns:
            Dictionary with processing order, parameters, and confidence levels
        """
        detected_noise_types = detection_results['detected_noise_types']
        
        # Step 1: Filter noise types by confidence threshold
        significant_noise = self._filter_by_confidence(
            detected_noise_types, aggressive_processing
        )
        
        # Step 2: Map noise types to modules
        module_assignments = self._map_noise_to_modules(significant_noise)
        
        # Step 3: Calculate interference-aware processing order
        processing_order = self._calculate_optimal_order(module_assignments)
        
        # Step 4: Determine processing parameters
        processing_params = self._determine_parameters(module_assignments)
        
        # Step 5: Create final processing plan
        processing_plan = self._create_processing_plan(
            processing_order, processing_params, module_assignments
        )
        
        return processing_plan
    
    def _filter_by_confidence(self, detected_noise_types, aggressive_processing):
        """Filter noise types based on confidence thresholds"""
        significant_noise = {}
        
        # Adjust thresholds based on processing mode
        threshold_multiplier = 0.7 if aggressive_processing else 1.0
        
        for noise_type, data in detected_noise_types:
            confidence = data['probability']
            
            # Get module for this noise type
            module_id = self.noise_to_module.get(noise_type)
            if module_id is None:
                continue
                
            # Check against threshold
            threshold = self.confidence_thresholds.get(module_id, 0.1) * threshold_multiplier
            
            if confidence >= threshold:
                if module_id not in significant_noise:
                    significant_noise[module_id] = []
                
                significant_noise[module_id].append({
                    'noise_type': noise_type,
                    'confidence': confidence,
                    'evidence': data['evidence']
                })
        
        return significant_noise
    
    def _map_noise_to_modules(self, significant_noise):
        """Create module assignments with aggregated confidence"""
        module_assignments = {}
        
        for module_id, noise_list in significant_noise.items():
            # Calculate combined confidence for the module
            confidences = [item['confidence'] for item in noise_list]
            
            # Use weighted average, giving more weight to higher confidences
            weights = np.array(confidences)
            combined_confidence = np.average(confidences, weights=weights)
            
            # Boost confidence if multiple noise types point to same module
            if len(noise_list) > 1:
                combined_confidence = min(1.0, combined_confidence * (1 + 0.1 * len(noise_list)))
            
            module_assignments[module_id] = {
                'combined_confidence': combined_confidence,
                'noise_types': noise_list,
                'processing_strength': self._calculate_processing_strength(combined_confidence)
            }
        
        return module_assignments
    
    def _calculate_optimal_order(self, module_assignments):
        """Calculate optimal processing order considering interference"""
        if not module_assignments:
            return []
        
        modules = list(module_assignments.keys())
        
        # Start with priority-based order
        initial_order = sorted(modules, key=lambda x: self.processing_priority.get(x, 999))
        
        # Optimize order based on interference patterns
        optimized_order = self._optimize_for_interference(initial_order, module_assignments)
        
        return optimized_order
    
    def _optimize_for_interference(self, initial_order, module_assignments):
        """Optimize order to minimize interference between modules"""
        if len(initial_order) <= 1:
            return initial_order
        
        # Calculate total interference cost for current order
        def calculate_interference_cost(order):
            total_cost = 0
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    module1, module2 = order[i], order[j]
                    
                    # Get interference factor
                    interference = self.interference_matrix.get((module1, module2), 0)
                    if interference == 0:
                        interference = self.interference_matrix.get((module2, module1), 0)
                    
                    # Weight by confidences
                    conf1 = module_assignments[module1]['combined_confidence']
                    conf2 = module_assignments[module2]['combined_confidence']
                    
                    # Cost is higher when high-confidence modules interfere
                    # and when interfering modules are processed close to each other
                    position_penalty = 1.0 / (j - i)  # Closer modules have higher penalty
                    total_cost += interference * conf1 * conf2 * position_penalty
            
            return total_cost
        
        # Try different permutations to find better order
        best_order = initial_order
        best_cost = calculate_interference_cost(initial_order)
        
        # Use simple local search for small numbers of modules
        if len(initial_order) <= 6:
            from itertools import permutations
            for perm in permutations(initial_order):
                cost = calculate_interference_cost(list(perm))
                if cost < best_cost:
                    best_cost = cost
                    best_order = list(perm)
        else:
            # For larger sets, use greedy improvement
            current_order = initial_order.copy()
            improved = True
            
            while improved:
                improved = False
                for i in range(len(current_order) - 1):
                    # Try swapping adjacent modules
                    test_order = current_order.copy()
                    test_order[i], test_order[i + 1] = test_order[i + 1], test_order[i]
                    
                    cost = calculate_interference_cost(test_order)
                    if cost < best_cost:
                        best_cost = cost
                        best_order = test_order.copy()
                        current_order = test_order.copy()
                        improved = True
        
        return best_order
    
    def _calculate_processing_strength(self, confidence):
        """Calculate processing strength based on confidence"""
        if confidence >= 0.8:
            return 'strong'
        elif confidence >= 0.5:
            return 'moderate' 
        elif confidence >= 0.2:
            return 'gentle'
        else:
            return 'minimal'
    
    def _determine_parameters(self, module_assignments):
        """Determine processing parameters for each module"""
        parameters = {}
        
        for module_id, assignment in module_assignments.items():
            confidence = assignment['combined_confidence']
            strength = assignment['processing_strength']
            
            # Base parameters for each module
            if module_id == 1:  # Mathematical/Additive/Speckle
                parameters[module_id] = {
                    'iterations': 3 if strength == 'strong' else (2 if strength == 'moderate' else 1),
                    'sigma_noise': confidence * 10,
                    'preservation_factor': 1.0 - confidence * 0.3
                }
            
            elif module_id == 2:  # Impulse
                parameters[module_id] = {
                    'kernel_size': 5 if strength == 'strong' else 3,
                    'threshold_factor': confidence,
                    'iterations': 2 if strength == 'strong' else 1
                }
            
            elif module_id == 3:  # Frequency Domain
                parameters[module_id] = {
                    'cutoff_frequency': 0.1 * (1 + confidence),
                    'filter_strength': confidence,
                    'preserve_edges': strength != 'strong'
                }
            
            elif module_id == 4:  # Structured/Spatial
                parameters[module_id] = {
                    'lambda_tv': confidence * 0.1,
                    'lambda_structure': confidence * 0.05,
                    'decomposition_levels': 3 if strength == 'strong' else 2
                }
            
            elif module_id == 5:  # Motion
                parameters[module_id] = {
                    'temporal_window': 5 if strength == 'strong' else 3,
                    'motion_threshold': 2.0 * (1 - confidence * 0.5),
                    'enable_temporal_consistency': strength in ['strong', 'moderate']
                }
            
            elif module_id == 6:  # Camera Pipeline
                parameters[module_id] = {
                    'noise_model_accuracy': confidence,
                    'pipeline_correction_strength': strength,
                    'color_noise_reduction': confidence > 0.3
                }
        
        return parameters
    
    def _create_processing_plan(self, processing_order, processing_params, module_assignments):
        """Create final processing plan"""
        plan = {
            'processing_order': processing_order,
            'total_modules': len(processing_order),
            'estimated_processing_time': self._estimate_processing_time(processing_order, processing_params),
            'modules': {}
        }
        
        for i, module_id in enumerate(processing_order):
            assignment = module_assignments[module_id]
            params = processing_params.get(module_id, {})
            
            plan['modules'][module_id] = {
                'order': i + 1,
                'module_name': self._get_module_name(module_id),
                'confidence': assignment['combined_confidence'],
                'strength': assignment['processing_strength'],
                'detected_noise_types': [n['noise_type'] for n in assignment['noise_types']],
                'parameters': params,
                'reasoning': self._generate_reasoning(module_id, assignment)
            }
        
        return plan
    
    def _get_module_name(self, module_id):
        """Get human-readable module name"""
        names = {
            1: "Mathematical/Additive/Speckle Denoiser",
            2: "Impulse Denoiser", 
            3: "Frequency Domain Denoiser",
            4: "Structured/Spatial Denoiser",
            5: "Motion Denoiser",
            6: "Camera Pipeline Denoiser"
        }
        return names.get(module_id, f"Module {module_id}")
    
    def _generate_reasoning(self, module_id, assignment):
        """Generate human-readable reasoning for module selection"""
        noise_types = [n['noise_type'] for n in assignment['noise_types']]
        confidence = assignment['combined_confidence']
        
        reasoning = f"Selected due to detection of {', '.join(noise_types)} "
        reasoning += f"with combined confidence of {confidence:.3f}. "
        
        if len(noise_types) > 1:
            reasoning += f"Multiple related noise types ({len(noise_types)}) indicate this module is essential."
        else:
            reasoning += "Single strong detection indicates targeted processing needed."
        
        return reasoning
    
    def _estimate_processing_time(self, processing_order, processing_params):
        """Estimate total processing time"""
        # Base processing times per module (in relative units)
        base_times = {1: 1.0, 2: 0.5, 3: 1.5, 4: 2.0, 5: 2.5, 6: 1.2}
        
        total_time = 0
        for module_id in processing_order:
            base_time = base_times.get(module_id, 1.0)
            params = processing_params.get(module_id, {})
            
            # Adjust for processing strength
            strength_multiplier = {
                'minimal': 0.5, 'gentle': 0.7, 'moderate': 1.0, 'strong': 1.5
            }
            
            if module_id in processing_params:
                # Get strength from parameters (this would need to be calculated)
                multiplier = strength_multiplier.get('moderate', 1.0)  # Default
                total_time += base_time * multiplier
            else:
                total_time += base_time
        
        return total_time
    
    def print_processing_plan(self, plan):
        """Print formatted processing plan"""
        print("=" * 70)
        print("MULTI-NOISE DENOISING PROCESSING PLAN")
        print("=" * 70)
        
        print(f"Total modules to process: {plan['total_modules']}")
        print(f"Estimated relative processing time: {plan['estimated_processing_time']:.1f}")
        print()
        
        print("Processing Order:")
        print("-" * 50)
        
        for module_id in plan['processing_order']:
            module_info = plan['modules'][module_id]
            
            print(f"{module_info['order']}. {module_info['module_name']}")
            print(f"   Confidence: {module_info['confidence']:.3f} ({module_info['strength']} processing)")
            print(f"   Detected: {', '.join(module_info['detected_noise_types'])}")
            print(f"   Reasoning: {module_info['reasoning']}")
            
            # Print key parameters
            if module_info['parameters']:
                print("   Key Parameters:")
                for key, value in list(module_info['parameters'].items())[:3]:  # Show top 3
                    print(f"     - {key}: {value}")
            print()


# Example usage
def test_fast_detector():
    """Test function with different speed modes"""
    print("Fast Blind Noise Detector - No Time Limits")
    print("=" * 60)
    
    print("\nAvailable speed modes:")
    print("• ultra_fast: Fastest processing with basic accuracy")
    print("• fast:       Good balance of speed and accuracy") 
    print("• balanced:   Better accuracy, moderate speed")
    print("• thorough:   Best accuracy, slower processing")
    
    print("\nUsage examples:")
    print("# For huge images (>16MP), ultra-fast processing:")
    print("detector = create_detector('ultra_fast', 'huge')")
    print("results = detector.detect_noise_type_fast('huge_image.tiff', verbose=True)")
    
    print("\n# For medium images, balanced processing:")
    print("detector = create_detector('balanced', 'medium')")
    print("results = detector.detect_noise_type_fast('medium_image.jpg')")
    
    print("\n# Manual configuration:")
    print("detector = FastBlindNoiseDetector(max_size=512, num_samples=30, num_threads=4)")
    print("results = detector.detect_noise_type_fast('image.jpg')")
    
    print("\nOptimization techniques used:")
    print("✓ Multi-scale image pyramids")
    print("✓ ROI sampling instead of full-image analysis") 
    print("✓ Parallel processing with no time limits")
    print("✓ Fast approximation algorithms")
    print("✓ Early termination strategies")
    print("✓ Memory-efficient operations")
    print("\n✓ NO TIME LIMITS - processes until complete")
    
    detector = create_detector('ultra_fast', 'huge')
    results = detector.detect_noise_type_fast('photo_2025-08-13_10-38-14.jpg')
    detector.print_results(results)

    # Generate processing order
    processing_plan = detector.create_processing_order(results, aggressive_processing=True)

    # Print the plan
    detector.print_processing_plan(processing_plan)


# if __name__ == "__main__":
#     test_fast_detector()
    
    