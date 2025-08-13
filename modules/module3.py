"""
Module 3: frequency denoiser

Homomorphic Filtering method is better
"""

import numpy as np
import cv2
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

class NonLinearTransforms:
    """Advanced transform implementations for non-linear frequency domain processing"""
    
    def __init__(self):
        self.transforms = {}
    
    def fractional_fourier_transform(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        """
        Implement Fractional Fourier Transform
        
        Args:
            signal: Input signal (1D array)
            alpha: Fractional parameter (0-4)
        
        Returns:
            Fractionally transformed signal
        """
        N = len(signal)
        
        # Special cases for computational efficiency
        if alpha == 0:
            return signal
        elif alpha == 1:
            return np.fft.fft(signal)
        elif alpha == 2:
            return signal[::-1]
        elif alpha == 3:
            return np.fft.ifft(signal)
        else:
            # General fractional case
            phi = alpha * np.pi / 2
            
            if abs(np.sin(phi)) < 1e-10:  # Near special cases
                return signal
            
            # Create fractional FT matrix
            n = np.arange(N)
            k = np.arange(N)
            
            # Chirp function with numerical stability
            cot_phi = np.cos(phi) / (np.sin(phi) + 1e-10)
            chirp_n = np.exp(-1j * np.pi * cot_phi * n**2 / N)
            
            # Kernel matrix
            kernel = np.exp(1j * 2 * np.pi * np.outer(n, k) / (N * np.sin(phi)))
            
            # Apply transformation
            result = np.zeros(N, dtype=complex)
            for i in range(N):
                result[i] = np.sum(signal * chirp_n * kernel[i, :])
            
            # Normalization
            result = result * np.sqrt(np.abs(np.sin(phi))) / np.sqrt(N)
            
            return result
    
    def chirp_z_transform(self, signal: np.ndarray, M: int, W: complex, A: complex) -> np.ndarray:
        """
        Chirp-Z Transform for frequency zoom
        
        Args:
            signal: Input signal
            M: Number of output samples
            W: Complex frequency spacing
            A: Starting complex frequency
        
        Returns:
            CZT of input signal
        """
        N = len(signal)
        
        # Create chirp sequences
        n = np.arange(N)
        m = np.arange(M)
        
        # Chirp filter
        h = W ** (-n**2 / 2.0)
        
        # Input sequence preparation
        x_chirp = signal * A**(-n) * W**(n**2 / 2.0)
        
        # Convolution using FFT
        L = 2**int(np.ceil(np.log2(N + M - 1)))
        h_pad = np.zeros(L, dtype=complex)
        x_pad = np.zeros(L, dtype=complex)
        
        h_pad[:N] = h
        x_pad[:N] = x_chirp
        
        # FFT convolution
        H = np.fft.fft(h_pad)
        X = np.fft.fft(x_pad)
        Y = np.fft.ifft(H * X)
        
        # Extract result and apply final chirp
        result = Y[:M] * W**(m**2 / 2.0)
        
        return result
    
    def wigner_ville_distribution(self, signal: np.ndarray) -> np.ndarray:
        """
        Wigner-Ville Distribution for time-frequency analysis
        
        Args:
            signal: Input signal
        
        Returns:
            WVD matrix
        """
        N = len(signal)
        WVD = np.zeros((N, N), dtype=complex)
        
        for t in range(N):
            for f in range(N):
                sum_val = 0
                for tau in range(-N//2, N//2):
                    t_plus = t + tau
                    t_minus = t - tau
                    
                    if 0 <= t_plus < N and 0 <= t_minus < N:
                        sum_val += (signal[t_plus] * 
                                  np.conj(signal[t_minus]) * 
                                  np.exp(-2j * np.pi * f * tau / N))
                
                WVD[t, f] = sum_val
        
        return np.real(WVD)


class SpectralAnalyzer:
    """Advanced spectral analysis and peak detection"""
    
    def __init__(self, n_harmonics: int = 10):
        self.n_harmonics = n_harmonics
    
    def detect_spectral_peaks(self, spectrum: np.ndarray, prominence: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect spectral peaks and harmonics
        
        Args:
            spectrum: Input spectrum
            prominence: Minimum prominence for peak detection
        
        Returns:
            Tuple of (peaks, prominences)
        """
        magnitude_spectrum = np.abs(spectrum)
        
        # Find peaks
        peaks, properties = find_peaks(magnitude_spectrum, 
                                     prominence=prominence * np.max(magnitude_spectrum))
        
        if len(peaks) == 0:
            return np.array([]), np.array([])
        
        # Compute peak prominences
        prominences = peak_prominences(magnitude_spectrum, peaks)[0]
        
        # Sort by prominence and take top harmonics
        sorted_indices = np.argsort(prominences)[::-1]
        n_peaks = min(self.n_harmonics, len(peaks))
        significant_peaks = peaks[sorted_indices[:n_peaks]]
        significant_prominences = prominences[sorted_indices[:n_peaks]]
        
        return significant_peaks, significant_prominences
    
    def estimate_noise_floor(self, spectrum: np.ndarray, percentile: float = 10) -> float:
        """
        Estimate noise floor in spectrum
        
        Args:
            spectrum: Input spectrum
            percentile: Percentile for noise floor estimation
        
        Returns:
            Estimated noise floor
        """
        magnitude_spectrum = np.abs(spectrum)
        noise_floor = np.percentile(magnitude_spectrum, percentile)
        
        # Adaptive noise floor using robust statistics
        median_mag = np.median(magnitude_spectrum)
        mad = np.median(np.abs(magnitude_spectrum - median_mag))
        adaptive_floor = median_mag - 2.5 * mad
        
        return max(noise_floor, adaptive_floor, 0)
    
    def spectral_envelope_estimation(self, spectrum: np.ndarray, window_size: int = 64) -> np.ndarray:
        """
        Estimate spectral envelope using cepstral analysis
        
        Args:
            spectrum: Input spectrum
            window_size: Cepstral window size
        
        Returns:
            Spectral envelope
        """
        # Cepstral analysis
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        
        # Lifter to extract envelope
        liftered_cepstrum = cepstrum.copy()
        liftered_cepstrum[window_size:] = 0
        
        # Back to frequency domain
        envelope = np.exp(np.fft.fft(liftered_cepstrum).real)
        
        return envelope


class AdvancedSpectralProcessor:
    """Advanced spectral processing techniques"""
    
    def __init__(self):
        self.processor_methods = {}
    
    def adaptive_spectral_subtraction(self, noisy_spectrum: np.ndarray, 
                                    noise_spectrum: np.ndarray,
                                    alpha: float = 2.0, 
                                    beta: float = 0.01) -> np.ndarray:
        """
        Advanced spectral subtraction with over-subtraction
        
        Args:
            noisy_spectrum: Noisy signal spectrum
            noise_spectrum: Noise spectrum estimate
            alpha: Over-subtraction factor
            beta: Spectral floor factor
        
        Returns:
            Enhanced spectrum
        """
        # Magnitude and phase separation
        noisy_magnitude = np.abs(noisy_spectrum)
        noisy_phase = np.angle(noisy_spectrum)
        noise_magnitude = np.abs(noise_spectrum)
        
        # Adaptive over-subtraction factor
        snr_estimate = noisy_magnitude / (noise_magnitude + 1e-10)
        adaptive_alpha = alpha * (1 + np.exp(-snr_estimate))
        
        # Spectral subtraction
        enhanced_magnitude = noisy_magnitude - adaptive_alpha * noise_magnitude
        
        # Apply spectral floor (beta rule)
        spectral_floor = beta * noisy_magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)
        
        # Reconstruct complex spectrum
        enhanced_spectrum = enhanced_magnitude * np.exp(1j * noisy_phase)
        
        return enhanced_spectrum
    
    def multi_band_spectral_gating(self, spectrum: np.ndarray, n_bands: int = 8) -> np.ndarray:
        """
        Multi-band spectral gating
        
        Args:
            spectrum: Input spectrum
            n_bands: Number of frequency bands
        
        Returns:
            Gated spectrum
        """
        N = len(spectrum)
        band_size = N // n_bands
        processed_spectrum = spectrum.copy()
        
        for band in range(n_bands):
            start_idx = band * band_size
            end_idx = min((band + 1) * band_size, N)
            
            band_spectrum = spectrum[start_idx:end_idx]
            band_magnitude = np.abs(band_spectrum)
            
            if len(band_magnitude) == 0:
                continue
            
            # Adaptive threshold for this band
            threshold = np.percentile(band_magnitude, 30) + \
                       0.1 * np.std(band_magnitude)
            
            # Apply gating
            gate = (band_magnitude > threshold).astype(float)
            gate_smooth = self._smooth_gate(gate, kernel_size=5)
            
            processed_spectrum[start_idx:end_idx] = band_spectrum * gate_smooth
        
        return processed_spectrum
    
    def _smooth_gate(self, gate: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Smooth gating function"""
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(gate, kernel, mode='same')
    
    def homomorphic_filtering(self, image: np.ndarray, 
                            cutoff_frequency: float = 0.1,
                            gamma_low: float = 0.25, 
                            gamma_high: float = 2.0) -> np.ndarray:
        """
        Homomorphic filtering for illumination correction
        
        Args:
            image: Input image (2D array)
            cutoff_frequency: Filter cutoff frequency
            gamma_low: Low frequency gain
            gamma_high: High frequency gain
        
        Returns:
            Filtered image
        """
        # Ensure positive values for log transform
        image_float = image.astype(np.float64) + 1
        
        # Log transform
        log_image = np.log(image_float)
        
        # Frequency domain
        fft_log = np.fft.fft2(log_image)
        fft_shifted = np.fft.fftshift(fft_log)
        
        # Create high-pass filter
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        
        # Distance matrix
        u = np.arange(rows) - center_row
        v = np.arange(cols) - center_col
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        
        # Butterworth high-pass filter
        n = 2  # Filter order
        D0 = cutoff_frequency * min(rows, cols)
        H = 1 / (1 + (D0 / (D + 1e-10))**(2*n))
        
        # Apply frequency-dependent gain
        H_homomorphic = (gamma_high - gamma_low) * H + gamma_low
        
        # Apply filter
        filtered_fft = fft_shifted * H_homomorphic
        
        # Inverse transform
        filtered_fft = np.fft.ifftshift(filtered_fft)
        filtered_log = np.fft.ifft2(filtered_fft).real
        
        # Exponential to get back to image domain
        result = np.exp(filtered_log) - 1
        
        return np.clip(result, 0, 255).astype(np.uint8)


class NonLinearFrequencyProcessor:
    """Main class for non-linear frequency domain processing of RGB images"""
    
    def __init__(self):
        self.transforms = NonLinearTransforms()
        self.analyzer = SpectralAnalyzer()
        self.processor = AdvancedSpectralProcessor()
        
    def load_rgb_image(self, image_path: str) -> np.ndarray:
        """Load RGB image from file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def rgb_to_channels(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split RGB image into separate channels"""
        return rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    
    def channels_to_rgb(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Combine channels back to RGB image"""
        return np.stack([r, g, b], axis=2)
    
    def process_channel_1d(self, channel: np.ndarray, method: str = 'fft', **kwargs) -> np.ndarray:
        """Process single channel using 1D transforms (applied row-wise)"""
        rows, cols = channel.shape
        processed_channel = np.zeros_like(channel, dtype=complex)
        
        for i in range(rows):
            row_data = channel[i, :].astype(complex)
            
            if method == 'fft':
                processed_row = np.fft.fft(row_data)
            elif method == 'fractional_fft':
                alpha = kwargs.get('alpha', 0.5)
                processed_row = self.transforms.fractional_fourier_transform(row_data, alpha)
            elif method == 'chirp_z':
                M = kwargs.get('M', cols)
                W = kwargs.get('W', np.exp(-2j * np.pi / cols))
                A = kwargs.get('A', 1.0)
                processed_row = self.transforms.chirp_z_transform(row_data, M, W, A)
            else:
                processed_row = np.fft.fft(row_data)
            
            # Ensure output has same length as input
            if len(processed_row) != cols:
                processed_row = np.fft.fft(row_data)
            
            processed_channel[i, :] = processed_row
        
        return processed_channel
    
    def process_channel_2d(self, channel: np.ndarray, method: str = 'homomorphic', **kwargs) -> np.ndarray:
        """Process single channel using 2D transforms"""
        if method == 'homomorphic':
            return self.processor.homomorphic_filtering(channel, **kwargs)
        elif method == 'spectral_enhancement':
            # Apply 2D FFT
            fft_channel = np.fft.fft2(channel)
            
            # Enhance using spectral analysis
            enhanced_fft = self._enhance_2d_spectrum(fft_channel, **kwargs)
            
            # Inverse FFT
            enhanced_channel = np.fft.ifft2(enhanced_fft).real
            return np.clip(enhanced_channel, 0, 255).astype(np.uint8)
        else:
            return channel
    
    def _enhance_2d_spectrum(self, spectrum_2d: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance 2D spectrum using advanced techniques"""
        # Apply enhancement row-wise
        rows, cols = spectrum_2d.shape
        enhanced_spectrum = np.zeros_like(spectrum_2d)
        
        for i in range(rows):
            row_spectrum = spectrum_2d[i, :]
            
            # Detect peaks and estimate noise
            peaks, prominences = self.analyzer.detect_spectral_peaks(row_spectrum)
            noise_floor = self.analyzer.estimate_noise_floor(row_spectrum)
            
            # Apply multi-band gating
            enhanced_row = self.processor.multi_band_spectral_gating(row_spectrum)
            
            enhanced_spectrum[i, :] = enhanced_row
        
        return enhanced_spectrum
    
    def process_rgb_image(self, rgb_image: np.ndarray, 
                         method: str = 'homomorphic',
                         process_channels_separately: bool = True,
                         **kwargs) -> np.ndarray:
        """
        Process RGB image using non-linear frequency domain techniques
        
        Args:
            rgb_image: Input RGB image
            method: Processing method ('homomorphic', 'fractional_fft', 'spectral_enhancement')
            process_channels_separately: Whether to process each channel separately
            **kwargs: Additional parameters for the processing method
        
        Returns:
            Processed RGB image
        """
        if process_channels_separately:
            # Split into channels
            r_channel, g_channel, b_channel = self.rgb_to_channels(rgb_image)
            
            # Process each channel
            if method in ['homomorphic', 'spectral_enhancement']:
                # 2D processing
                r_processed = self.process_channel_2d(r_channel, method, **kwargs)
                g_processed = self.process_channel_2d(g_channel, method, **kwargs)
                b_processed = self.process_channel_2d(b_channel, method, **kwargs)
            else:
                # 1D processing (row-wise)
                r_processed = np.real(self.process_channel_1d(r_channel, method, **kwargs))
                g_processed = np.real(self.process_channel_1d(g_channel, method, **kwargs))
                b_processed = np.real(self.process_channel_1d(b_channel, method, **kwargs))
                
                # Clip and convert to uint8
                r_processed = np.clip(r_processed, 0, 255).astype(np.uint8)
                g_processed = np.clip(g_processed, 0, 255).astype(np.uint8)
                b_processed = np.clip(b_processed, 0, 255).astype(np.uint8)
            
            # Combine channels
            processed_image = self.channels_to_rgb(r_processed, g_processed, b_processed)
        else:
            # Process as grayscale then apply to all channels
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            if method in ['homomorphic', 'spectral_enhancement']:
                processed_gray = self.process_channel_2d(gray_image, method, **kwargs)
            else:
                processed_gray = np.real(self.process_channel_1d(gray_image, method, **kwargs))
                processed_gray = np.clip(processed_gray, 0, 255).astype(np.uint8)
            
            # Apply the same processing to all channels
            processed_image = np.stack([processed_gray, processed_gray, processed_gray], axis=2)
        
        return processed_image
    
    def analyze_image_spectrum(self, rgb_image: np.ndarray) -> dict:
        """
        Analyze the spectral properties of an RGB image
        
        Args:
            rgb_image: Input RGB image
        
        Returns:
            Dictionary containing spectral analysis results
        """
        results = {}
        
        # Process each channel
        r_channel, g_channel, b_channel = self.rgb_to_channels(rgb_image)
        channels = {'red': r_channel, 'green': g_channel, 'blue': b_channel}
        
        for channel_name, channel_data in channels.items():
            # 2D FFT
            fft_2d = np.fft.fft2(channel_data)
            
            # Analyze first row as example
            row_spectrum = fft_2d[0, :]
            
            # Detect peaks
            peaks, prominences = self.analyzer.detect_spectral_peaks(row_spectrum)
            
            # Estimate noise floor
            noise_floor = self.analyzer.estimate_noise_floor(row_spectrum)
            
            # Spectral envelope
            envelope = self.analyzer.spectral_envelope_estimation(row_spectrum)
            
            results[channel_name] = {
                'peaks': peaks,
                'prominences': prominences,
                'noise_floor': noise_floor,
                'spectral_envelope': envelope,
                'spectrum_magnitude': np.abs(row_spectrum)
            }
        
        return results
    
    # def visualize_results(self, original_image: np.ndarray, 
    #                      processed_image: np.ndarray,
    #                      analysis_results: Optional[dict] = None):
    #     """
    #     Visualize processing results
        
    #     Args:
    #         original_image: Original RGB image
    #         processed_image: Processed RGB image
    #         analysis_results: Optional spectral analysis results
    #     """
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
    #     # Original image
    #     axes[0, 0].imshow(original_image)
    #     axes[0, 0].set_title('Original Image')
    #     axes[0, 0].axis('off')
        
    #     # Processed image
    #     axes[0, 1].imshow(processed_image)
    #     axes[0, 1].set_title('Processed Image')
    #     axes[0, 1].axis('off')
        
    #     # Difference image
    #     diff_image = np.abs(processed_image.astype(float) - original_image.astype(float))
    #     axes[1, 0].imshow(diff_image / 255, cmap='hot')
    #     axes[1, 0].set_title('Difference Image')
    #     axes[1, 0].axis('off')
        
    #     # Spectral analysis (if available)
    #     if analysis_results:
    #         # Plot spectrum for red channel as example
    #         red_spectrum = analysis_results['red']['spectrum_magnitude']
    #         axes[1, 1].plot(red_spectrum)
    #         axes[1, 1].set_title('Red Channel Spectrum')
    #         axes[1, 1].set_xlabel('Frequency Bin')
    #         axes[1, 1].set_ylabel('Magnitude')
    #     else:
    #         axes[1, 1].axis('off')
        
    #     plt.tight_layout()
    #     plt.show()


# # Example usage
# if __name__ == "__main__":
#     # Initialize processor
#     processor = NonLinearFrequencyProcessor()
    
#     # Create a sample RGB image for testing
#     height, width = 256, 256
#     sample_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
#     # Add some structured patterns for more interesting results
#     x, y = np.meshgrid(np.arange(width), np.arange(height))
#     pattern = (np.sin(2 * np.pi * x / 32) + np.cos(2 * np.pi * y / 32)) * 50 + 128
#     pattern = np.clip(pattern, 0, 255).astype(np.uint8)
    
#     for i in range(3):
#         sample_image[:, :, i] = pattern
    
#     print("Processing sample RGB image...")
    
#     # Process using different methods
#     methods_to_test = ['homomorphic', 'fractional_fft', 'spectral_enhancement']
    
#     for method in methods_to_test:
#         print(f"Testing method: {method}")
        
#         if method == 'fractional_fft':
#             processed = processor.process_rgb_image(sample_image, method=method, alpha=0.5)
#         elif method == 'homomorphic':
#             processed = processor.process_rgb_image(sample_image, method=method, 
#                                                   cutoff_frequency=0.1, 
#                                                   gamma_low=0.3, gamma_high=2.0)
#         else:
#             processed = processor.process_rgb_image(sample_image, method=method)
        
#         print(f"Original image shape: {sample_image.shape}")
#         print(f"Processed image shape: {processed.shape}")
#         print(f"Original image range: [{sample_image.min()}, {sample_image.max()}]")
#         print(f"Processed image range: [{processed.min()}, {processed.max()}]")
#         print("-" * 50)
    
#     # Analyze spectral properties
#     print("Analyzing spectral properties...")
#     analysis = processor.analyze_image_spectrum(sample_image)
    
#     for channel in ['red', 'green', 'blue']:
#         peaks = analysis[channel]['peaks']
#         noise_floor = analysis[channel]['noise_floor']
#         print(f"{channel.capitalize()} channel - Peaks found: {len(peaks)}, Noise floor: {noise_floor:.2f}")
    
#     print("Processing complete!")
