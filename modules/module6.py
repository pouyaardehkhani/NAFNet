"""
Module 6: camera denoiser
"""

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, gaussian_filter
import json

class CameraNoiseModeler:
    """Models various noise sources in camera sensors"""
    
    def __init__(self):
        self.noise_models = {
            'shot_noise': self.model_shot_noise,
            'read_noise': self.model_read_noise,
            'dark_current': self.model_dark_current_noise,
            'fixed_pattern': self.model_fixed_pattern_noise
        }
    
    def comprehensive_noise_analysis(self, image, metadata=None):
        """Analyze all camera noise sources"""
        
        noise_components = {}
        
        # Extract camera metadata if available
        if metadata:
            iso = metadata.get('iso', 100)
            exposure_time = metadata.get('exposure_time', 1/60)
            temperature = metadata.get('temperature', 25)  # Celsius
        else:
            # Estimate from image statistics
            iso = self.estimate_iso(image)
            exposure_time = self.estimate_exposure_time(image)
            temperature = 25  # Default
        
        # Model shot noise (Poisson)
        shot_noise_variance = self.model_shot_noise(image, iso)
        noise_components['shot_noise'] = shot_noise_variance
        
        # Model read noise (Gaussian)
        read_noise_variance = self.model_read_noise(iso)
        noise_components['read_noise'] = read_noise_variance
        
        # Model dark current noise (temperature dependent)
        dark_current_variance = self.model_dark_current_noise(
            exposure_time, temperature
        )
        noise_components['dark_current'] = dark_current_variance
        
        # Model fixed pattern noise
        fpn_map = self.model_fixed_pattern_noise(image)
        noise_components['fixed_pattern'] = fpn_map
        
        return noise_components
    
    def model_shot_noise(self, image, iso):
        """Model Poisson shot noise"""
        # Shot noise variance proportional to signal
        gain_factor = iso / 100.0  # Normalized to ISO 100
        
        # Convert to grayscale if RGB for noise estimation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        shot_noise_variance = np.maximum(gray.astype(np.float64) * gain_factor, 1.0)
        return shot_noise_variance
    
    def model_read_noise(self, iso):
        """Model Gaussian read noise"""
        # Typical read noise increases with gain (ISO)
        base_read_noise = 2.0  # electrons
        gain_factor = np.sqrt(iso / 100.0)
        read_noise_std = base_read_noise * gain_factor
        return read_noise_std**2
    
    def model_dark_current_noise(self, exposure_time, temperature):
        """Model dark current noise (temperature and exposure dependent)"""
        # Dark current doubles every 6-8°C
        temp_factor = 2.0**((temperature - 20) / 7.0)
        dark_current_rate = 0.1 * temp_factor  # electrons per second per pixel
        dark_current_variance = dark_current_rate * exposure_time
        return dark_current_variance
    
    def model_fixed_pattern_noise(self, image):
        """Model fixed pattern noise"""
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Use median filtering to estimate FPN
        median_filtered = cv2.medianBlur(gray.astype(np.uint8), 5)
        fpn_estimate = gray.astype(np.float64) - median_filtered.astype(np.float64)
        
        # Smooth FPN estimate
        fpn_map = cv2.GaussianBlur(fpn_estimate, (15, 15), 5)
        
        return fpn_map
    
    def estimate_iso(self, image):
        """Estimate ISO from image statistics"""
        # Simple heuristic based on image brightness and noise
        mean_brightness = np.mean(image)
        noise_estimate = np.std(image - cv2.GaussianBlur(image, (5, 5), 2))
        
        # Rough estimation
        if noise_estimate < 5:
            return 100
        elif noise_estimate < 15:
            return 400
        elif noise_estimate < 30:
            return 1600
        else:
            return 3200
    
    def estimate_exposure_time(self, image):
        """Estimate exposure time from image statistics"""
        # Simple heuristic based on brightness
        mean_brightness = np.mean(image)
        
        if mean_brightness > 200:
            return 1/1000  # Fast shutter
        elif mean_brightness > 128:
            return 1/60    # Normal
        else:
            return 1/30    # Slower shutter


class CameraPipelineProcessor:
    """Processes images considering camera pipeline effects"""
    
    def __init__(self):
        self.pipeline_stages = [
            'dark_current_subtraction',
            'fixed_pattern_correction',
            'demosaicing_aware_denoising',
            'color_space_processing',
            'gamma_correction_compensation'
        ]
    
    def pipeline_aware_denoising(self, image, noise_components, pipeline_info=None):
        """Process considering camera pipeline effects"""
        
        processed_image = image.astype(np.float64)
        
        # Stage 1: Dark current and fixed pattern correction
        processed_image = self.correct_systematic_noise(
            processed_image, noise_components
        )
        
        # Stage 2: Demosaicing-aware denoising
        processed_image = self.demosaicing_aware_denoising(
            processed_image, noise_components
        )
        
        # Stage 3: Color space aware processing
        processed_image = self.color_space_aware_processing(
            processed_image, noise_components
        )
        
        # Stage 4: Gamma correction consideration
        processed_image = self.gamma_aware_processing(
            processed_image, pipeline_info
        )
        
        return np.clip(processed_image, 0, 255).astype(np.uint8)
    
    def correct_systematic_noise(self, image, noise_components):
        """Correct dark current and fixed pattern noise"""
        
        corrected = image.copy()
        
        # Fixed pattern noise correction
        if 'fixed_pattern' in noise_components:
            fpn_map = noise_components['fixed_pattern']
            if len(image.shape) == 3:
                # Apply to each channel
                for c in range(3):
                    corrected[:, :, c] = corrected[:, :, c] - fpn_map
            else:
                corrected = corrected - fpn_map
        
        # Dark current subtraction (apply as constant offset)
        if 'dark_current' in noise_components:
            dark_current = noise_components['dark_current']
            corrected = corrected - dark_current
        
        return corrected
    
    def demosaicing_aware_denoising(self, image, noise_components):
        """Denoising that considers demosaicing artifacts"""
        
        # Separate luminance and chrominance
        if len(image.shape) == 3:
            # Convert to YUV for separate processing
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            yuv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2YUV)
            y, u, v = yuv[:,:,0], yuv[:,:,1], yuv[:,:,2]
            
            # Process luminance with edge-preserving denoising
            y_denoised = self.edge_preserving_denoising(
                y.astype(np.float64), 
                np.mean(noise_components.get('shot_noise', 10))
            )
            
            # Process chrominance with stronger smoothing
            u_denoised = cv2.bilateralFilter(u, 9, 75, 75)
            v_denoised = cv2.bilateralFilter(v, 9, 75, 75)
            
            # Reconstruct
            yuv_denoised = np.stack([y_denoised, u_denoised, v_denoised], axis=2)
            processed = cv2.cvtColor(yuv_denoised.astype(np.uint8), cv2.COLOR_YUV2BGR)
        else:
            # Grayscale processing
            processed = self.edge_preserving_denoising(
                image, 
                np.mean(noise_components.get('shot_noise', 10))
            )
        
        return processed.astype(np.float64)
    
    def edge_preserving_denoising(self, image, noise_variance):
        """Edge-preserving denoising using anisotropic diffusion"""
        
        def g_function(grad_mag, k):
            """Diffusion function"""
            return np.exp(-(grad_mag / k)**2)
        
        result = image.copy().astype(np.float64)
        dt = 0.25  # Time step
        iterations = 15
        k = max(np.sqrt(2 * noise_variance), 5.0)  # Noise-dependent threshold
        
        for _ in range(iterations):
            # Compute gradients with padding
            grad_n = np.diff(result, axis=0, prepend=result[:1])
            grad_s = np.diff(result, axis=0, append=result[-1:])
            grad_w = np.diff(result, axis=1, prepend=result[:, :1])
            grad_e = np.diff(result, axis=1, append=result[:, -1:])
            
            # Compute diffusion coefficients
            c_n = g_function(np.abs(grad_n), k)
            c_s = g_function(np.abs(grad_s), k)
            c_w = g_function(np.abs(grad_w), k)
            c_e = g_function(np.abs(grad_e), k)
            
            # Update equation
            result = result + dt * (
                c_n * grad_n + c_s * grad_s + c_w * grad_w + c_e * grad_e
            )
        
        return result
    
    def color_space_aware_processing(self, image, noise_components):
        """Process in appropriate color space"""
        if len(image.shape) == 3:
            # Convert to LAB for perceptually uniform processing
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB)
            
            # Process L channel with detail preservation
            lab[:,:,0] = cv2.bilateralFilter(lab[:,:,0], 9, 80, 80)
            
            # Process A and B channels with smoothing
            lab[:,:,1] = cv2.bilateralFilter(lab[:,:,1], 9, 100, 100)
            lab[:,:,2] = cv2.bilateralFilter(lab[:,:,2], 9, 100, 100)
            
            # Convert back
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return processed.astype(np.float64)
        
        return image
    
    def gamma_aware_processing(self, image, pipeline_info):
        """Process considering gamma correction"""
        # Apply mild gamma correction awareness
        if pipeline_info and 'gamma' in pipeline_info:
            gamma = pipeline_info['gamma']
            # Reverse gamma for processing, then reapply
            normalized = image / 255.0
            linearized = np.power(normalized, gamma)
            # Process in linear space would go here
            gamma_corrected = np.power(linearized, 1/gamma)
            return gamma_corrected * 255.0
        
        return image


class CameraSpecificOptimizer:
    """Applies camera and lens specific optimizations"""
    
    def __init__(self):
        self.camera_profiles = self.load_camera_profiles()
        self.lens_profiles = self.load_lens_profiles()
    
    def load_camera_profiles(self):
        """Load camera-specific profiles"""
        return {
            'Canon': {
                'sensor': {'type': 'CMOS', 'size': 'full_frame'},
                'color_matrix': np.array([[1.2, -0.1, 0], [-0.05, 1.15, -0.05], [0, -0.1, 1.1]])
            },
            'Nikon': {
                'sensor': {'type': 'CMOS', 'size': 'full_frame'},
                'color_matrix': np.array([[1.15, -0.08, 0], [-0.03, 1.12, -0.02], [0, -0.08, 1.08]])
            },
            'Sony': {
                'sensor': {'type': 'CMOS', 'size': 'aps_c'},
                'color_matrix': np.array([[1.1, -0.05, 0], [-0.02, 1.1, -0.01], [0, -0.05, 1.05]])
            }
        }
    
    def load_lens_profiles(self):
        """Load lens-specific profiles"""
        return {
            'standard': {'vignetting_correction': True, 'distortion_correction': False},
            'wide_angle': {'vignetting_correction': True, 'distortion_correction': True},
            'telephoto': {'vignetting_correction': False, 'distortion_correction': False}
        }
    
    def manufacturer_specific_optimization(self, image, camera_model, lens_model=None):
        """Apply manufacturer and model specific optimizations"""
        
        processed = image.copy()
        
        # Find matching camera profile
        profile = None
        for manufacturer, prof in self.camera_profiles.items():
            if manufacturer.lower() in camera_model.lower():
                profile = prof
                break
        
        if profile:
            # Apply sensor-specific noise model
            processed = self.apply_sensor_specific_processing(processed, profile['sensor'])
            
            # Apply color matrix correction
            if 'color_matrix' in profile:
                processed = self.apply_color_matrix(processed, profile['color_matrix'])
        
        # Apply lens-specific corrections
        if lens_model and lens_model in self.lens_profiles:
            lens_profile = self.lens_profiles[lens_model]
            processed = self.apply_lens_corrections(processed, lens_profile)
        
        return processed
    
    def apply_sensor_specific_processing(self, image, sensor_profile):
        """Apply sensor-specific noise reduction"""
        
        processed = image.copy()
        
        if sensor_profile['type'] == 'CMOS':
            # CMOS sensors have row-wise readout noise patterns
            processed = self.correct_row_noise(processed)
        elif sensor_profile['type'] == 'CCD':
            # CCD sensors have different noise characteristics
            processed = self.correct_ccd_noise(processed)
        
        # Apply sensor size specific processing
        if sensor_profile['size'] == 'full_frame':
            noise_reduction_factor = 0.8
        elif sensor_profile['size'] == 'aps_c':
            noise_reduction_factor = 1.0
        elif sensor_profile['size'] == 'micro_43':
            noise_reduction_factor = 1.2
        else:
            noise_reduction_factor = 1.0
        
        # Apply adaptive noise reduction
        processed = self.adaptive_noise_reduction(processed, noise_reduction_factor)
        
        return processed
    
    def correct_row_noise(self, image):
        """Correct CMOS row-wise noise patterns"""
        corrected = image.copy().astype(np.float64)
        
        if len(image.shape) == 3:
            # Process each channel
            for c in range(3):
                for row in range(image.shape[0]):
                    row_data = corrected[row, :, c]
                    
                    # Estimate row bias using robust statistics
                    row_median = np.median(row_data)
                    row_mad = np.median(np.abs(row_data - row_median))
                    
                    # Detect outliers
                    outlier_mask = np.abs(row_data - row_median) > 3 * row_mad
                    
                    if np.sum(outlier_mask) > 0:
                        valid_indices = np.where(~outlier_mask)[0]
                        if len(valid_indices) > 2:
                            try:
                                interp_func = interp1d(valid_indices, row_data[valid_indices],
                                                     kind='linear', fill_value='extrapolate')
                                corrected[row, outlier_mask, c] = interp_func(np.where(outlier_mask)[0])
                            except:
                                # Fallback to median filtering
                                corrected[row, outlier_mask, c] = row_median
        else:
            # Grayscale processing
            for row in range(image.shape[0]):
                row_data = corrected[row, :]
                row_median = np.median(row_data)
                row_mad = np.median(np.abs(row_data - row_median))
                outlier_mask = np.abs(row_data - row_median) > 3 * row_mad
                
                if np.sum(outlier_mask) > 0:
                    valid_indices = np.where(~outlier_mask)[0]
                    if len(valid_indices) > 2:
                        try:
                            interp_func = interp1d(valid_indices, row_data[valid_indices],
                                                 kind='linear', fill_value='extrapolate')
                            corrected[row, outlier_mask] = interp_func(np.where(outlier_mask)[0])
                        except:
                            corrected[row, outlier_mask] = row_median
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def correct_ccd_noise(self, image):
        """Correct CCD-specific noise patterns"""
        # CCD sensors typically have more uniform noise
        # Apply gentle smoothing
        return cv2.bilateralFilter(image, 5, 50, 50)
    
    def adaptive_noise_reduction(self, image, factor):
        """Apply adaptive noise reduction based on sensor characteristics"""
        # Use non-local means with factor-adjusted parameters
        if len(image.shape) == 3:
            h = 10 * factor
            result = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        else:
            h = 10 * factor
            result = cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        
        return result
    
    def apply_color_matrix(self, image, color_matrix):
        """Apply color matrix correction"""
        if len(image.shape) == 3:
            # Reshape image for matrix multiplication
            h, w, c = image.shape
            image_reshaped = image.reshape(-1, c).astype(np.float64)
            
            # Apply color matrix
            corrected = np.dot(image_reshaped, color_matrix.T)
            
            # Reshape back and clip
            corrected = corrected.reshape(h, w, c)
            return np.clip(corrected, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_lens_corrections(self, image, lens_profile):
        """Apply lens-specific corrections"""
        processed = image.copy()
        
        # Vignetting correction
        if lens_profile.get('vignetting_correction', False):
            processed = self.correct_vignetting(processed)
        
        # Distortion correction would go here
        if lens_profile.get('distortion_correction', False):
            # Placeholder for distortion correction
            pass
        
        return processed
    
    def correct_vignetting(self, image):
        """Simple vignetting correction"""
        h, w = image.shape[:2]
        
        # Create radial distance map
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize distance
        normalized_distance = distance / max_distance
        
        # Create vignetting correction map
        correction = 1 + 0.3 * normalized_distance**2
        
        # Apply correction
        if len(image.shape) == 3:
            correction = correction[:, :, np.newaxis]
        
        corrected = image.astype(np.float64) * correction
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def shooting_condition_adaptation(self, image, shooting_conditions):
        """Adapt processing based on shooting conditions"""
        
        processed = image.copy()
        
        # Low light adaptation
        if shooting_conditions.get('light_level') == 'low':
            processed = self.low_light_processing(processed)
        
        # High ISO adaptation
        if shooting_conditions.get('iso', 100) > 1600:
            processed = self.high_iso_processing(processed)
        
        # Scene-specific adaptation
        scene_type = shooting_conditions.get('scene_type')
        if scene_type == 'portrait':
            processed = self.portrait_optimized_processing(processed)
        elif scene_type == 'landscape':
            processed = self.landscape_optimized_processing(processed)
        elif scene_type == 'night':
            processed = self.night_photography_processing(processed)
        
        return processed
    
    def low_light_processing(self, image):
        """Specialized processing for low light images"""
        
        if len(image.shape) == 3:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
            
            # Luminance noise reduction with detail preservation
            l_denoised = self.detail_preserving_luminance_denoising(l)
            
            # Aggressive chrominance denoising
            a_denoised = cv2.bilateralFilter(a, 15, 80, 80)
            b_denoised = cv2.bilateralFilter(b, 15, 80, 80)
            
            # Reconstruct
            lab_denoised = np.stack([l_denoised, a_denoised, b_denoised], axis=2)
            result = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
            
            return result
        else:
            return self.detail_preserving_luminance_denoising(image)
    
    def detail_preserving_luminance_denoising(self, luminance):
        """Detail-preserving denoising for luminance channel"""
        return cv2.bilateralFilter(luminance, 9, 75, 75)
    
    def high_iso_processing(self, image):
        """Processing optimized for high ISO images"""
        
        if len(image.shape) == 3:
            # Convert to YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Luma channel - preserve details
            yuv[:,:,0] = self.edge_aware_denoising(yuv[:,:,0])
            
            # Chroma channels - aggressive denoising
            yuv[:,:,1] = self.strong_smoothing_filter(yuv[:,:,1])
            yuv[:,:,2] = self.strong_smoothing_filter(yuv[:,:,2])
            
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return result
        else:
            return self.edge_aware_denoising(image)
    
    def edge_aware_denoising(self, channel):
        """Edge-aware denoising"""
        return cv2.bilateralFilter(channel, 9, 80, 80)
    
    def strong_smoothing_filter(self, channel):
        """Strong smoothing for color channels"""
        return cv2.bilateralFilter(channel, 15, 100, 100)
    
    def portrait_optimized_processing(self, image):
        """Portrait-optimized processing"""
        # Gentle noise reduction to preserve skin texture
        return cv2.bilateralFilter(image, 9, 60, 60)
    
    def landscape_optimized_processing(self, image):
        """Landscape-optimized processing"""
        # Detail preservation for textures
        return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    
    def night_photography_processing(self, image):
        """Night photography processing"""
        # Strong noise reduction for very low light
        return cv2.fastNlMeansDenoisingColored(image, None, 12, 12, 7, 21)


class ComprehensiveCameraDenoiser:
    """Main class that orchestrates the entire denoising pipeline"""
    
    def __init__(self):
        self.noise_modeler = CameraNoiseModeler()
        self.pipeline_processor = CameraPipelineProcessor()
        self.camera_optimizer = CameraSpecificOptimizer()
    
    def process_image(self, image, metadata=None, camera_model=None, 
                     lens_model=None, shooting_conditions=None):
        """
        Main processing function
        
        Args:
            image: RGB image as numpy array
            metadata: Dictionary with camera metadata (iso, exposure_time, temperature)
            camera_model: Camera model string
            lens_model: Lens model string
            shooting_conditions: Dictionary with shooting conditions
        
        Returns:
            Processed RGB image
        """
        
        # Step 1: Analyze noise components
        noise_components = self.noise_modeler.comprehensive_noise_analysis(
            image, metadata
        )
        
        # Step 2: Apply pipeline-aware processing
        processed_image = self.pipeline_processor.pipeline_aware_denoising(
            image, noise_components, metadata
        )
        
        # Step 3: Apply camera-specific optimizations
        if camera_model:
            processed_image = self.camera_optimizer.manufacturer_specific_optimization(
                processed_image, camera_model, lens_model
            )
        
        # Step 4: Apply shooting condition adaptations
        if shooting_conditions:
            processed_image = self.camera_optimizer.shooting_condition_adaptation(
                processed_image, shooting_conditions
            )
        
        return processed_image


# # Example usage
# if __name__ == "__main__":
#     # Create denoiser instance
#     denoiser = ComprehensiveCameraDenoiser()
    
#     # Example metadata
#     metadata = {
#         'iso': 1600,
#         'exposure_time': 1/60,
#         'temperature': 25
#     }
    
#     # Example shooting conditions
#     shooting_conditions = {
#         'light_level': 'low',
#         'iso': 1600,
#         'scene_type': 'portrait'
#     }
    
#     # Load and process an image (placeholder)
#     # image = cv2.imread('noisy_image.jpg')
#     # processed = denoiser.process_image(
#     #     image, 
#     #     metadata=metadata,
#     #     camera_model='Canon EOS R5',
#     #     lens_model='standard',
#     #     shooting_conditions=shooting_conditions
#     # )
#     # cv2.imwrite('denoised_image.jpg', processed)
    
    