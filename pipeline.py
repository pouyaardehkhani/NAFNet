"""
Denoising Pipeline Implementation
This module creates a complete denoising pipeline that:
1. Detects noise types using noise_identifier
2. Creates a processing plan
3. Executes the plan using appropriate modules in the correct order
"""

import cv2
import numpy as np
from noise_finder import NoiseDetector
from modules.module1 import PatchBasedDenoiser
from modules.module2 import StatisticalJumpInpaintingSystem
from modules.module3 import NonLinearFrequencyProcessor
from modules.module4 import VariationalPatternDecomposer
from modules.module5 import HybridMotionStaticDenoiser
from modules.module6 import ComprehensiveCameraDenoiser

class DenoisingPipeline:
    """Main pipeline class that orchestrates the entire denoising process"""
    
    def __init__(self, detector_type='ultra_fast', model_size='huge'):
        """
        Initialize the pipeline with a noise detector
        
        Args:
            detector_type: Type of detector to use
            model_size: Size of the detection model
        """
        self.noise_detector = NoiseDetector(detector_type, model_size)
        self.setup_denoisers()
        
    def setup_denoisers(self):
        """Initialize all denoising modules with default parameters"""
        # Module 1: Mathematical/Additive/Speckle
        self.math_denoiser = PatchBasedDenoiser(patch_sizes=[8], n_clusters=16)
        
        # Module 2: Impulse
        self.impulse_denoiser = StatisticalJumpInpaintingSystem(sensitivity=2.0, multi_scale=True)
        
        # Module 3: Frequency
        self.freq_denoiser = NonLinearFrequencyProcessor()
        
        # Module 4: Structured/Spatial
        self.spatial_denoiser = VariationalPatternDecomposer(method='multi_component', n_components=3)
        
        # Module 5: Motion
        self.motion_denoiser = HybridMotionStaticDenoiser()
        
        # Module 6: Camera
        self.camera_denoiser = ComprehensiveCameraDenoiser()
    
    def analyze_image(self, image_path):
        """
        Analyze the image and create a processing plan
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (results, plan)
        """
        # Detect noise types
        results = self.noise_detector.detect_noise_type_fast(image_path)
        
        # Generate processing plan based on detections
        plan = self.noise_detector.create_processing_order(results, aggressive_processing=True)
        
        return results, plan
    
    def adjust_denoiser_parameters(self, denoiser_module, confidence):
        """
        Adjust denoiser parameters based on confidence level
        
        Args:
            denoiser_module: The denoiser module to adjust
            confidence: Confidence score from the plan
            
        Returns:
            Dict of adjusted parameters
        """
        params = {}
        
        # Scale parameters based on confidence
        if confidence > 0.8:
            strength = 'strong'
        elif confidence > 0.4:
            strength = 'moderate'
        elif confidence > 0.2:
            strength = 'gentle'
        else:
            strength = 'minimal'
            
        # Set module-specific parameters
        if isinstance(denoiser_module, PatchBasedDenoiser):
            params['iterations'] = max(1, int(confidence * 3))
            params['n_clusters'] = max(16, int(confidence * 64))
            
        elif isinstance(denoiser_module, NonLinearFrequencyProcessor):
            params['cutoff_frequency'] = min(0.1 + (confidence * 0.2), 0.5)
            params['gamma_low'] = max(0.2, confidence * 0.5)
            params['gamma_high'] = min(2.0, 1.5 + confidence)
            
        elif isinstance(denoiser_module, VariationalPatternDecomposer):
            params['lambda_tv'] = 0.1 * confidence
            params['lambda_structure'] = 0.05 * confidence
            
        elif isinstance(denoiser_module, HybridMotionStaticDenoiser):
            params['temporal_window'] = max(3, int(5 * confidence))
            
        elif isinstance(denoiser_module, ComprehensiveCameraDenoiser):
            params['strength'] = confidence
            
        return params
    
    def execute_plan(self, image_path, plan, save_intermediate=False):
        """
        Execute the denoising plan
        
        Args:
            image_path: Path to input image
            plan: Processing plan from analyze_image
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Denoised image
        """
        # Read input image
        current_image = cv2.imread(image_path)
        
        # Process through each module in plan order
        for step in plan['steps']:
            module_type = step['module']
            confidence = step['confidence']
            
            print(f"Applying {module_type} denoiser with confidence {confidence:.3f}")
            
            # Select appropriate module and parameters
            if module_type == 'structured/spatial':
                denoiser = self.spatial_denoiser
                params = self.adjust_denoiser_parameters(denoiser, confidence)
                current_image = denoiser.decompose(current_image)['denoised']
                
            elif module_type == 'camera':
                denoiser = self.camera_denoiser
                params = self.adjust_denoiser_parameters(denoiser, confidence)
                current_image = denoiser.apply_camera_denoising(current_image, params['strength'])
                
            elif module_type == 'frequency':
                denoiser = self.freq_denoiser
                params = self.adjust_denoiser_parameters(denoiser, confidence)
                current_image = denoiser.process_rgb_image(current_image, 'homomorphic', **params)
                
            elif module_type == 'mathematical/additive/speckle':
                denoiser = self.math_denoiser
                params = self.adjust_denoiser_parameters(denoiser, confidence)
                current_image = denoiser.process(current_image, **params)
                
            elif module_type == 'motion':
                denoiser = self.motion_denoiser
                params = self.adjust_denoiser_parameters(denoiser, confidence)
                current_image = denoiser.process_single_image(current_image)
            
            # Save intermediate result if requested
            if save_intermediate:
                intermediate_path = f"intermediate_{module_type}.jpg"
                cv2.imwrite(intermediate_path, current_image)
        
        return current_image

    def process_image(self, image_path, output_path=None, save_intermediate=False):
        """
        Complete pipeline processing of an image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Denoised image
        """
        # Analyze and create plan
        results, plan = self.analyze_image(image_path)
        
        # Print plan for reference
        self.noise_detector.print_processing_plan(plan)
        
        # Execute the plan
        denoised_image = self.execute_plan(image_path, plan, save_intermediate)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, denoised_image)
        
        return denoised_image

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DenoisingPipeline(detector_type='ultra_fast', model_size='huge')
    
    # Process an image
    input_path = "noisy_image.jpg"
    output_path = "denoised_result.jpg"
    
    denoised = pipeline.process_image(input_path, output_path, save_intermediate=True)
    print("Denoising complete!")
