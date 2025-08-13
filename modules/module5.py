"""
Module 5: motion denoiser
"""

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedMotionAnalyzer:
    """Advanced motion analysis for video denoising"""
    
    def __init__(self, block_size=16, search_range=32):
        self.block_size = block_size
        self.search_range = search_range
    
    def optical_flow_analysis(self, frame1, frame2, method='lucas_kanade'):
        """Comprehensive optical flow analysis"""
        
        if method == 'lucas_kanade':
            flow = self.lucas_kanade_optical_flow(frame1, frame2)
        elif method == 'horn_schunck':
            flow = self.horn_schunck_optical_flow(frame1, frame2)
        elif method == 'farneback':
            flow = self.farneback_optical_flow(frame1, frame2)
        else:
            flow = self.lucas_kanade_optical_flow(frame1, frame2)
        
        # Motion vector reliability estimation
        motion_confidence = self.estimate_motion_confidence(frame1, frame2, flow)
        
        # Motion vector smoothing
        smoothed_flow = self.smooth_motion_vectors(flow, motion_confidence)
        
        return smoothed_flow, motion_confidence
    
    def lucas_kanade_optical_flow(self, frame1, frame2, window_size=15):
        """Lucas-Kanade optical flow with pyramid"""
        
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        # Detect features to track
        corners = cv2.goodFeaturesToTrack(gray1, maxCorners=1000, 
                                        qualityLevel=0.01, minDistance=10)
        
        # Lucas-Kanade optical flow
        if corners is not None and len(corners) > 0:
            next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, corners, None,
                winSize=(window_size, window_size),
                maxLevel=3
            )
            
            # Create dense flow field
            flow = self.interpolate_dense_flow(corners, next_corners, 
                                             status, frame1.shape)
        else:
            flow = np.zeros((frame1.shape[0], frame1.shape[1], 2))
        
        return flow
    
    def farneback_optical_flow(self, frame1, frame2):
        """Farneback optical flow method"""
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
            
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow
    
    def horn_schunck_optical_flow(self, frame1, frame2, alpha=0.001, iterations=100):
        """Horn-Schunck optical flow implementation"""
        if len(frame1.shape) == 3:
            I1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float64)
            I2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            I1, I2 = frame1.astype(np.float64), frame2.astype(np.float64)
        
        # Compute derivatives
        Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=3)
        It = I2 - I1
        
        # Initialize flow
        u = np.zeros_like(I1)
        v = np.zeros_like(I1)
        
        # Iterative solution
        for _ in range(iterations):
            u_avg = cv2.blur(u, (3, 3))
            v_avg = cv2.blur(v, (3, 3))
            
            denominator = alpha**2 + Ix**2 + Iy**2
            u = u_avg - Ix * (Ix * u_avg + Iy * v_avg + It) / denominator
            v = v_avg - Iy * (Ix * u_avg + Iy * v_avg + It) / denominator
        
        flow = np.dstack([u, v])
        return flow
    
    def interpolate_dense_flow(self, corners, next_corners, status, frame_shape):
        """Interpolate sparse flow to dense flow field"""
        h, w = frame_shape[:2]
        flow = np.zeros((h, w, 2))
        
        if corners is None or next_corners is None:
            return flow
        
        # Filter valid correspondences
        valid_idx = status.flatten() == 1
        if not np.any(valid_idx):
            return flow
        
        valid_corners = corners[valid_idx].reshape(-1, 2)
        valid_next = next_corners[valid_idx].reshape(-1, 2)
        
        # Compute flow vectors
        flow_vectors = valid_next - valid_corners
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        
        # Inverse distance weighting interpolation
        if len(valid_corners) > 0:
            distances = cdist(grid_points, valid_corners)
            weights = 1.0 / (distances + 1e-6)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
            interpolated_flow = np.dot(weights, flow_vectors)
            flow = interpolated_flow.reshape(h, w, 2)
        
        return flow
    
    def estimate_motion_confidence(self, frame1, frame2, flow):
        """Estimate motion vector reliability"""
        h, w = frame1.shape[:2]
        confidence = np.ones((h, w))
        
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        # Motion compensation error
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        new_x = np.clip(x_coords + flow[:,:,0], 0, w-1)
        new_y = np.clip(y_coords + flow[:,:,1], 0, h-1)
        
        compensated = map_coordinates(gray2, [new_y, new_x], order=1, mode='nearest')
        error = np.abs(gray1 - compensated)
        
        # Convert error to confidence
        confidence = np.exp(-error / 20.0)
        
        return confidence
    
    def smooth_motion_vectors(self, flow, confidence, kernel_size=5):
        """Smooth motion vectors based on confidence"""
        smoothed_flow = flow.copy()
        
        # Gaussian smoothing weighted by confidence
        for i in range(2):
            weighted_flow = flow[:,:,i] * confidence
            weight_sum = confidence + 1e-6
            
            smoothed_weighted = gaussian_filter(weighted_flow, sigma=kernel_size/3.0)
            smoothed_weights = gaussian_filter(weight_sum, sigma=kernel_size/3.0)
            
            smoothed_flow[:,:,i] = smoothed_weighted / smoothed_weights
        
        return smoothed_flow
    
    def motion_segmentation(self, flow_field, threshold=2.0):
        """Segment image into motion and static regions"""
        
        # Compute motion magnitude
        motion_magnitude = np.sqrt(flow_field[:,:,0]**2 + flow_field[:,:,1]**2)
        
        # Adaptive thresholding based on distribution
        motion_threshold = np.percentile(motion_magnitude, 75) * threshold
        
        # Create motion mask
        motion_mask = motion_magnitude > motion_threshold
        
        # Morphological operations for clean segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), 
                                     cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to ensure coverage
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        static_mask = ~motion_mask.astype(bool)
        
        return motion_mask.astype(bool), static_mask
    
    def temporal_gradient_analysis(self, frames):
        """Analyze temporal gradients for motion detection"""
        
        if len(frames) < 3:
            return None
        
        temporal_gradients = []
        
        for i in range(1, len(frames)-1):
            # Temporal derivative
            temp_grad = (frames[i+1].astype(np.float64) - 
                        frames[i-1].astype(np.float64)) / 2.0
            temporal_gradients.append(temp_grad)
        
        # Average temporal gradient magnitude
        avg_temp_grad = np.mean([np.abs(grad) for grad in temporal_gradients], axis=0)
        
        return avg_temp_grad

class NoiseCharacteristics:
    """Noise characteristics analysis"""
    
    def __init__(self):
        self.noise_variance = 0
        self.noise_type = 'gaussian'
        self.spatial_correlation = 0
    
    def estimate_noise(self, image):
        """Estimate noise characteristics from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Estimate noise variance using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        self.noise_variance = np.var(laplacian) / 6.0
        
        # Simple noise type classification (could be more sophisticated)
        if self.noise_variance < 10:
            self.noise_type = 'low'
        elif self.noise_variance < 50:
            self.noise_type = 'medium'
        else:
            self.noise_type = 'high'
        
        return self

class StaticRegionDenoiser:
    """Denoiser specialized for static regions"""
    
    def __init__(self):
        pass
    
    def process(self, image, static_mask, noise_characteristics):
        """Process static regions with spatial denoising"""
        processed = image.copy().astype(np.float64)
        
        # Apply Non-local Means denoising to static regions
        if len(image.shape) == 3:
            for c in range(3):
                channel = image[:,:,c].astype(np.uint8)
                if noise_characteristics.noise_type == 'low':
                    denoised = cv2.fastNlMeansDenoising(channel, None, 3, 7, 21)
                elif noise_characteristics.noise_type == 'medium':
                    denoised = cv2.fastNlMeansDenoising(channel, None, 10, 7, 21)
                else:
                    denoised = cv2.fastNlMeansDenoising(channel, None, 20, 7, 21)
                processed[:,:,c] = denoised
        else:
            if noise_characteristics.noise_type == 'low':
                processed = cv2.fastNlMeansDenoising(image.astype(np.uint8), None, 3, 7, 21)
            elif noise_characteristics.noise_type == 'medium':
                processed = cv2.fastNlMeansDenoising(image.astype(np.uint8), None, 10, 7, 21)
            else:
                processed = cv2.fastNlMeansDenoising(image.astype(np.uint8), None, 20, 7, 21)
        
        return processed

class MotionRegionDenoiser:
    """Denoiser specialized for motion regions"""
    
    def __init__(self):
        pass
    
    def process(self, image, motion_mask, noise_characteristics):
        """Process motion regions with temporal denoising"""
        processed = image.copy().astype(np.float64)
        
        # Apply gentler spatial filtering for motion regions
        if noise_characteristics.noise_type == 'low':
            sigma = 0.5
        elif noise_characteristics.noise_type == 'medium':
            sigma = 1.0
        else:
            sigma = 1.5
        
        # Gaussian filtering
        if len(image.shape) == 3:
            for c in range(3):
                processed[:,:,c] = gaussian_filter(processed[:,:,c], sigma=sigma)
        else:
            processed = gaussian_filter(processed, sigma=sigma)
        
        return processed

class BoundaryProcessor:
    """Processor for boundary regions between motion and static areas"""
    
    def __init__(self):
        pass
    
    def process(self, image, boundary_mask, motion_mask, static_mask, noise_characteristics):
        """Process boundary regions with hybrid approach"""
        processed = image.copy().astype(np.float64)
        
        # Apply moderate denoising for boundary regions
        if noise_characteristics.noise_type == 'low':
            sigma = 0.8
        elif noise_characteristics.noise_type == 'medium':
            sigma = 1.2
        else:
            sigma = 1.8
        
        # Bilateral filtering for edge-preserving denoising
        if len(image.shape) == 3:
            processed = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75).astype(np.float64)
        else:
            processed = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75).astype(np.float64)
        
        return processed

class AdaptiveProcessor:
    """Adaptive processor that selects denoising method based on region type"""
    
    def __init__(self):
        self.static_denoiser = StaticRegionDenoiser()
        self.motion_denoiser = MotionRegionDenoiser()
        self.boundary_processor = BoundaryProcessor()
    
    def adaptive_processing_selection(self, image, motion_mask, static_mask, 
                                   noise_characteristics):
        """Select processing method based on region type and noise"""
        
        processing_map = np.zeros_like(motion_mask, dtype=int)
        
        # Classify regions
        # 0: Static region
        # 1: Motion region  
        # 2: Boundary region
        
        processing_map[static_mask] = 0
        processing_map[motion_mask] = 1
        
        # Detect boundary regions
        boundary_mask = self.detect_boundary_regions(motion_mask, static_mask)
        processing_map[boundary_mask] = 2
        
        # Process each region type
        processed_image = image.copy().astype(np.float64)
        
        # Static regions - use spatial methods
        if np.any(static_mask):
            static_result = self.static_denoiser.process(
                image, static_mask, noise_characteristics
            )
            processed_image[static_mask] = static_result[static_mask]
        
        # Motion regions - use temporal methods
        if np.any(motion_mask):
            motion_result = self.motion_denoiser.process(
                image, motion_mask, noise_characteristics
            )
            processed_image[motion_mask] = motion_result[motion_mask]
        
        # Boundary regions - use hybrid methods
        if np.any(boundary_mask):
            boundary_result = self.boundary_processor.process(
                image, boundary_mask, motion_mask, static_mask, 
                noise_characteristics
            )
            processed_image[boundary_mask] = boundary_result[boundary_mask]
        
        return processed_image
    
    def detect_boundary_regions(self, motion_mask, static_mask, 
                              boundary_width=5):
        """Detect boundary regions between motion and static areas"""
        
        # Create boundary mask
        motion_boundary = cv2.dilate(motion_mask.astype(np.uint8), 
                                   np.ones((boundary_width, boundary_width)), 
                                   iterations=1)
        static_boundary = cv2.dilate(static_mask.astype(np.uint8), 
                                   np.ones((boundary_width, boundary_width)), 
                                   iterations=1)
        
        # Boundary is intersection of dilated regions
        boundary_mask = (motion_boundary & static_boundary).astype(bool)
        
        return boundary_mask

class TemporalConsistencyEnforcer:
    """Enforce temporal consistency across frames"""
    
    def __init__(self, temporal_window=5):
        self.temporal_window = temporal_window
        self.frame_buffer = []
    
    def enforce_temporal_consistency(self, processed_frames, 
                                   motion_vectors, confidence_maps):
        """Enforce temporal consistency across frames"""
        
        consistent_frames = []
        
        for i, frame in enumerate(processed_frames):
            if i == 0:
                # First frame - no temporal constraint
                consistent_frames.append(frame)
                continue
            
            # Motion-compensated prediction
            predicted_frame = self.motion_compensated_prediction(
                consistent_frames[-1], motion_vectors[i-1]
            )
            
            # Temporal filtering
            temporally_filtered = self.temporal_filter(
                frame, predicted_frame, confidence_maps[i]
            )
            
            # Flickering suppression
            flicker_suppressed = self.suppress_flickering(
                temporally_filtered, consistent_frames[-1], 
                motion_vectors[i-1]
            )
            
            consistent_frames.append(flicker_suppressed)
        
        return consistent_frames
    
    def motion_compensated_prediction(self, reference_frame, motion_vector):
        """Create motion-compensated prediction"""
        
        h, w = reference_frame.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Apply motion vectors
        new_x = x_coords + motion_vector[:,:,0]
        new_y = y_coords + motion_vector[:,:,1]
        
        # Clip coordinates
        new_x = np.clip(new_x, 0, w-1)
        new_y = np.clip(new_y, 0, h-1)
        
        # Handle different image dimensions
        if len(reference_frame.shape) == 3:
            predicted = np.zeros_like(reference_frame)
            for c in range(reference_frame.shape[2]):
                channel = reference_frame[:,:,c]
                predicted_channel = map_coordinates(
                    channel, [new_y, new_x], order=1, mode='nearest'
                )
                predicted[:,:,c] = predicted_channel
        else:
            predicted = map_coordinates(
                reference_frame, [new_y, new_x], order=1, mode='nearest'
            )
        
        return predicted
    
    def temporal_filter(self, current_frame, predicted_frame, confidence_map):
        """Apply temporal filtering with confidence weighting"""
        
        # Ensure confidence_map has right dimensions
        if len(current_frame.shape) == 3 and len(confidence_map.shape) == 2:
            confidence_map = np.stack([confidence_map] * 3, axis=2)
        
        # Adaptive blending weight based on confidence
        alpha = 0.7 * confidence_map + 0.3 * (1 - confidence_map)
        
        # Temporal filtering
        filtered_frame = (alpha * current_frame + 
                         (1 - alpha) * predicted_frame)
        
        return filtered_frame
    
    def suppress_flickering(self, current_frame, previous_frame, 
                          motion_vector, threshold=10):
        """Suppress temporal flickering artifacts"""
        
        # Motion-compensated difference
        mc_previous = self.motion_compensated_prediction(
            previous_frame, motion_vector
        )
        
        frame_diff = np.abs(current_frame - mc_previous)
        
        # Detect potential flickering
        flickering_mask = frame_diff < threshold
        
        # Apply temporal smoothing only to flickering regions
        smoothed_frame = current_frame.copy()
        smoothed_frame[flickering_mask] = (
            0.5 * current_frame[flickering_mask] + 
            0.5 * mc_previous[flickering_mask]
        )
        
        return smoothed_frame

class HybridMotionStaticDenoiser:
    """Main class for hybrid motion-static denoising"""
    
    def __init__(self, temporal_window=5):
        self.motion_analyzer = AdvancedMotionAnalyzer()
        self.adaptive_processor = AdaptiveProcessor()
        self.temporal_enforcer = TemporalConsistencyEnforcer(temporal_window)
        self.frame_buffer = []
        
    def process_single_image(self, image):
        """Process a single RGB image (spatial denoising only)"""
        # Estimate noise characteristics
        noise_chars = NoiseCharacteristics().estimate_noise(image)
        
        # For single image, treat as static region
        h, w = image.shape[:2]
        static_mask = np.ones((h, w), dtype=bool)
        motion_mask = np.zeros((h, w), dtype=bool)
        
        # Apply adaptive processing
        processed = self.adaptive_processor.adaptive_processing_selection(
            image, motion_mask, static_mask, noise_chars
        )
        
        return processed.astype(np.uint8)
    
    def process_frame_sequence(self, frames, optical_flow_method='lucas_kanade'):
        """Process a sequence of RGB frames with motion analysis"""
        if len(frames) < 2:
            return [self.process_single_image(frame) for frame in frames]
        
        processed_frames = []
        motion_vectors = []
        confidence_maps = []
        
        for i in range(len(frames)):
            current_frame = frames[i]
            
            # Estimate noise characteristics
            noise_chars = NoiseCharacteristics().estimate_noise(current_frame)
            
            if i == 0:
                # First frame - treat as static
                h, w = current_frame.shape[:2]
                motion_mask = np.zeros((h, w), dtype=bool)
                static_mask = np.ones((h, w), dtype=bool)
                motion_vector = np.zeros((h, w, 2))
                confidence_map = np.ones((h, w))
            else:
                # Analyze motion between current and previous frame
                previous_frame = frames[i-1]
                
                # Optical flow analysis
                motion_vector, confidence_map = self.motion_analyzer.optical_flow_analysis(
                    previous_frame, current_frame, method=optical_flow_method
                )
                
                # Motion segmentation
                motion_mask, static_mask = self.motion_analyzer.motion_segmentation(
                    motion_vector
                )
            
            # Adaptive processing
            processed_frame = self.adaptive_processor.adaptive_processing_selection(
                current_frame, motion_mask, static_mask, noise_chars
            )
            
            processed_frames.append(processed_frame)
            motion_vectors.append(motion_vector)
            confidence_maps.append(confidence_map)
        
        # Enforce temporal consistency
        consistent_frames = self.temporal_enforcer.enforce_temporal_consistency(
            processed_frames, motion_vectors, confidence_maps
        )
        
        # Convert to uint8
        return [frame.astype(np.uint8) for frame in consistent_frames]
    
    def process_video_file(self, video_path, output_path=None, 
                          optical_flow_method='lucas_kanade'):
        """Process video file with hybrid denoising"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Process frames
        processed_frames = self.process_frame_sequence(frames, optical_flow_method)
        
        # Write output video if path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in processed_frames:
                out.write(frame)
            
            out.release()
        
        return processed_frames

# # Example usage
# if __name__ == "__main__":
#     # Initialize the denoiser
#     denoiser = HybridMotionStaticDenoiser()
    
#     # Example 1: Process single image
#     # image = cv2.imread('noisy_image.jpg')
#     # denoised_image = denoiser.process_single_image(image)
#     # cv2.imwrite('denoised_image.jpg', denoised_image)
    
#     # Example 2: Process frame sequence
#     # frames = [cv2.imread(f'frame_{i:03d}.jpg') for i in range(10)]
#     # denoised_frames = denoiser.process_frame_sequence(frames, 'farneback')
    
#     # Example 3: Process video file
#     # denoised_frames = denoiser.process_video_file('input_video.mp4', 'output_video.mp4')
    
#     print("Hybrid Motion-Static Denoising System initialized successfully!")
#     print("Available methods:")
#     print("- process_single_image(image): For single RGB image denoising")
#     print("- process_frame_sequence(frames, method): For frame sequence processing")  
#     print("- process_video_file(input_path, output_path, method): For video file processing")
#     print("Optical flow methods: 'lucas_kanade', 'farneback', 'horn_schunck'")