"""
Module 1: mathematical + additive + speckle denoiser
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.mixture import GaussianMixture
from scipy.fft import dct, idct
import warnings
warnings.filterwarnings('ignore')

class PatchExtractor:
    """Extracts overlapping patches at multiple scales from images"""
    
    def __init__(self, patch_sizes=[8, 16], overlap_ratio=0.5):
        self.patch_sizes = patch_sizes
        self.overlap_ratio = overlap_ratio
    
    def extract_patches(self, image):
        """Extract overlapping patches at multiple scales
        
        Args:
            image: 2D numpy array (grayscale image)
            
        Returns:
            patches: Array of flattened patches
            positions: List of (i, j, patch_size) tuples
        """
        all_patches = []
        patch_positions = []
        
        for patch_size in self.patch_sizes:
            step = max(1, int(patch_size * (1 - self.overlap_ratio)))
            
            for i in range(0, image.shape[0] - patch_size + 1, step):
                for j in range(0, image.shape[1] - patch_size + 1, step):
                    patch = image[i:i+patch_size, j:j+patch_size]
                    all_patches.append(patch.flatten())
                    patch_positions.append((i, j, patch_size))
        
        return np.array(all_patches), patch_positions

class StatisticalPatchModeling:
    """Creates statistical models for patches using clustering and dimensionality reduction"""
    
    def __init__(self, n_clusters=64, n_components_pca=32):
        self.n_clusters = n_clusters
        self.n_components_pca = n_components_pca
        self.patch_models = {}
        self.pca = None
        self.kmeans = None
    
    def cluster_and_model_patches(self, patches):
        """Cluster patches and create statistical models
        
        Args:
            patches: Array of flattened patches
            
        Returns:
            cluster_labels: Array of cluster assignments
        """
        # Handle empty patches
        if len(patches) == 0:
            return np.array([])
        
        # 1. PCA for dimensionality reduction
        n_components = min(self.n_components_pca, patches.shape[1], patches.shape[0])
        self.pca = PCA(n_components=n_components)
        
        try:
            patches_pca = self.pca.fit_transform(patches)
        except ValueError:
            # Fallback if PCA fails
            patches_pca = patches
        
        # 2. K-means clustering
        n_clusters = min(self.n_clusters, len(patches))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(patches_pca)
        
        # 3. Create models for each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_patches = patches[cluster_mask]
            
            if len(cluster_patches) > 5:  # Minimum patches for reliable modeling
                try:
                    # Gaussian Mixture Model
                    n_gmm_components = min(3, len(cluster_patches)//3)
                    if n_gmm_components > 0:
                        gmm = GaussianMixture(n_components=n_gmm_components, 
                                            random_state=42)
                        gmm.fit(cluster_patches)
                    else:
                        gmm = None
                    
                    # Dictionary Learning
                    n_dict_components = min(32, len(cluster_patches)//2)
                    if n_dict_components > 0:
                        dict_learner = DictionaryLearning(
                            n_components=n_dict_components,
                            alpha=0.1, 
                            max_iter=50,
                            random_state=42
                        )
                        dict_learner.fit(cluster_patches)
                    else:
                        dict_learner = None
                    
                    self.patch_models[cluster_id] = {
                        'gmm': gmm,
                        'dictionary': dict_learner,
                        'centroid': np.mean(cluster_patches, axis=0),
                        'std': np.std(cluster_patches, axis=0)
                    }
                except:
                    # Fallback model
                    self.patch_models[cluster_id] = {
                        'gmm': None,
                        'dictionary': None,
                        'centroid': np.mean(cluster_patches, axis=0),
                        'std': np.std(cluster_patches, axis=0) + 1e-6
                    }
        
        return cluster_labels

class CollaborativeFiltering:
    """Implements 3D collaborative filtering for patch denoising"""
    
    def __init__(self, search_window=21, patch_size=8, max_similar_patches=32):
        self.search_window = search_window
        self.patch_size = patch_size
        self.max_similar_patches = max_similar_patches
    
    def block_matching(self, image, reference_patch, ref_pos):
        """Find similar patches using block matching
        
        Args:
            image: Input image
            reference_patch: Reference patch as 2D array
            ref_pos: (i, j) position of reference patch
            
        Returns:
            similar_positions: List of similar patch positions
            similarities: List of similarity scores
        """
        similarities = []
        positions = []
        
        ref_i, ref_j = ref_pos
        half_window = self.search_window // 2
        
        # Search in local window
        start_i = max(0, ref_i - half_window)
        end_i = min(image.shape[0] - self.patch_size + 1, ref_i + half_window)
        start_j = max(0, ref_j - half_window)
        end_j = min(image.shape[1] - self.patch_size + 1, ref_j + half_window)
        
        ref_flat = reference_patch.flatten()
        ref_mean = np.mean(ref_flat)
        ref_std = np.std(ref_flat)
        
        for i in range(start_i, end_i, 2):  # Skip every other pixel for speed
            for j in range(start_j, end_j, 2):
                candidate_patch = image[i:i+self.patch_size, j:j+self.patch_size]
                candidate_flat = candidate_patch.flatten()
                
                # Normalized cross-correlation
                candidate_mean = np.mean(candidate_flat)
                candidate_std = np.std(candidate_flat)
                
                if ref_std > 1e-6 and candidate_std > 1e-6:
                    correlation = np.mean((ref_flat - ref_mean) * (candidate_flat - candidate_mean))
                    correlation /= (ref_std * candidate_std)
                    
                    # Distance-based similarity
                    distance = np.linalg.norm(ref_flat - candidate_flat)
                    similarity = correlation * np.exp(-distance / (self.patch_size * 255))
                    
                    similarities.append(similarity)
                    positions.append((i, j))
        
        # Sort and select top matches
        if len(similarities) == 0:
            return [(ref_i, ref_j)], [1.0]
        
        sorted_indices = np.argsort(similarities)[::-1]
        n_select = min(self.max_similar_patches, len(sorted_indices))
        
        selected_positions = [positions[i] for i in sorted_indices[:n_select]]
        selected_similarities = [similarities[i] for i in sorted_indices[:n_select]]
        
        return selected_positions, selected_similarities
    
    def apply_3d_dct(self, patches_3d):
        """Apply 3D DCT transform"""
        return dct(dct(dct(patches_3d, axis=0), axis=1), axis=2)
    
    def inverse_3d_dct(self, dct_3d):
        """Apply inverse 3D DCT transform"""
        return idct(idct(idct(dct_3d, axis=0), axis=1), axis=2)
    
    def wiener_filter_3d(self, dct_coeffs, noise_variance=0.01):
        """Apply Wiener filtering in 3D DCT domain"""
        signal_power = np.abs(dct_coeffs) ** 2
        wiener_factor = signal_power / (signal_power + noise_variance)
        return dct_coeffs * wiener_factor
    
    def collaborative_filter(self, image, patches, positions):
        """Apply 3D transform domain collaborative filtering
        
        Args:
            image: Input image
            patches: Array of patches
            positions: List of patch positions
            
        Returns:
            filtered_patches: Array of filtered patches
        """
        filtered_patches = []
        
        for idx, (patch, pos) in enumerate(zip(patches, positions)):
            patch_2d = patch.reshape(self.patch_size, self.patch_size)
            
            # Find similar patches
            similar_positions, similarities = self.block_matching(
                image, patch_2d, pos[:2]
            )
            
            # Create 3D array of similar patches
            similar_patches = []
            for sim_pos in similar_positions:
                sim_patch = image[sim_pos[0]:sim_pos[0]+self.patch_size,
                                sim_pos[1]:sim_pos[1]+self.patch_size]
                similar_patches.append(sim_patch)
            
            if len(similar_patches) > 1:
                similar_patches = np.array(similar_patches)
                
                # 3D DCT transform
                dct_3d = self.apply_3d_dct(similar_patches)
                
                # Wiener filtering in transform domain
                filtered_dct = self.wiener_filter_3d(dct_3d, noise_variance=0.05)
                
                # Inverse 3D DCT
                filtered_group = self.inverse_3d_dct(filtered_dct)
                
                # Take the reference patch (first in group)
                filtered_patches.append(filtered_group[0].flatten())
            else:
                # Fallback: simple denoising
                filtered_patch = cv2.GaussianBlur(patch_2d, (3, 3), 0.5)
                filtered_patches.append(filtered_patch.flatten())
        
        return np.array(filtered_patches)

class AdaptiveAggregation:
    """Performs adaptive weighted aggregation of patches"""
    
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
    
    def compute_confidence_weights(self, original_patches, denoised_patches, patch_size=8):
        """Compute confidence weights for aggregation
        
        Args:
            original_patches: Original patch array
            denoised_patches: Denoised patch array
            patch_size: Size of patches
            
        Returns:
            weights: Confidence weights
        """
        weights = []
        
        for orig, denoised in zip(original_patches, denoised_patches):
            orig_2d = orig.reshape(patch_size, patch_size)
            
            # Local variance as confidence measure
            local_variance = np.var(orig_2d)
            
            # Gradient magnitude as edge indicator
            grad_x = np.gradient(orig_2d, axis=1)
            grad_y = np.gradient(orig_2d, axis=0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2).mean()
            
            # Noise reduction quality (inverse of MSE)
            mse = np.mean((orig - denoised)**2)
            quality = 1.0 / (1.0 + mse)
            
            # Combine metrics
            confidence = 0.4 * np.tanh(local_variance/100) + \
                        0.3 * np.tanh(grad_mag/50) + \
                        0.3 * quality
            
            weights.append(max(0.1, confidence))  # Minimum weight threshold
        
        return np.array(weights)
    
    def weighted_aggregation(self, patches, positions, weights, image_shape):
        """Perform weighted aggregation with confidence
        
        Args:
            patches: Denoised patches
            positions: Patch positions
            weights: Confidence weights
            image_shape: Shape of output image
            
        Returns:
            reconstructed: Reconstructed image
        """
        reconstructed = np.zeros(image_shape, dtype=np.float64)
        weight_map = np.zeros(image_shape, dtype=np.float64)
        
        for patch, pos, weight in zip(patches, positions, weights):
            i, j, patch_size = pos
            patch_2d = patch.reshape(patch_size, patch_size)
            
            # Apply confidence weight
            weighted_patch = patch_2d * weight
            
            # Add to reconstruction with boundary checking
            end_i = min(i + patch_size, image_shape[0])
            end_j = min(j + patch_size, image_shape[1])
            patch_h = end_i - i
            patch_w = end_j - j
            
            reconstructed[i:end_i, j:end_j] += weighted_patch[:patch_h, :patch_w]
            weight_map[i:end_i, j:end_j] += weight
        
        # Normalize by weight map
        weight_map[weight_map < 1e-6] = 1.0  # Avoid division by zero
        reconstructed = reconstructed / weight_map
        
        return reconstructed

class PatchBasedDenoiser:
    """Main class for patch-based statistical denoising"""
    
    def __init__(self, patch_sizes=[8], n_clusters=32):
        self.patch_extractor = PatchExtractor(patch_sizes=patch_sizes)
        self.patch_modeler = StatisticalPatchModeling(n_clusters=n_clusters)
        self.collaborative_filter = CollaborativeFiltering(patch_size=patch_sizes[0])
        self.aggregator = AdaptiveAggregation()
    
    def post_process_enhancement(self, denoised_image, original_image, alpha=0.1):
        """Post-processing enhancement
        
        Args:
            denoised_image: Denoised image
            original_image: Original noisy image
            alpha: Blending factor
            
        Returns:
            enhanced_image: Enhanced image
        """
        # Edge-preserving smoothing
        enhanced = cv2.bilateralFilter(
            denoised_image.astype(np.float32), 5, 50, 50
        )
        
        # Blend with original for detail preservation
        enhanced = (1 - alpha) * enhanced + alpha * original_image
        
        return enhanced
    
    def process_channel(self, channel, iterations=2):
        """Process a single color channel
        
        Args:
            channel: Single channel image (grayscale)
            iterations: Number of denoising iterations
            
        Returns:
            denoised_channel: Denoised channel
        """
        current_channel = channel.astype(np.float64)
        
        for iteration in range(iterations):
            # 1. Extract patches
            patches, positions = self.patch_extractor.extract_patches(current_channel)
            
            if len(patches) == 0:
                continue
                
            # 2. Statistical modeling and clustering
            cluster_labels = self.patch_modeler.cluster_and_model_patches(patches)
            
            # 3. Collaborative filtering
            filtered_patches = self.collaborative_filter.collaborative_filter(
                current_channel, patches, positions
            )
            
            # 4. Compute confidence weights
            weights = self.aggregator.compute_confidence_weights(
                patches, filtered_patches, 
                patch_size=self.patch_extractor.patch_sizes[0]
            )
            
            # 5. Weighted aggregation
            current_channel = self.aggregator.weighted_aggregation(
                filtered_patches, positions, weights, current_channel.shape
            )
            
            # 6. Post-processing enhancement
            current_channel = self.post_process_enhancement(
                current_channel, channel, alpha=0.05
            )
        
        return np.clip(current_channel, 0, 255).astype(np.uint8)
    
    def process(self, rgb_image, iterations=2):
        """Main processing pipeline for RGB images
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            iterations: Number of denoising iterations
            
        Returns:
            denoised_image: Denoised RGB image
        """
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Input must be an RGB image with shape (H, W, 3)")
        
        # Process each color channel separately
        denoised_channels = []
        
        for c in range(3):
            channel = rgb_image[:, :, c]
            denoised_channel = self.process_channel(channel, iterations)
            denoised_channels.append(denoised_channel)
        
        # Combine channels
        denoised_image = np.stack(denoised_channels, axis=2)
        
        return denoised_image

# # Example usage and testing
# if __name__ == "__main__":
#     # Create a sample noisy image for testing
#     def create_test_image():
#         # Create a simple test pattern
#         image = np.zeros((128, 128, 3), dtype=np.uint8)
        
#         # Add some geometric shapes
#         cv2.rectangle(image, (20, 20), (60, 60), (255, 100, 100), -1)
#         cv2.circle(image, (80, 80), 20, (100, 255, 100), -1)
        
#         # Add noise
#         noise = np.random.normal(0, 25, image.shape)
#         noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
#         return image, noisy_image
    
#     # Test the denoiser
#     clean_image, noisy_image = create_test_image()
    
#     # Initialize denoiser
#     denoiser = PatchBasedDenoiser(patch_sizes=[8], n_clusters=16)
    
#     # Process the image
#     print("Processing image...")
#     denoised_image = denoiser.process(noisy_image, iterations=1)
    
#     print("Processing complete!")
#     print(f"Input shape: {noisy_image.shape}")
#     print(f"Output shape: {denoised_image.shape}")
#     print(f"Output dtype: {denoised_image.dtype}")
#     print(f"Output range: [{denoised_image.min()}, {denoised_image.max()}]")
    
    