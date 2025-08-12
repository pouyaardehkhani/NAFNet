import os
import itertools
import cv2
import numpy as np
from typing import List, Dict, Any, Callable

from camera_noise import apply_camera_noise
from frequency_noise import apply_frequency_noise
from impulse_noise import apply_impulse_noise
from math_noise import apply_mathematical_noise
from additive_noise import apply_additive_noise
from speckle_noise import apply_speckle_noise
from spatial_noise import apply_spatial_pattern
from structured_noise import apply_structured_noise
from motion_noise import apply_motion_effect

TEST_CONFIG = [
    # 1. Camera Noise
    {'name': 'camera', 'func': apply_camera_noise, 'params': {'noise_type': 'iso', 'noise_params': {'iso_level': [400, 1600], 'dark_noise': [0.02, 0.08]}}},
    {'name': 'camera', 'func': apply_camera_noise, 'params': {'noise_type': 'chromatic', 'noise_params': {'fringe_strength': [0.1, 0.4], 'blur_radius': [0.8, 1.5]}}},
    
    # 2. Frequency Noise
    {'name': 'frequency', 'func': apply_frequency_noise, 'params': {'noise_type': ['white', 'pink', 'blue', 'brown'], 'intensity': [0.1, 0.4], 'blend_mode': ['add', 'multiply', 'overlay']}},
    
    # 3. Impulse Noise
    {'name': 'impulse', 'func': apply_impulse_noise, 'params': {'noise_type': 'shot', 'shot_intensity': [0.05, 0.15]}},
    {'name': 'impulse', 'func': apply_impulse_noise, 'params': {'noise_type': 'quantization', 'quant_bits': [2, 4]}},
    {'name': 'impulse', 'func': apply_impulse_noise, 'params': {'noise_type': 'both', 'shot_intensity': [0.05, 0.15], 'quant_bits': [3, 5]}},
    
    # 4. Mathematical Noise
    {'name': 'math', 'func': apply_mathematical_noise, 'params': {'noise_type': 'awgn', 'sigma': [15, 40]}},
    {'name': 'math', 'func': apply_mathematical_noise, 'params': {'noise_type': 'rayleigh', 'scale': [20, 45]}},
    {'name': 'math', 'func': apply_mathematical_noise, 'params': {'noise_type': 'exponential', 'scale': [15, 35]}},
    {'name': 'math', 'func': apply_mathematical_noise, 'params': {'noise_type': 'gamma', 'shape': [1, 4], 'scale': [10, 25]}},

    # 5. Additive Noise
    {'name': 'additive', 'func': apply_additive_noise, 'params': {'noise_type': 'gaussian', 'std': [15, 40]}},
    {'name': 'additive', 'func': apply_additive_noise, 'params': {'noise_type': 'salt_pepper', 'salt_prob': [0.01, 0.05], 'pepper_prob': [0.01, 0.05]}},
    {'name': 'additive', 'func': apply_additive_noise, 'params': {'noise_type': 'uniform', 'low': [-20, -60], 'high': [20, 60]}},
    {'name': 'additive', 'func': apply_additive_noise, 'params': {'noise_type': ['poisson', 'speckle']}},

    # 6. Speckle Noise
    {'name': 'speckle', 'func': apply_speckle_noise, 'params': {'variance': [0.05, 0.2], 'distribution': ['normal', 'uniform', 'gamma']}},

    # 7. Spatial Patterns
    {'name': 'spatial', 'func': apply_spatial_pattern, 'params': {'pattern': 'checkerboard', 'intensity': [0.1, 0.3], 'block_size': [8, 24]}},
    {'name': 'spatial', 'func': apply_spatial_pattern, 'params': {'pattern': 'stripe', 'intensity': [0.15, 0.4], 'stripe_width': [4, 10], 'direction': ['horizontal', 'vertical', 'diagonal']}},
    {'name': 'spatial', 'func': apply_spatial_pattern, 'params': {'pattern': 'ring', 'intensity': [0.1, 0.25], 'ring_spacing': [15, 40]}},
    {'name': 'spatial', 'func': apply_spatial_pattern, 'params': {'pattern': 'moire', 'intensity': [0.2, 0.4], 'freq1': [0.05, 0.15], 'angle2': [15, 45]}},

    # 8. Structured Noise
    {'name': 'structured', 'func': apply_structured_noise, 'params': {'noise_type': 'periodic', 'frequency': [0.05, 0.2], 'amplitude': [15, 40]}},
    {'name': 'structured', 'func': apply_structured_noise, 'params': {'noise_type': 'banding', 'band_width': [5, 20], 'band_intensity': [-20, 30]}},

    # 9. Motion Effects
    {'name': 'motion', 'func': apply_motion_effect, 'params': {'effect': 'motion_blur', 'length': [15, 40], 'angle': [0, 45]}},
    {'name': 'motion', 'func': apply_motion_effect, 'params': {'effect': 'vibration', 'intensity': [1.5, 3.0], 'frequency': [20, 50]}},
    {'name': 'motion', 'func': apply_motion_effect, 'params': {'effect': 'both', 'length': [20, 50], 'angle': [30, 120], 'intensity': [2.0]}}
]

def _params_to_filename(params: Dict[str, Any]) -> str:
    """Converts parameters to the format: key1_(value1)_key2_(value2).png"""
    parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, dict):
            for n_key, n_value in sorted(value.items()):
                parts.append(f"{n_key}_({n_value})")
        else:
            parts.append(f"{key}_({value})")
    return "_".join(parts).replace(" ", "") + ".png"


def _run_test_case(
    config: Dict[str, Any],
    base_image: np.ndarray,
    noise_specific_output_dir: str
):
    """Generates all parameter combinations for a single test case and runs them."""
    test_func = config['func']
    params = config['params']

    iter_source, nested_key = (params.get('noise_params'), 'noise_params') if 'noise_params' in params else \
                              (params.get('pattern_params'), 'pattern_params') if 'pattern_params' in params else \
                              (params, None)

    iter_params = {k: v for k, v in iter_source.items() if isinstance(v, list)}
    static_params_in_source = {k: v for k, v in iter_source.items() if not isinstance(v, list)}
    
    param_combinations = [{}] if not iter_params else \
                         [dict(zip(iter_params.keys(), v)) for v in itertools.product(*iter_params.values())]
    
    top_level_static_params = {k: v for k, v in params.items() if k != nested_key}

    for combo in param_combinations:
        final_params = top_level_static_params.copy()
        current_iter_source_params = {**static_params_in_source, **combo}
        if nested_key:
            final_params[nested_key] = current_iter_source_params
        else:
            final_params.update(current_iter_source_params)

        output_filename = _params_to_filename(final_params)
        output_path = os.path.join(noise_specific_output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping existing file: {output_filename}")
            continue

        try:
            print(f"Generating: {output_filename}")
            noisy_image = test_func(base_image, **final_params)
            cv2.imwrite(output_path, noisy_image)
        except Exception as e:
            print(f"ERROR generating {output_filename}: {e}")


def generate_all_noise_variations(base_image: np.ndarray, base_image_name: str, output_dir: str):
    """Applies all configured noise types to a pre-loaded image array."""
    print(f"Starting exhaustive noise generation for '{base_image_name}'...")
    print(f"Output will be saved to subfolders inside '{output_dir}'")

    for i, config in enumerate(TEST_CONFIG):
        noise_name = config['name']
        print(f"\n--- Running Test Group {i+1}/{len(TEST_CONFIG)}: {noise_name} ---")
        
        noise_specific_output_dir = os.path.join(output_dir, noise_name)
        os.makedirs(noise_specific_output_dir, exist_ok=True)
        
        _run_test_case(
            config=config,
            base_image=base_image,
            noise_specific_output_dir=noise_specific_output_dir
        )

    print("\n--- All noise generation tasks complete. ---")


if __name__ == '__main__':
    INPUT_IMAGE = '/home/amir_vahedi/NAFNet/images/test.jpg' 
    MASTER_OUTPUT_DIR = '/home/amir_vahedi/NAFNet/images/test'
    OUTPUT_WIDTH = 512
    OUTPUT_HEIGHT = None

    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Input image '{INPUT_IMAGE}' not found.")
        exit()
        
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        print(f"Error: Could not read image from '{INPUT_IMAGE}'.")
        exit()

    if OUTPUT_WIDTH or OUTPUT_HEIGHT:
        original_h, original_w = image.shape[:2]
        if OUTPUT_WIDTH and not OUTPUT_HEIGHT:
            new_dims = (OUTPUT_WIDTH, int(OUTPUT_WIDTH * original_h / original_w))
        elif OUTPUT_HEIGHT and not OUTPUT_WIDTH:
            new_dims = (int(OUTPUT_HEIGHT * original_w / original_h), OUTPUT_HEIGHT)
        else:
            new_dims = (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        print(f"Resizing input image from {original_w}x{original_h} to {new_dims[0]}x{new_dims[1]}...")
        image = cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)

    base_name = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
    
    generate_all_noise_variations(image, base_name, MASTER_OUTPUT_DIR)

