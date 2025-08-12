# Package Versions:
numpy : 1.24.3

cv2 : 4.11.0

scipy : 1.13.1

skimage : 0.24.0

# Help
## frequency noise
```
python noise_generator.py --img input.jpg --output output.jpg --noise_type pink --intensity 0.3 --blend_mode add
python noise_generator.py --img input.jpg --output output.jpg --noise_type blue --intensity 0.5 --blend_mode multiply
python noise_generator.py --img input.jpg --output output.jpg --noise_type brown --intensity 0.2 --blend_mode overlay
python noise_generator.py --img input.jpg --output output.jpg --noise_type white --intensity 0.1 --blend_mode add
```

## impulse noise
Adds shot noise and quantization noise to input images.

```
python impulse_noise.py --img input.jpg output_shot.jpg --type shot --shot_intensity 0.1
python impulse_noise.py --img input.jpg output_quant.jpg --type quantization --quant_bits 4
python impulse_noise.py --img input.jpg output_both.jpg --type both --shot_intensity 0.05 --quant_bits 6
python impulse_noise.py --img input.jpg output.jpg --seed 42
```

## mathematical noise models
- Additive White Gaussian Noise (AWGN)
- Rayleigh Noise
- Exponential Noise  
- Gamma Noise

```
python math_noise.py --img input.jpg --output output_awgn.jpg --noise awgn --sigma 25
python math_noise.py --img input.jpg --output output_rayleigh.jpg --noise rayleigh --scale 30
python math_noise.py --img input.jpg --output output_exponential.jpg --noise exponential --scale 20
python math_noise.py --img input.jpg --output output_gamma.jpg --noise gamma --shape 2 --scale 25
```

## additive noise
```
# Basic usage
python noise_additive.py --img input.jpg --output output.jpg --noise gaussian
python noise_additive.py --img input.jpg --output output.jpg --noise salt_pepper
python noise_additive.py --img input.jpg --output output.jpg --noise uniform
python noise_additive.py --img input.jpg --output output.jpg --noise poisson
    
# With custom parameters
python noise_additive.py --img input.jpg --output output.jpg --noise gaussian --gaussian_mean 0 --gaussian_std 50
python noise_additive.py --img input.jpg --output output.jpg --noise salt_pepper --salt_prob 0.1 --pepper_prob 0.08
python noise_additive.py --img input.jpg --output output.jpg --noise uniform --uniform_low -50 --uniform_high 50
python noise_additive.py --img input.jpg --output output.jpg --noise poisson --poisson_scale 0.8
```

## spackle noise
Speckle noise affects signal intensity and is commonly found in ultrasound and radar imaging.

```
python noise_spackle.py --img input_image.jpg --output output_image.jpg --variance 0.1
```

## Structured noise
Structured Noise Generator

Adds periodic noise and banding effects to images.

```
# Basic usage:
python structured_noise.py --img input.jpg --output noisy_output.jpg

# Apply only periodic noise:
python structured_noise.py --img input.jpg --output periodic_noise.jpg --noise_type periodic --periodic_frequency 0.05 --periodic_amplitude 25

# Apply only horizontal banding:
python structured_noise.py --img input.jpg --output banding.jpg --noise_type banding --band_width 5 --band_intensity 40 --banding_direction horizontal

# Custom periodic noise with both directions:
python structured_noise.py --img input.jpg --output complex_noise.jpg --periodic_frequency 0.15 --periodic_amplitude 30 --periodic_direction both --periodic_phase_x 1.57

# Complex structured noise with custom parameters:
python structured_noise.py --img input.jpg --output structured.jpg --noise_type both --periodic_frequency 0.08 --periodic_amplitude 15 --band_width 8 --band_intensity 25 --banding_direction vertical
```

## spatial noise
This script adds various spatial noise patterns to images including:
- Checkerboard noise: Alternating pattern of noise blocks
- Stripe noise: Horizontal or vertical bands
- Ring artifacts: Circular patterns (common in CT/MRI)
- Moiré patterns: Interference between regular structures

```
python spatial_noise.py --img input.jpg --output output.jpg --pattern checkerboard --intensity 0.3 --block_size 16
python spatial_noise.py --img input.jpg --output output.jpg --pattern stripe --intensity 0.4 --stripe_width 5 --direction horizontal
python spatial_noise.py --img input.jpg --output output.jpg --pattern ring --intensity 0.25 --ring_spacing 20 --center_x 256 --center_y 256
python spatial_noise.py --img input.jpg --output output.jpg --pattern moire --intensity 0.2 --freq1 0.1 --freq2 0.12 --angle1 0 --angle2 15
```

## motion noise
- Motion blur noise: Simulates camera/subject movement blur
- Vibration noise: Adds high-frequency shake artifacts

```
python motion_noise.py --img input.jpg --output output.jpg --effect motion_blur --length 15 --angle 45
python motion_noise.py --img input.jpg --output output.jpg --effect vibration --intensity 2.0 --frequency 50
python motion_noise.py --img input.jpg --output output.jpg --effect both --length 10 --angle 30 --intensity 1.5 --frequency 30
```

## camera noise
```
python camera_noise.py --img input.jpg --output output.jpg --noise_type iso --iso_level 800 --intensity 0.3
python camera_noise.py --img input.jpg --output output.jpg --noise_type chromatic --fringe_strength 0.2 --blur_radius 1.5
python camera_noise.py --img input.jpg --output output.jpg --noise_type iso --iso_level 1600 --dark_noise 0.1 --read_noise 0.05
python camera_noise.py --img input.jpg --output output.jpg --noise_type chromatic --fringe_strength 0.3 --color_shift 2.0
```

Parameters Available:

ISO/Sensor Noise:
- `--iso_level`: ISO sensitivity (100, 200, 400, 800, 1600, etc.)
- `--dark_noise`: Dark current noise intensity (0.0 to 0.2)
- `--read_noise`: Read noise intensity (0.0 to 0.1)
- `--shot_noise_factor`: Shot noise scaling (0.5 to 2.0)

Chromatic Noise:
- `--fringe_strength`: Chromatic fringing strength (0.0 to 1.0)
- `--color_shift`: Color channel shift in pixels (0.0 to 5.0)
- `--blur_radius`: Differential blur radius (0.5 to 3.0)
