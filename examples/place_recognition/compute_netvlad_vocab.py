import os
import random
import sys
import argparse
import cv2
import kornia
import tqdm
from pathlib import Path
import torch
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/surfslam")

from examples.tbnms.tbnms_calibration import TBNMSCalibration
from examples.utils.h5_log_reader import H5LogReader
from surfslam.common.settings import Settings
from examples.utils.pose_utils import load_calibration
from surfslam.common.dataset_paths import load_config, resolve_config_path
from surfslam.mapping.place_recognition.image_descriptor import ImageDescriptorManager
from surfslam.mapping.descriptors.matcher import SuperGlueMatcher


def load_vocab_config(config_path: str) -> dict:
    """Load and validate the vocabulary fitting configuration."""
    config = load_config(config_path)


    required_keys = ['baseline', 'all_data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")

    return config


def load_image_files(data_root: str, dataset_path: str, dataset_family: str = None):
    """Load image file paths for a dataset.

    Returns:
        List of image paths (mono) or list of (left, right) tuples (stereo)
    """
    full_path = os.path.join(data_root, dataset_path)

    if dataset_family == 'tbnms':
        # Stereo dataset
        left_dir = os.path.join(full_path, "raw_left")
        right_dir = os.path.join(full_path, "raw_right")

        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            print(f"  Warning: Stereo directories not found for {dataset_path}")
            return []

        left_frames = sorted(os.listdir(left_dir))
        image_pairs = [
            (os.path.join(left_dir, f), os.path.join(right_dir, f))
            for f in left_frames
        ]
        return image_pairs
    else:
        # Monocular dataset - look for common image extensions
        if not os.path.exists(full_path):
            print(f"  Warning: Directory not found: {full_path}")
            return []

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.tiff', '*.TIFF']:
            image_files.extend(Path(full_path).glob(ext))

        return sorted([str(f) for f in image_files])


def process_image(img_path_or_tuple, calibration=None, target_width=640):
    """Process an image: load, resize, and optionally rectify.

    Args:
        img_path_or_tuple: Either a single image path (mono) or (left, right) tuple (stereo)
        calibration: Calibration object for stereo rectification (None for mono)
        target_width: Target width for resized images

    Returns:
        Processed image (grayscale)
    """
    if isinstance(img_path_or_tuple, tuple):
        # Stereo case
        left_path, right_path = img_path_or_tuple
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            return None

        if calibration is not None:
            # Rectify stereo images - compute scale factor from target width
            h, w = left_img.shape[:2]
            scale_factor = target_width / w
            leftRect, rightRect = calibration.rectify_images(left_img, right_img, scale_factor)
            return leftRect
        else:
            # Just resize left image
            h, w = left_img.shape[:2]
            scale_factor = target_width / w
            new_h = int(h * scale_factor)
            resized = cv2.resize(left_img, (target_width, new_h))
            return resized
    else:
        # Monocular case
        img = cv2.imread(img_path_or_tuple)
        if img is None:
            return None

        # Resize image
        h, w = img.shape[:2]
        scale_factor = target_width / w
        new_h = int(h * scale_factor)
        resized = cv2.resize(img, (target_width, new_h))
        return resized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit VLAD vocabulary from multiple datasets specified in a single config file"
    )
    parser.add_argument("config", help="Path to vocabulary fitting config YAML file")
    args = parser.parse_args()

    # Load configuration
    config = load_vocab_config(args.config)

    # Extract parameters from config with defaults
    output_path = config.get('output_path', 'vlad_vocab.npz')
    frame_skip = config.get('frame_skip', 1)
    max_descriptors = config.get('max_descriptors', 100000)
    max_keypoints_per_image = config.get('max_keypoints_per_image', 1024)
    max_imgs_per_sequence = config.get('max_imgs_per_sequence', None)
    max_imgs_per_dataset = config.get('max_imgs_per_dataset', None)
    target_im_width = config.get('target_im_width', 640)
    device = config.get('device', 'cuda:0')

    # Load baseline settings
    baseline_settings_path = resolve_config_path(config['baseline'])
    settings = Settings.load_from_file(baseline_settings_path)

    # Apply any global setting changes
    if 'settings_changes' in config and config['settings_changes'] is not None:
        settings.augment(config['settings_changes'])

    # Initialize descriptor and matcher
    descriptor = ImageDescriptorManager(
        settings.mapper.registration.place_recognition.image_descriptors,
        device
    )

    # We'll create a matcher per dataset family (since calibration may differ)
    dataset_descriptors = []  # List of descriptor arrays, one per dataset

    # Process each dataset family
    for family_config in config['all_data']:
        dataset_family = family_config.get('dataset_family', None)
        data_root = family_config['data_root']
        datasets = family_config['datasets']
        ignore_pattern = family_config.get('ignore_pattern', None)

        # Load calibration if available
        calibration = None
        if 'calibration' in family_config:
            calibration = load_calibration(dataset_family, family_config['calibration'])

        # Compile ignore pattern if provided
        import re
        ignore_regex = re.compile(ignore_pattern) if ignore_pattern else None

        print(f"\n{'='*80}")
        print(f"Processing dataset family: {dataset_family or 'monocular'}")
        print(f"Data root: {data_root}")
        print(f"Number of sequences: {len(datasets)}")
        print('='*80)

        # Process each dataset in this family
        for dataset_idx, dataset_path in enumerate(datasets):
            print(f"\n{'='*60}")
            print(f"Processing sequence {dataset_idx+1}/{len(datasets)}: {dataset_path}")
            print('='*60)

            # Load image files
            image_files = load_image_files(data_root, dataset_path, dataset_family)

            if not image_files:
                raise RuntimeError(f"No images found for dataset", dataset_path)

            # Filter out images matching ignore pattern
            if ignore_regex:
                original_count = len(image_files)
                image_files = [
                    img for img in image_files
                    if not ignore_regex.search(str(img) if not isinstance(img, tuple) else str(img[0]))
                ]
                filtered_count = original_count - len(image_files)
                if filtered_count > 0:
                    print(f"  Filtered out {filtered_count} images matching ignore pattern")
                if not image_files:
                    raise RuntimeError(f"After filtering, no images found for dataset", dataset_path)

            # Apply max_imgs_per_sequence if specified
            if max_imgs_per_sequence is not None and len(image_files) > max_imgs_per_sequence:
                np.random.shuffle(image_files)
                image_files = image_files[:max_imgs_per_sequence]
                print(f"  Limited to {max_imgs_per_sequence} images per sequence")

            # Create matcher if we have calibration
            matcher = None
            if calibration is not None:
                # Compute scale factor from target width (use a reference image to get original width)
                if image_files:
                    ref_img_path = image_files[0][0] if isinstance(image_files[0], tuple) else image_files[0]
                    ref_img = cv2.imread(ref_img_path)
                    if ref_img is not None:
                        _, w = ref_img.shape[:2]
                        scale_factor = target_im_width / w
                    else:
                        scale_factor = 0.5  # fallback
                else:
                    scale_factor = 0.5  # fallback

                matcher = SuperGlueMatcher(
                    settings.mapper.registration.matching,
                    calibration.to_dict(scale_factor),
                    descriptor
                )
            else:
                # Create a simple matcher without calibration
                matcher = SuperGlueMatcher(
                    settings.mapper.registration.matching,
                    None,
                    descriptor
                )

            current_dataset_descriptors = []
            processed_count = 0
            dataset_img_count = 0

            # Process images
            for frame_idx, img_file in enumerate(tqdm.tqdm(image_files)):
                # Apply frame skip
                if frame_idx % frame_skip != 0:
                    continue

                # Check max_imgs_per_dataset limit
                if max_imgs_per_dataset is not None and dataset_img_count >= max_imgs_per_dataset:
                    print(f"  Reached max_imgs_per_dataset limit ({max_imgs_per_dataset})")
                    break

                # Load and process image
                img = process_image(img_file, calibration, target_im_width)
                if img is None:
                    continue

                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(gray).float().to(device) / 255.0
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

                with torch.no_grad():
                    img_torch = torch.from_numpy(img).permute(2,0,1)
                    img_tensor = matcher.equalize(img_torch)
                    img_gray = kornia.color.rgb_to_grayscale(img_tensor.unsqueeze(0)).squeeze()
                    pred = matcher.extract_features(img_gray[None,None])

                # Visualization for validation
                if processed_count % 100 == 0:  # Visualize every 100th frame
                    keypoints = pred['keypoints'][0].cpu().numpy()
                    scores = pred['scores'][0].cpu().numpy()

                    # Create visualization
                    vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else img.copy()
                    for kp, score in zip(keypoints, scores):
                        x, y = int(kp[0]), int(kp[1])
                        color = (0, int(255 * score), int(255 * (1 - score)))
                        cv2.circle(vis_img, (x, y), 3, color, -1)

                    cv2.imshow('SuperPoint Features', vis_img)
                    cv2.waitKey(1)
                    
                descriptors = pred['descriptors'][0].cpu().numpy().T  # Shape: (N, 256)

                # Limit keypoints per image
                if len(descriptors) > max_keypoints_per_image:
                    scores = pred['scores'][0].cpu().numpy()
                    top_indices = np.argsort(scores)[-max_keypoints_per_image:]
                    descriptors = descriptors[top_indices]

                if len(descriptors) > 0:
                    current_dataset_descriptors.append(descriptors)

                processed_count += 1
                dataset_img_count += 1

                # Progress update
                if processed_count % 50 == 0:
                    total_desc = sum(len(d) for d in current_dataset_descriptors)
                    print(f"  Processed {processed_count} frames, {total_desc} descriptors collected")

            # Combine all descriptors from this dataset
            if current_dataset_descriptors:
                combined = np.vstack(current_dataset_descriptors)
                dataset_descriptors.append(combined)
                print(f"  Finished dataset: {processed_count} frames, {len(combined)} descriptors")
            else:
                print(f"  Finished dataset: {processed_count} frames, 0 descriptors")

    # Downsample each dataset proportionally, then combine
    num_datasets = len(dataset_descriptors)
    if num_datasets == 0:
        print("ERROR: No descriptors collected from any dataset!")
        sys.exit(1)

    per_dataset_target = max_descriptors // num_datasets

    print(f"\n{'='*60}")
    print(f"Downsampling: {per_dataset_target} descriptors per dataset")
    print('='*60)

    downsampled = []
    for i, desc in enumerate(dataset_descriptors):
        if len(desc) > per_dataset_target:
            indices = np.random.choice(len(desc), per_dataset_target, replace=False)
            downsampled.append(desc[indices])
            print(f"  Dataset {i}: {len(desc)} -> {per_dataset_target}")
        else:
            downsampled.append(desc)
            print(f"  Dataset {i}: {len(desc)} (kept all)")

    all_descriptors = np.vstack(downsampled)

    # Final downsample if still over limit
    if len(all_descriptors) > max_descriptors:
        indices = np.random.choice(len(all_descriptors), max_descriptors, replace=False)
        all_descriptors = all_descriptors[indices]
        print(f"  Final downsample: {len(all_descriptors)} descriptors")

    # Fit the vocabulary
    total_sequences = sum(len(family['datasets']) for family in config['all_data'])
    print(f"\n{'='*60}")
    print(f"Fitting vocabulary with {len(all_descriptors)} total descriptors")
    print(f"From {total_sequences} sequences across {len(config['all_data'])} dataset families")
    print('='*60)

    descriptor.fit([all_descriptors], max_descriptors=max_descriptors)

    # Save the fitted vocabulary
    descriptor.save(output_path)
    print(f"\nVocabulary saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Clusters: {descriptor.n_clusters}")
    print(f"  Descriptor dim: {descriptor.descriptor_dim}")
    print(f"  Embedding dim: {descriptor.embedding_dim}")
    print('='*60)