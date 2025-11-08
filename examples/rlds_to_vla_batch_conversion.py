#!/usr/bin/env python3
"""
Batch conversion script for converting RLDS format datasets to VLA format.

This script:
1. Scans a directory for RLDS datasets
2. Loads each dataset using tensorflow_datasets
3. Converts each episode to VLA format using robodm
4. Saves converted episodes with organized naming structure

Usage:
    python rlds_to_vla_batch_conversion.py

Or modify the source_dir and target_dir variables below to match your setup.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

import robodm

# Prevent tensorflow from allocating GPU memory
tf.config.set_visible_devices([], "GPU")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _transpose_list_of_dicts(list_of_dicts):
    """Converts a list of nested dictionaries to a nested dictionary of lists.

    This is needed to convert RLDS episode steps from list format to the
    dict-of-lists format expected by robodm.

    Args:
        list_of_dicts: A list where each element is a dictionary with the same keys

    Returns:
        A dictionary where each key maps to a list of values from the input dicts
    """
    if not list_of_dicts:
        return {}

    # Base case: if the first element is not a dictionary, it's a leaf.
    if not isinstance(list_of_dicts[0], dict):
        return list_of_dicts

    dict_of_lists = {}
    # Assume all dicts in the list have the same keys as the first one.
    for key in list_of_dicts[0].keys():
        # Recursively process the values for each key.
        dict_of_lists[key] = _transpose_list_of_dicts(
            [d[key] for d in list_of_dicts])
    return dict_of_lists


def convert_episode(episode, output_path, video_codec="libx264"):
    """Convert a single RLDS episode to VLA format.

    Args:
        episode: The episode data from RLDS dataset
        output_path: Path where to save the VLA file
        video_codec: Video codec to use for encoding (default: libx264)

    Returns:
        Number of steps in the episode, or None if conversion failed
    """
    try:
        # Convert steps from list to dict-of-lists format
        steps_list = list(episode["steps"])

        if not steps_list:
            logger.warning(f"Episode is empty, skipping: {output_path}")
            return None

        episode_steps = _transpose_list_of_dicts(steps_list)
        num_steps = len(steps_list)

        # Convert and save using robodm
        robodm.Trajectory.from_dict_of_lists(
            data=episode_steps,
            path=output_path,
            video_codec=video_codec
        )

        return num_steps

    except Exception as e:
        logger.error(f"Error converting episode to {output_path}: {e}")
        return None


def find_rlds_datasets(source_dir):
    """Find all RLDS dataset directories in the source directory.

    Args:
        source_dir: Root directory containing RLDS datasets

    Returns:
        List of tuples (dataset_name, dataset_path)
    """
    datasets = []
    source_path = Path(source_dir)

    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return datasets

    # Look for directories containing dataset_info.json
    for item in source_path.iterdir():
        if item.is_dir():
            dataset_info = item / "dataset_info.json"
            if dataset_info.exists():
                datasets.append((item.name, str(item)))
                logger.info(f"Found dataset: {item.name}")

    return datasets


def convert_dataset(dataset_path, dataset_name, output_dir,
                   split="train", max_episodes=None,
                   video_codec="libx264", skip_existing=True):
    """Convert an entire RLDS dataset to VLA format.

    Args:
        dataset_path: Path to the RLDS dataset directory
        dataset_name: Name of the dataset
        output_dir: Directory where to save VLA files
        split: Dataset split to convert (default: "train")
        max_episodes: Maximum number of episodes to convert (None = all)
        video_codec: Video codec for video encoding
        skip_existing: Skip episodes that already exist as VLA files

    Returns:
        Tuple of (successful_conversions, failed_conversions)
    """
    logger.info(f"Loading dataset {dataset_name} from {dataset_path}")

    try:
        # Load the dataset using tfds
        builder = tfds.builder_from_directory(builder_dir=dataset_path)

        # Determine split string
        if max_episodes is not None:
            split_str = f"{split}[:{max_episodes}]"
        else:
            split_str = split

        ds = builder.as_dataset(split=split_str)

        # Create output directory for this dataset
        dataset_output_dir = Path(output_dir) / dataset_name / split
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert each episode
        successful = 0
        failed = 0

        logger.info(f"Converting episodes for {dataset_name} ({split} split)...")

        for episode_idx, episode in enumerate(tqdm(tfds.as_numpy(ds),
                                                   desc=f"Converting {dataset_name}")):
            output_path = dataset_output_dir / f"episode_{episode_idx:06d}.vla"

            # Skip if already exists
            if skip_existing and output_path.exists():
                logger.debug(f"Skipping existing episode: {output_path}")
                successful += 1
                continue

            num_steps = convert_episode(episode, str(output_path), video_codec)

            if num_steps is not None:
                successful += 1
                logger.debug(f"Converted episode {episode_idx} with {num_steps} steps")
            else:
                failed += 1

        logger.info(f"Dataset {dataset_name} conversion complete: "
                   f"{successful} successful, {failed} failed")

        return successful, failed

    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {e}")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert RLDS format datasets to VLA format"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/nvme0n1/xinyu/robodm/rlds",
        help="Source directory containing RLDS datasets"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/mnt/nvme0n1/xinyu/robodm/vla",
        help="Target directory for VLA files"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to convert (e.g., train test)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert per split (None = all)"
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="auto",
        help="Video codec to use for encoding"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to convert (if None, convert all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip episodes that already exist"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing episodes"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RLDS to VLA Batch Conversion")
    logger.info("=" * 70)
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Target directory: {args.target_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Max episodes per split: {args.max_episodes or 'all'}")
    logger.info(f"Video codec: {args.video_codec}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 70)

    # Find all datasets
    datasets = find_rlds_datasets(args.source_dir)

    if not datasets:
        logger.error("No RLDS datasets found in source directory")
        return

    # Filter to specific dataset if requested
    if args.dataset:
        datasets = [(name, path) for name, path in datasets if name == args.dataset]
        if not datasets:
            logger.error(f"Dataset '{args.dataset}' not found")
            return

    logger.info(f"Found {len(datasets)} dataset(s) to convert")

    # Convert each dataset
    total_successful = 0
    total_failed = 0

    for dataset_name, dataset_path in datasets:
        logger.info(f"\nProcessing dataset: {dataset_name}")
        logger.info("-" * 70)

        for split in args.splits:
            successful, failed = convert_dataset(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                output_dir=args.target_dir,
                split=split,
                max_episodes=args.max_episodes,
                video_codec=args.video_codec,
                skip_existing=args.skip_existing
            )

            total_successful += successful
            total_failed += failed

    logger.info("\n" + "=" * 70)
    logger.info("Batch Conversion Complete")
    logger.info("=" * 70)
    logger.info(f"Total successful conversions: {total_successful}")
    logger.info(f"Total failed conversions: {total_failed}")
    logger.info(f"VLA files saved to: {args.target_dir}")


if __name__ == "__main__":
    main()
