#!/usr/bin/env python3
"""
Batch conversion script for converting VLA format files to HDF5 format.

This script converts VLA files to HDF5 format with proper directory structure.
"""

import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm

from robodm.loader.hdf5 import convert_trajectory_to_hdf5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_vla_split(
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    output_dir: Path,
    compression: str = "gzip",
    compression_level: int = 9,
    skip_existing: bool = True
):
    """Convert VLA files from a specific split to HDF5 format.

    Args:
        dataset_dir: Dataset directory containing split subdirectories
        dataset_name: Name of the dataset
        split: Split name (e.g., "train", "test")
        output_dir: Output directory for HDF5 files
        compression: Compression algorithm (default: gzip)
        compression_level: Compression level 0-9 (default: 9)
        skip_existing: Skip files that already exist (default: True)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    split_path = dataset_dir / split

    if not split_path.exists():
        logger.warning(f"Split directory does not exist: {split_path}")
        return 0, 0

    # Find all VLA files in this split
    vla_files = sorted(split_path.glob("*.vla"))

    if not vla_files:
        logger.warning(f"No VLA files found in {split_path}")
        return 0, 0

    logger.info(f"Found {len(vla_files)} VLA files in {dataset_name}/{split}")

    # Create output directory for this dataset/split
    output_split_dir = output_dir / dataset_name / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for vla_file in tqdm(vla_files, desc=f"Converting {dataset_name}/{split}"):
        # Output file has same name but .h5 extension
        output_file = output_split_dir / vla_file.with_suffix('.h5').name

        # Skip if exists
        if skip_existing and output_file.exists():
            logger.debug(f"Skipping existing file: {output_file}")
            successful += 1
            continue

        try:
            convert_trajectory_to_hdf5(
                str(vla_file),
                str(output_file),
                compression=compression,
                compression_opts=compression_level
            )
            successful += 1
        except Exception as e:
            logger.error(f"Failed to convert {vla_file}: {e}")
            failed += 1

    return successful, failed


def convert_vla_directory(
    input_dir: str,
    output_dir: str,
    splits: list,
    compression: str = "gzip",
    compression_level: int = 9,
    skip_existing: bool = True
):
    """Convert VLA files in a directory to HDF5 format.

    Args:
        input_dir: Input directory containing VLA dataset directories
        output_dir: Output directory for HDF5 files
        splits: List of splits to convert (e.g., ["train", "test"])
        compression: Compression algorithm (default: gzip)
        compression_level: Compression level 0-9 (default: 9)
        skip_existing: Skip files that already exist (default: True)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 0, 0

    total_successful = 0
    total_failed = 0

    # Find all dataset directories (directories containing split subdirectories)
    for dataset_dir in sorted(input_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        logger.info(f"\nProcessing dataset: {dataset_name}")
        logger.info("-" * 70)

        # Convert each requested split
        for split in splits:
            successful, failed = convert_vla_split(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                split=split,
                output_dir=output_path,
                compression=compression,
                compression_level=compression_level,
                skip_existing=skip_existing
            )
            total_successful += successful
            total_failed += failed

    return total_successful, total_failed


def main():
    parser = argparse.ArgumentParser(
        description="Convert VLA format files to HDF5 format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/nvme0n1/xinyu/robodm/vla",
        help="Input directory containing VLA dataset directories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/nvme0n1/xinyu/robodm/hdf5",
        help="Output directory for HDF5 files"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to convert (e.g., train test). Default: train"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression algorithm (default: gzip)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="Compression level 0-9 (default: 9)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing files"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("VLA to HDF5 Batch Conversion")
    logger.info("=" * 70)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Compression: {args.compression} (level {args.compression_level})")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 70)

    successful, failed = convert_vla_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        compression=args.compression,
        compression_level=args.compression_level,
        skip_existing=args.skip_existing
    )

    logger.info("\n" + "=" * 70)
    logger.info("Conversion Complete")
    logger.info("=" * 70)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
