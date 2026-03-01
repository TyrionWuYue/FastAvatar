#!/usr/bin/env python3
"""
Generate frame groups JSON file for training dataset.

This script scans the dataset directory structure and generates a JSON file
containing frame groups (input and target frame indices) for training.

Usage:
    python scripts/generate_frame_groups.py --config configs/train/train.yaml
"""

import argparse
import json
import os
import sys
from omegaconf import OmegaConf

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FastAvatar.utils.preprocess_dataset import (
    JsonStreamWriter,
    process_dataset_monocular,
    process_dataset_unified
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate frame groups JSON file for training dataset"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to train.yaml config file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regenerate even if JSON exists'
    )
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found")
        sys.exit(1)
    
    cfg = OmegaConf.load(args.config)
    
    if not hasattr(cfg, 'dataset'):
        print("Error: No dataset configuration found in config file")
        sys.exit(1)
    
    meta_path = cfg.dataset.meta_path
    
    # Check if meta file already exists
    if os.path.exists(meta_path) and not args.force:
        # Check if empty or valid json
        try:
            with open(meta_path, 'r') as f:
                # Just read first char to check emptiness
                if f.read(1):
                    print(f"Meta file {meta_path} already exists. Use --force to regenerate.")
                    return
        except Exception:
            pass

    max_input_frames = getattr(cfg.dataset, 'input_frames', 300)
    target_frames = getattr(cfg.dataset, 'target_frames', 16)
    seed = getattr(cfg.experiment, 'seed', 42)
    multiply = getattr(cfg.dataset, 'multiply', 1)

    print(f"Frame groups generation configuration:")
    print(f"  Config file: {args.config}")
    print(f"  Meta path: {meta_path}")
    print(f"  Max input frames: {max_input_frames}")
    print(f"  Target frames: {target_frames}")
    print(f"  Multiply: {multiply}")
    print(f"  Seed: {seed}")
    
    # Collect datasets to process
    datasets_to_process = []
    
    if hasattr(cfg.dataset, 'datasets') and cfg.dataset.datasets:
        for name, dataset_cfg in cfg.dataset.datasets.items():
            datasets_to_process.append({
                "name": name,
                "root_dir": dataset_cfg.root_dir,
                "task_type": dataset_cfg.task_type
            })
    else:
        # Single dataset fallback
        datasets_to_process.append({
            "name": "default",
            "root_dir": cfg.dataset.root_dir,
            "task_type": getattr(cfg.dataset, 'task_type', 'monocular')
        })

    # STREAMING WRITER
    print(f"writing to {meta_path}...")
    try:
        with JsonStreamWriter(meta_path) as writer:
            for i, ds in enumerate(datasets_to_process):
                root_dir = ds["root_dir"]
                task_type = ds["task_type"]
                name = ds["name"]
                
                # Vary seed for each dataset to avoid identical sequences (e.g. always starting with 2)
                current_seed = seed + i if seed is not None else None
                
                print(f"\nProcessing dataset [{name}]: {root_dir} -> {task_type}")
                
                if not os.path.exists(root_dir):
                    print(f"Error: Root directory {root_dir} not found for dataset {name}")
                    continue

                if task_type == "monocular":
                    process_dataset_monocular(
                        root_dir=root_dir,
                        output_writer=writer,
                        max_input_frames=max_input_frames,
                        target_frames=target_frames,
                        seed=seed,
                        dataset_name=name,
                        multiply=multiply
                    )
                elif task_type == "unified":
                    process_dataset_unified(
                        root_dir=root_dir,
                        output_writer=writer,
                        max_input_frames=max_input_frames,
                        target_frames=target_frames,
                        seed=seed,
                        dataset_name=name,
                        multiply=multiply
                    )
                else:
                    print(f"Unknown task_type: {task_type}, skipping.")

            print(f"\nTotal groups generated: {writer.get_count()}")

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        # Clean up partial file
        if os.path.exists(meta_path):
            os.remove(meta_path)
        sys.exit(1)

    print(f"Successfully generated {meta_path}")


if __name__ == "__main__":
    main()
