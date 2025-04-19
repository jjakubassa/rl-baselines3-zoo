import os
import glob
import re
import shutil
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Tuple, Optional
import argparse

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory based on the step number."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not checkpoints:
        return None

    # Extract step numbers from filenames (assuming format like 'rl_model_10000_steps.zip')
    checkpoint_steps = []
    for checkpoint in checkpoints:
        match = re.search(r'_(\d+)_steps\.zip$', checkpoint)
        if match:
            steps = int(match.group(1))
            checkpoint_steps.append((checkpoint, steps))

    if not checkpoint_steps:
        return None

    # Return the checkpoint with the highest step number
    latest_checkpoint = max(checkpoint_steps, key=lambda x: x[1])
    return latest_checkpoint[0]

def collect_latest_checkpoints(base_path: str, algo_pattern: str = "*", copy_dir: str = None) -> pd.DataFrame:
    """
    Collect information about the latest checkpoints for all experiments.

    Args:
        base_path: Base directory containing the experiment folders
        algo_pattern: Pattern to filter algorithms (e.g., "ppo", "sac", etc.)
        copy_dir: Directory to copy checkpoints to (with helpful renaming)

    Returns:
        DataFrame with checkpoint information
    """
    # Find all experiment folders matching the pattern
    exp_folders = glob.glob(os.path.join(base_path, f"{algo_pattern}_*_seed*"))
    exp_folders.sort()

    print(f"Found {len(exp_folders)} experiment folders")

    if copy_dir and not os.path.exists(copy_dir):
        os.makedirs(copy_dir)
        print(f"Created directory: {copy_dir}")

    checkpoint_data = []
    copied_files = []

    for folder in exp_folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split('_')

        if len(parts) < 3:
            print(f"Skipping folder with unexpected format: {folder_name}")
            continue

        algo = parts[0]
        training_seed = parts[-1].replace('seed', '')
        env_name = '_'.join(parts[1:-1])

        # Find all experiment runs for this algorithm and environment
        algo_path = os.path.join(folder, algo)
        if not os.path.exists(algo_path):
            print(f"No {algo} subfolder found in {folder}")
            continue

        exp_runs = glob.glob(os.path.join(algo_path, f"{env_name}_*"))
        if not exp_runs:
            exp_runs = [algo_path]  # If no specific runs, use the main folder

        for exp_run in exp_runs:
            exp_id = os.path.basename(exp_run).split('_')[-1] if '_' in os.path.basename(exp_run) else '0'

            # Check for best_model.zip first
            best_model_path = os.path.join(exp_run, "best_model.zip")
            have_best_model = os.path.exists(best_model_path)

            # Then look for the latest checkpoint
            latest_checkpoint = find_latest_checkpoint(exp_run)

            # If no checkpoints found, check if there's a 'model.zip'
            final_model_path = os.path.join(exp_run, "model.zip")
            have_final_model = os.path.exists(final_model_path)

            # Process best model if available
            if have_best_model:
                checkpoint_info = {
                    "algo": algo,
                    "env": env_name,
                    "training_seed": training_seed,
                    "exp_id": exp_id,
                    "checkpoint_path": best_model_path,
                    "training_steps": None,  # Can't determine from filename
                    "checkpoint_type": "best"
                }

                checkpoint_data.append(checkpoint_info)

                if copy_dir:
                    # Create a descriptive filename
                    new_filename = f"{algo}_{env_name}_seed{training_seed}_exp{exp_id}_best.zip"
                    new_path = os.path.join(copy_dir, new_filename)

                    try:
                        shutil.copy2(best_model_path, new_path)
                        copied_files.append((best_model_path, new_path))
                        print(f"Copied best model: {new_filename}")
                    except Exception as e:
                        print(f"Error copying {best_model_path}: {e}")

            # Process latest step checkpoint if available
            if latest_checkpoint:
                # Extract step number from checkpoint name
                step_match = re.search(r'_(\d+)_steps\.zip$', os.path.basename(latest_checkpoint))
                step_count = int(step_match.group(1)) if step_match else None

                checkpoint_info = {
                    "algo": algo,
                    "env": env_name,
                    "training_seed": training_seed,
                    "exp_id": exp_id,
                    "checkpoint_path": latest_checkpoint,
                    "training_steps": step_count,
                    "checkpoint_type": "latest"
                }

                checkpoint_data.append(checkpoint_info)

                if copy_dir:
                    # Create a descriptive filename
                    new_filename = f"{algo}_{env_name}_seed{training_seed}_exp{exp_id}_steps{step_count}.zip"
                    new_path = os.path.join(copy_dir, new_filename)

                    try:
                        shutil.copy2(latest_checkpoint, new_path)
                        copied_files.append((latest_checkpoint, new_path))
                        print(f"Copied latest checkpoint: {new_filename}")
                    except Exception as e:
                        print(f"Error copying {latest_checkpoint}: {e}")

            # Process final model if no other checkpoints are available
            elif not have_best_model and have_final_model:
                checkpoint_info = {
                    "algo": algo,
                    "env": env_name,
                    "training_seed": training_seed,
                    "exp_id": exp_id,
                    "checkpoint_path": final_model_path,
                    "training_steps": None,
                    "checkpoint_type": "final"
                }

                checkpoint_data.append(checkpoint_info)

                if copy_dir:
                    # Create a descriptive filename
                    new_filename = f"{algo}_{env_name}_seed{training_seed}_exp{exp_id}_final.zip"
                    new_path = os.path.join(copy_dir, new_filename)

                    try:
                        shutil.copy2(final_model_path, new_path)
                        copied_files.append((final_model_path, new_path))
                        print(f"Copied final model: {new_filename}")
                    except Exception as e:
                        print(f"Error copying {final_model_path}: {e}")

            if not (have_best_model or latest_checkpoint or have_final_model):
                print(f"No checkpoint found in {exp_run}")

    if checkpoint_data:
        # Convert to DataFrame and return
        df = pd.DataFrame(checkpoint_data)

        if copy_dir:
            print(f"\nCopied {len(copied_files)} checkpoint files to {copy_dir}")
            # Add the new copied path to the DataFrame
            copied_paths_map = {orig: new for orig, new in copied_files}
            df["copied_path"] = df["checkpoint_path"].map(copied_paths_map)

        return df
    else:
        print("No checkpoints found!")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Collect latest checkpoints from training runs")
    parser.add_argument("--logs-dir", type=str, default="./logs", help="Directory containing training logs")
    parser.add_argument("--algo", type=str, default="*", help="Algorithm filter (default: all)")
    parser.add_argument("--output", type=str, default="latest_checkpoints.csv", help="Output file path")
    parser.add_argument("--copy-dir", type=str, help="Directory to copy checkpoints to with helpful renaming")
    parser.add_argument("--only-best", action="store_true", help="Copy only best models")
    parser.add_argument("--only-latest", action="store_true", help="Copy only latest checkpoints")

    args = parser.parse_args()

    print(f"Scanning for checkpoints in {args.logs_dir}...")
    checkpoints_df = collect_latest_checkpoints(args.logs_dir, args.algo, args.copy_dir)

    if not checkpoints_df.empty:
        # Filter checkpoints based on command-line arguments if needed
        if args.copy_dir:
            # Create a new DataFrame with only the checkpoints we want to keep in the output file
            if args.only_best:
                filtered_df = checkpoints_df[checkpoints_df["checkpoint_type"] == "best"]
                if not filtered_df.empty:
                    checkpoints_df = filtered_df
                else:
                    print("Warning: No best models found, keeping all checkpoints")
            elif args.only_latest:
                filtered_df = checkpoints_df[checkpoints_df["checkpoint_type"] == "latest"]
                if not filtered_df.empty:
                    checkpoints_df = filtered_df
                else:
                    print("Warning: No latest checkpoints found, keeping all checkpoints")

        checkpoints_df.to_csv(args.output, index=False)
        print(f"Saved checkpoint information to {args.output}")

        # Print summary
        print("\nCheckpoint Summary:")
        print(f"Total checkpoints found: {len(checkpoints_df)}")
        print("\nCheckpoints per algorithm:")
        print(checkpoints_df.groupby('algo').size())
        print("\nCheckpoints per environment:")
        print(checkpoints_df.groupby('env').size())
        print("\nCheckpoints by type:")
        print(checkpoints_df.groupby('checkpoint_type').size())
    else:
        print("No checkpoints were found.")

if __name__ == "__main__":
    main()
