import glob
import os
import pandas as pd
from huggingface_sb3 import EnvironmentName
from rl_zoo3.enjoy import enjoy
from multiprocessing import Pool, cpu_count
import itertools

def evaluate_single_config(config):
    folder, env_name, algo, training_seed, exp_id, eval_seed, n_timesteps = config

    try:
        # Set up command line arguments programmatically
        import sys
        sys.argv = [
            "enjoy.py",
            "--env", env_name,
            "--algo", algo,
            "-f", folder,
            "--no-render",
            "--load-last-checkpoint",
            "--seed", str(eval_seed),
            "-n", str(n_timesteps),
            "--deterministic",
            "--exp-id", exp_id if exp_id.isdigit() else "0"
        ]

        # Run evaluation
        results = enjoy()

        if results is not None:
            results['training_seed'] = training_seed
            results['exp_run'] = exp_id
            results['eval_seed'] = eval_seed
            print(f"Successfully evaluated {os.path.basename(folder)} - Experiment run {exp_id} with seed {eval_seed}")
            return results
        else:
            print(f"No results returned for {os.path.basename(folder)} - Experiment run {exp_id} with seed {eval_seed}")
            return None

    except Exception as e:
        print(f"Error evaluating {os.path.basename(folder)} - Experiment run {exp_id} with seed {eval_seed}: {str(e)}")
        return None

def evaluate_all_experiments():
    base_path = "./logs"
    eval_seeds = [42, 43, 44, 45, 46]  # List of evaluation seeds
    n_timesteps = 150  # Single episode length
    n_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    # n_processes = 4

    # Find all experiment folders
    exp_folders = glob.glob(os.path.join(base_path, "ppo_*_seed*"))
    exp_folders.sort()

    print(f"Found {len(exp_folders)} experiment folders:")
    for folder in exp_folders:
        print(f"  {os.path.basename(folder)}")

    # Prepare all configurations to evaluate
    configs = []

    for folder in exp_folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split('_')

        algo = parts[0]  # ppo
        training_seed = parts[-1].replace('seed', '')  # seed number
        env_name = '_'.join(parts[1:-1])

        # Find all experiment runs
        exp_runs = glob.glob(os.path.join(folder, "ppo", f"{env_name}_*"))
        if not exp_runs:
            exp_runs = [folder]  # If no subfolders, use the main folder

        for exp_run in exp_runs:
            exp_id = os.path.basename(exp_run).split('_')[-1] if '_' in os.path.basename(exp_run) else 'main'

            # Add a configuration for each evaluation seed
            for eval_seed in eval_seeds:
                configs.append((folder, env_name, algo, training_seed, exp_id, eval_seed, n_timesteps))

    print(f"\nPrepared {len(configs)} configurations to evaluate")
    print(f"Using {n_processes} processes for parallel evaluation")

    # Run evaluations in parallel
    all_results = []
    with Pool(processes=n_processes) as pool:
        results = list(pool.imap_unordered(evaluate_single_config, configs))
        all_results = [r for r in results if r is not None]

    if all_results:
        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)

        # Save results
        output_file = "all_experiments_results.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        # Print summary
        print("\nSummary of results:")
        summary = final_df.groupby(['env', 'training_seed', 'exp_run', 'eval_seed'])['episode_reward'].agg(['mean', 'std', 'count'])
        print(summary)

        # Additional statistics
        print("\nAverage performance across evaluation seeds:")
        avg_stats = final_df.groupby(['env', 'training_seed', 'exp_run'])['episode_reward'].agg(['mean', 'std', 'count'])
        print(avg_stats)

        return final_df
    else:
        print("No results were collected!")
        return None

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    evaluate_all_experiments()
