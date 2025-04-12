import os
import submitit
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class TrainingConfig:
    """Configuration for SLURM job submission of multiple training runs."""
    # SLURM configuration
    partition: str = "cpu"
    job_name: str = "rl_training"
    cpus_per_task: int = 32
    time_minutes: str = "24:00:00"
    mem_gb: str = "320G"
    gpus_per_node: int = 0
    nodes: int = 1
    tasks_per_node: int = 1

    # Experiment configuration
    algos: tuple[str, ...] = ("ppo",)
    envs: tuple[str, ...] = ("CederFix-v0", "CederFlex-v0", "CederReplace-v0", "CederFixRandomFreq-v0", "MandlFix-v0", "MandlFlex-v0", "MandlReplace-v0", "MandlFixRandomFreq-v0", "Mumford0Fix-v0", "Mumford0FixRandomFreq-v0")
    # ("CederFix-v0", "CederFlex-v0", "CederReplace-v0", "CederFixRandomFreq-v0", "MandlFix-v0", "MandlFlex-v0", "MandlReplace-v0", "MandlFixRandomFreq-v0" "Mumford0Fix-v0", "Mumford0FixRandomFreq-v0")
    seeds: tuple[int, ...] = (1, 2, 3,)

def run_training(algo: str, env: str, seed: int):
    """Run a single training job."""
    from rl_zoo3.train import train
    import sys

    sys.argv = [
        "train.py",
        "--algo", algo,
        "--env", env,
        "--seed", str(seed),
        "--track",  # Enable W&B tracking
        "--wandb-project-name", "thesis",
        "--tensorboard-log", f"tensorboard/{algo}_{env}",
        "-f", f"logs/{algo}_{env}_seed{seed}",
        "--save-freq", "500_000",
        "-n", "10_000_000",
        "--n-eval-envs", "16",
        "--vec-env", "subproc"
        "--wandb-tags", "freq",

        # HPO
        # "--optimize",
        # "--optimization-log-path", "./log_hpo",
        # "--max-total-trials", "100",
        # "--study-name", f"{env}-{algo}",
        # "--storage", f"./log_hpo/{env}/{algo}/hpo.log",
        # "--wandb-tags", "HPO",
    ]

    # Run the training
    train()

    return f"Completed training for {algo} on {env} with seed {seed}"

def main():
    """Submit multiple training jobs to SLURM."""
    config = TrainingConfig()

    # Set up SLURM executor
    log_folder = Path("slurm_logs")
    log_folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_folder)

    # Configure SLURM parameters
    executor.update_parameters(
        slurm_partition=config.partition,
        nodes=1,
        tasks_per_node=config.tasks_per_node,
        cpus_per_task=config.cpus_per_task,
        slurm_job_name=config.job_name,
        slurm_cpus_per_task=config.cpus_per_task,
        slurm_time=config.time_minutes,
        slurm_mem=config.mem_gb,
        slurm_gpus_per_node=config.gpus_per_node,
    )

    # Create output folders
    Path("logs").mkdir(exist_ok=True)
    Path("tensorboard").mkdir(exist_ok=True)

    # Submit one job for each combination
    jobs = []
    for algo in config.algos:
        for env in config.envs:
            for seed in config.seeds:
                # Submit the job
                job = executor.submit(run_training, algo, env, seed)
                jobs.append((algo, env, seed, job))
                print(f"Submitted job {job.job_id}: {algo} on {env} with seed {seed}")

    print(f"\nSubmitted {len(jobs)} jobs to SLURM")
    print("\nTo check status: squeue -u $USER")
    print("To cancel all: scancel -u $USER")

if __name__ == "__main__":
    main()
