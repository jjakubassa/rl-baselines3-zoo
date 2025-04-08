from typing import Literal
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from stable_baselines3 import PPO
from rl_zoo3.train import ExperimentManager
import gymnasium as gym
import hydra
import submitit
import torch as th
import torch.nn as nn
from omegaconf import OmegaConf
from rich.traceback import install
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.checkpoint import Optional
import numpy as np

from rl_zoo3.custom_feature_extractor import MandlFeaturesExtractor

# 3. Test environment creation
from stable_baselines3.common.env_checker import check_env

test_env = gym.make('Mandl-v0')
check_env(test_env)
print("Test reset:", test_env.reset())
print("Observation space:", test_env.observation_space)
print("Action space:", test_env.action_space)

# 4. Create command line arguments
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args([])

# Set required arguments
args.algo = "ppo"
args.env = "Mandl-v0"
args.n_timesteps = 100_000
args.optimize_hyperparameters = False
args.n_trials = 10
args.n_evaluations = 2
args.eval_freq = 10000
args.n_eval_episodes = 5
args.log_folder = "logs"
args.verbose = 0

policy_kwargs = dict(
    features_extractor_class=MandlFeaturesExtractor,
)

from sb3_contrib import MaskablePPO

def make_env():
    return gym.make('Mandl-v0')

vec_env = VecMonitor(DummyVecEnv([make_env for _ in range(8)]))
vec_env = VecNormalize(
    vec_env,
    norm_obs=False,  # normalize observations
    norm_reward=True,  # normalize rewards
    clip_obs=10.,  # clip observations to this value
    # clip_reward=10.,  # clip rewards to this value
    # gamma=0.99,  # discount factor
    epsilon=1e-8,  # small constant to avoid division by zero
)

net_arch = {
    "pi": [256] * 2,
    "vf": [256] * 2,
}

model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    verbose=1,
    policy_kwargs={
        "net_arch": net_arch,
        "features_extractor_class": MandlFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    },
)

from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
model.learn(total_timesteps=10000, progress_bar=True)

# 5. Create and run the experiment manager
exp_manager = ExperimentManager(
    args,
    args.algo,
    args.env,
    args.log_folder,
    hyperparams={"policy_kwargs": policy_kwargs},
    tensorboard_log="tensorboard_logs",
    n_timesteps=args.n_timesteps,
    eval_freq=args.eval_freq,
    n_eval_episodes=args.n_eval_episodes,
    save_freq=-1,
    optimize_hyperparameters=args.optimize_hyperparameters,
    n_trials=args.n_trials,
    n_evaluations=args.n_evaluations,
    n_jobs=1,
    sampler="tpe",
    pruner="median",
    verbose=args.verbose,
    vec_env_type="dummy",
    show_progress=True,
)

exp_manager.setup_experiment()

# 6. Run the optimization
exp_manager.hyperparameters_optimization()
