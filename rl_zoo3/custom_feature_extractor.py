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



class MandlFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Constants for handling special values
        self.inf_replacement = 200_000.0
        self.max_finite_value = 100_000.0
        self.eps = 1e-8

        # Calculate expected feature sizes
        self.num_routes = observation_space.spaces["route_frequencies"].shape[0]  # 12
        self.num_nodes = observation_space.spaces["is_terminal"].shape[0]  # 15
        self.num_vehicles = observation_space.spaces["fleet_positions"].shape[0] // 2  # 12
        self.max_route_length = observation_space.spaces["route_stops"].shape[0] // self.num_routes  # 8

        # Calculate input sizes
        action_mask_size = self.num_routes * (self.num_nodes + 1)  # 192 for your case
        travel_times_size = self.num_nodes * self.num_nodes  # 225 for your case
        route_stops_size = self.num_routes * self.max_route_length  # 96 for your case
        fleet_positions_size = self.num_vehicles * 2  # 24 for your case
        route_types_size = self.num_routes  # 12 for your case
        route_frequencies_size = self.num_routes  # 12 for your case
        is_terminal_size = self.num_nodes  # 15 for your case

        # Define extractors with correct input sizes
        self.extractors = nn.ModuleDict({
            "action_mask": nn.Sequential(
                nn.Linear(action_mask_size, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            "travel_times": nn.Sequential(
                nn.Flatten(),
                nn.Linear(travel_times_size, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            "direct_travel_times": nn.Sequential(
                nn.Flatten(),
                nn.Linear(travel_times_size, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            "transfer_travel_times": nn.Sequential(
                nn.Flatten(),
                nn.Linear(travel_times_size, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            "network_shortest_times": nn.Sequential(
                nn.Flatten(),
                nn.Linear(travel_times_size, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            "route_stops": nn.Sequential(
                nn.Flatten(),
                nn.Linear(route_stops_size, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            "fleet_positions": nn.Sequential(
                nn.Flatten(),
                nn.Linear(fleet_positions_size, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            ),
            "route_types": nn.Sequential(
                nn.Linear(route_types_size, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            ),
            "route_frequencies": nn.Sequential(
                nn.Linear(route_frequencies_size, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            ),
            "is_terminal": nn.Sequential(
                nn.Linear(is_terminal_size, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            )
        })

        # Calculate total feature size
        total_features = (64 * 6) + (32 * 4)  # 6 large (64) + 4 small (32) feature extractors

        # Scalar features
        self.scalar_features = [
            "current_time",
            "max_route_length",
            "num_fix_routes",
            "num_flex_routes",
            "num_nodes",
            "num_routes",
            "num_vehicles",
        ]
        self.scalar_extractor = nn.Sequential(
            nn.Linear(len(self.scalar_features), 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        total_features += 32

        print(f"Total features before combination: {total_features}")

        # Final combination layers
        self.combination_layer = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensors = []

        # Handle scalar features
        scalar_features = []
        for key in self.scalar_features:
            value = observations[key]
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            scalar_features.append(value)
        scalar_features = th.cat(scalar_features, dim=1)
        scalar_output = self.scalar_extractor(scalar_features)
        encoded_tensors.append(scalar_output)

        # Process other features
        for key, extractor in self.extractors.items():
            if key in observations:
                x = observations[key].float()

                # Handle travel times tensors
                if key in ["travel_times", "network_shortest_times", "direct_travel_times", "transfer_travel_times"]:
                    x = th.where(th.isinf(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.where(th.isnan(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.clamp(x, 0.0, self.inf_replacement)
                    x = x / self.inf_replacement

                # Ensure correct shape
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                elif x.dim() == 2:
                    if key in ["travel_times", "network_shortest_times", "direct_travel_times", "transfer_travel_times"]:
                        x = x.reshape(x.size(0), -1)  # Flatten 2D matrix

                encoded = extractor(x)
                encoded_tensors.append(encoded)

        # Combine all features
        combined = th.cat(encoded_tensors, dim=1)

        return self.combination_layer(combined)
