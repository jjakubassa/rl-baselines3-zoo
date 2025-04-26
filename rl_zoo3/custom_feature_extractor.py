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
        self.num_routes = observation_space.spaces["route_frequencies"].shape[0]
        self.num_nodes = observation_space.spaces["is_terminal"].shape[0]
        self.num_vehicles = observation_space.spaces["fleet_positions"].shape[0]
        self.max_route_length = observation_space.spaces["route_stops"].shape[0] // self.num_routes  # 8

        # Calculate input sizes
        action_mask_size = self.num_routes * (self.num_nodes + 1)
        travel_times_size = self.num_nodes * self.num_nodes
        route_stops_size = self.num_routes * self.max_route_length
        fleet_positions_size = self.num_vehicles
        route_types_size = self.num_routes
        route_frequencies_size = self.num_routes
        is_terminal_size = self.num_nodes

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

class SmallMandlFeaturesExtractor(BaseFeaturesExtractor):
    """
    Lighter feature extractor for Mandl/Ceder/Mumford environments.
    Still does 2-layer (MLP+LayerNorm) per feature, and a 2-layer combiner.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.num_routes = observation_space.spaces["route_frequencies"].shape[0]
        self.num_nodes = observation_space.spaces["is_terminal"].shape[0]
        self.num_vehicles = observation_space.spaces["fleet_positions"].shape[0]
        self.max_route_length = observation_space.spaces["route_stops"].shape[0] // self.num_routes

        action_mask_size = self.num_routes * (self.num_nodes + 1)
        travel_times_size = self.num_nodes * self.num_nodes
        route_stops_size = self.num_routes * self.max_route_length
        fleet_positions_size = self.num_vehicles

        def small_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(),
                nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
            )

        # 64 for big matrices, 32 for vectors
        self.extractors = nn.ModuleDict({
            "action_mask":          small_mlp(action_mask_size, 64),
            "travel_times":         small_mlp(travel_times_size, 64),
            "direct_travel_times":  small_mlp(travel_times_size, 64),
            "transfer_travel_times":small_mlp(travel_times_size, 64),
            "network_shortest_times":small_mlp(travel_times_size, 64),
            "route_stops":          small_mlp(route_stops_size, 64),
            "fleet_positions":      small_mlp(fleet_positions_size, 32),
            "route_types":          small_mlp(self.num_routes, 32),
            "route_frequencies":    small_mlp(self.num_routes, 32),
            "is_terminal":          small_mlp(self.num_nodes, 32),
        })

        self.scalar_features = [
            "current_time",
            "max_route_length",
            "num_fix_routes",
            "num_flex_routes",
            "num_nodes",
            "num_routes",
            "num_vehicles",
        ]
        self.scalar_extractor = small_mlp(len(self.scalar_features), 16)

        total_features = (6 * 64) + (4 * 32) + 16

        self.combination_layer = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim), nn.Tanh()
        )

        self.inf_replacement = 200_000.0

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensors = []

        # Scalars
        scalar_values = [observations[k].unsqueeze(-1) if observations[k].dim() == 1 else observations[k]
                         for k in self.scalar_features]
        scalar_tensor = th.cat(scalar_values, dim=1)
        encoded_tensors.append(self.scalar_extractor(scalar_tensor))

        # Main features
        for key, extractor in self.extractors.items():
            if key in observations:
                x = observations[key].float()
                if "travel_times" in key or "network_shortest_times" in key:
                    x = th.where(th.isinf(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.where(th.isnan(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.clamp(x, 0.0, self.inf_replacement)
                    x = x / self.inf_replacement
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                x = x.reshape(x.size(0), -1)
                encoded_tensors.append(extractor(x))

        combined = th.cat(encoded_tensors, dim=1)
        return self.combination_layer(combined)

class DeepMandlFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Mandl/Ceder/Mumford large environments.
    Deeper MLPs per feature, and deeper combined MLP for high capacity.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)    # <--- SB3 API, sets self.features_dim

        # Calculate dimensions
        self.num_routes = observation_space.spaces["route_frequencies"].shape[0]
        self.num_nodes = observation_space.spaces["is_terminal"].shape[0]
        self.num_vehicles = observation_space.spaces["fleet_positions"].shape[0]
        self.max_route_length = observation_space.spaces["route_stops"].shape[0] // self.num_routes

        action_mask_size = self.num_routes * (self.num_nodes + 1)
        travel_times_size = self.num_nodes * self.num_nodes
        route_stops_size = self.num_routes * self.max_route_length
        fleet_positions_size = self.num_vehicles
        route_types_size = self.num_routes
        route_frequencies_size = self.num_routes
        is_terminal_size = self.num_nodes

        # Deeper per-feature MLPs (3 layers)
        def layered_mlp(inf, outf):
            return nn.Sequential(
                nn.Linear(inf, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, outf), nn.LayerNorm(outf), nn.ReLU(),
            )

        self.extractors = nn.ModuleDict({
            "action_mask": layered_mlp(action_mask_size, 128),
            "travel_times": layered_mlp(travel_times_size, 128),
            "direct_travel_times": layered_mlp(travel_times_size, 128),
            "transfer_travel_times": layered_mlp(travel_times_size, 128),
            "network_shortest_times": layered_mlp(travel_times_size, 128),
            "route_stops": layered_mlp(route_stops_size, 128),
            "fleet_positions": layered_mlp(fleet_positions_size, 64),
            "route_types": layered_mlp(route_types_size, 64),
            "route_frequencies": layered_mlp(route_frequencies_size, 64),
            "is_terminal": layered_mlp(is_terminal_size, 64),
        })

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
        self.scalar_extractor = layered_mlp(len(self.scalar_features), 32)

        # Total features (6*128 + 4*64 + 32)
        total_features = (6 * 128) + (4 * 64) + 32

        # Deep combination MLP block (3 layers before output)
        self.combination_layer = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim), nn.Tanh()
        )

        self.inf_replacement = 200_000.0

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensors = []

        # Scalar features
        scalar_features = [observations[k].unsqueeze(-1) if observations[k].dim() == 1 else observations[k]
                           for k in self.scalar_features]
        scalar_features = th.cat(scalar_features, dim=1)
        encoded_tensors.append(self.scalar_extractor(scalar_features))

        # Main features
        for key, extractor in self.extractors.items():
            if key in observations:
                x = observations[key].float()
                if "travel_times" in key or "network_shortest_times" in key:
                    x = th.where(th.isinf(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.where(th.isnan(x), th.tensor(self.inf_replacement, device=x.device), x)
                    x = th.clamp(x, 0.0, self.inf_replacement)
                    x = x / self.inf_replacement
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                x = x.reshape(x.size(0), -1)
                encoded_tensors.append(extractor(x))

        combined = th.cat(encoded_tensors, dim=1)
        return self.combination_layer(combined)
