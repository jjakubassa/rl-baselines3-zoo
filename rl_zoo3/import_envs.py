from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec, register, register_envs
from jumanji.environments.routing.mandl import Mandl
from jumanji.wrappers import JumanjiToGymWrapper

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

try:
    import ale_py

    # no-op
    gym.register_envs(ale_py)
except ImportError:
    pass

try:
    import highway_env
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs
except ImportError:
    pass

try:
    import gym_donkeycar
except ImportError:
    pass

try:
    import panda_gym
except ImportError:
    pass

try:
    import rocket_lander_gym
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass


# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )

def make_env(**kwargs) -> gym.Env:

    from jumanji.environments.routing.mandl import Mandl
    from jumanji.wrappers import JumanjiToGymWrapper

    default_kwargs = {
        "network_name" : "mandl1",
        "runtime" : 150.0,
        "buffer_time_end" : 50.0,
        "buffer_time_start" : 8,
        "vehicle_capacity" : 50,
        "solution_name" : None,  # None means no solution from file
        "num_fix_routes" : 4,
        "num_flex_routes" : 16,
        "max_route_length" : 8,
        "allow_actions_fixed_routes" : True,
        "total_vehicles" : 99,
        "vehicles_per_additional_fixed_route" : None,
        "passenger_init_mode" : "evenly_spaced",
    }
    # Update defaults with any provided kwargs
    default_kwargs.update(kwargs)

    # Create and wrap the environment
    env = JumanjiToGymWrapper(Mandl(**default_kwargs))

    # Set the environment spec
    env.spec = EnvSpec(
        id='Mandl-v0',
        entry_point=make_env,
        max_episode_steps=1000,
        kwargs=default_kwargs
    )

    return env

gym.register(
    id='CederFix-v0',
    entry_point=lambda **kwargs: make_env(
        network_name="ceder1",
        num_flex_routes=0,
        num_fix_routes=3,
        max_route_length=3,
        total_vehicles=12,
        **kwargs
    ),
)

gym.register(
    id='CederFlex-v0',
    entry_point=lambda **kwargs: make_env(
        network_name="ceder1",
        num_flex_routes=12,
        num_fix_routes=0,
        max_route_length=30,
        total_vehicles=12,
        **kwargs
    ),
)

gym.register(
    id='MandlFix-v0',
    entry_point=lambda **kwargs: make_env(
        network_name = "mandl1",
        num_fix_routes = 4,
        num_flex_routes = 0,
        max_route_length = 8,
        total_vehicles = 99,
        vehicles_per_additional_fixed_route = (14, 26, 29, 30),
        **kwargs
    ),
)

gym.register(
    id='MandlFlex-v0',
    entry_point=lambda **kwargs: make_env(
        network_name="mandl1",
        num_flex_routes=99,
        num_fix_routes=0,
        total_vehicles=99,
        max_route_length=75,
        vehicle_capacity=50,
        **kwargs
    ),
)

gym.register(
    id='MandlReplace-v0',
    entry_point=lambda **kwargs: make_env(
        network_name="mandl1",
        solution_name="yoo2023with8stopsreplace",
        allow_actions_fixed_routes=False,
        num_flex_routes=33,
        num_fix_routes=0,
        max_route_length=75,
        total_vehicles=99,
        vehicle_capacity=50,
        **kwargs
    ),
)

gym.register(
    id='CederReplace-v0',
    entry_point=lambda **kwargs: make_env(
        network_name="ceder1",
        solution_name="Solution2a",
        allow_actions_fixed_routes=False,
        num_flex_routes=3,
        num_fix_routes=0,
        max_route_length=30,
        total_vehicles=12,
        vehicle_capacity=50,
        **kwargs
    ),
)

gym.register(
    id='Mumford0Fix-v0',
    entry_point=lambda **kwargs: make_env(
        network_name = "mumford0",
        num_fix_routes = 12,
        num_flex_routes = 0,
        max_route_length = 8,
        total_vehicles = 288,
        **kwargs
    ),
)
