import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SimpleVehicleUtilizationCallback(BaseCallback):
    """
    A simpler callback that logs vehicle utilization metrics once every N steps.
    This avoids relying on specific environment attributes.
    """

    def __init__(self, log_freq=50, log_freq_offset=75, verbose=0):
        """
        Initialize the callback.

        Args:
            log_freq: How often to log metrics (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_freq_offset = log_freq_offset

    def _on_step(self) -> bool:
        """Log metrics periodically."""
        if self.n_calls % (self.log_freq + self.log_freq_offset) == 0:
            env = self.training_env
            states = env.get_attr("_state")
            for state in states:
                capacity_per_vehicle = state.fleet.passengers.shape[1]
                vehicle_occupancy = np.sum(state.fleet.passengers != -1, axis=1)
                avg_utilization = np.mean(vehicle_occupancy) / capacity_per_vehicle
                percent_empty = np.mean(vehicle_occupancy == 0)
                percent_full = np.mean(vehicle_occupancy == capacity_per_vehicle)

                self.logger.record_mean(f"vehicle/avg_utilization_step-{self.log_freq_offset}", avg_utilization)
                self.logger.record_mean(f"vehicle/percent_empty_periodic-{self.log_freq_offset}", percent_empty)
                self.logger.record_mean(f"vehicle/percent_full_periodic-{self.log_freq_offset}", percent_full)
        return True
