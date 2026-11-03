"""Define pressure-case geometry, controls, observations, and data cutoffs."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isclose, isfinite

import numpy as np
from numpy.typing import NDArray


class Phase(StrEnum):
    """Separate conditioning, validation, and protected final observations."""

    CONDITIONING = "conditioning"
    VALIDATION = "validation"
    FINAL = "final"


# Start/end times in seconds, then west/east/south/north rate multipliers.
type WellControlInterval = tuple[float, float, tuple[float, float, float, float]]


@dataclass(frozen=True)
class PressureCase:
    """Specify known grid, physics, controls, and observation settings, in SI units.

    observation_std_pa fixes observation-error standard deviation. Geological structure
    is supplied separately from these public experiment settings.
    """

    grid_size: int = 20
    length_m: float = 100.0
    thickness_m: float = 10.0
    storage_per_pa: float = 1e-9
    viscosity_pa_s: float = 1e-3
    initial_pressure_pa: float = 2e7
    time_step_s: float = 3600.0
    conditioning_end_s: float = 32 * 3600.0
    validation_end_s: float = 48 * 3600.0
    final_end_s: float = 64 * 3600.0
    sample_every: int = 2
    rate_m3_s: float = 1e-4
    observation_std_pa: float = 1000.0
    well_controls: tuple[WellControlInterval, ...] = (
        (0.0, 8 * 3600.0, (1.0, -1.0, 0.0, 0.0)),
        (16 * 3600.0, 24 * 3600.0, (0.0, 0.0, 1.0, -1.0)),
        (32 * 3600.0, 40 * 3600.0, (-0.6, 0.6, 0.0, 0.0)),
        (48 * 3600.0, 56 * 3600.0, (0.4, -0.4, -0.8, 0.8)),
    )

    def __post_init__(self) -> None:
        for name in ("grid_size", "sample_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if name != "well_controls" and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.grid_size < 4 or self.grid_size % 2:
            raise ValueError("grid_size must be even and at least four")
        if not self.conditioning_end_s < self.validation_end_s < self.final_end_s:
            raise ValueError("conditioning, validation, and final cutoffs must be increasing")
        for name in ("conditioning_end_s", "validation_end_s", "final_end_s"):
            steps = getattr(self, name) / self.time_step_s
            if not steps.is_integer():
                raise ValueError(f"{name} must align with a simulation time step")
            if steps % self.sample_every:
                raise ValueError(f"{name} must align with an observation time")
        previous_end_s = 0.0
        for start_s, end_s, multipliers in self.well_controls:
            if (
                not isfinite(start_s)
                or not isfinite(end_s)
                or not 0 <= start_s < end_s <= self.final_end_s
            ):
                raise ValueError("well-control intervals must lie within the experiment")
            if start_s < previous_end_s:
                raise ValueError("well-control intervals must be ordered and non-overlapping")
            if any(not (time_s / self.time_step_s).is_integer() for time_s in (start_s, end_s)):
                raise ValueError("well-control intervals must align with simulation time steps")
            if len(multipliers) != 4 or not all(isfinite(value) for value in multipliers):
                raise ValueError("well controls must contain four finite rate multipliers")
            if not isclose(sum(multipliers), 0.0, abs_tol=1e-12):
                raise ValueError("well-control rates must sum to zero")
            previous_end_s = end_s

    @property
    def conditioning_end_step(self) -> int:
        """Return the validated simulation-step index of the conditioning cutoff."""
        return int(self.conditioning_end_s / self.time_step_s)

    @property
    def total_steps(self) -> int:
        """Return the validated number of simulation steps through the final cutoff."""
        return int(self.final_end_s / self.time_step_s)

    @property
    def well_positions(self) -> NDArray[np.float64]:
        """Return west/east and south/north well coordinates as domain fractions."""
        return np.array([[0.2, 0.25], [0.8, 0.25], [0.25, 0.2], [0.25, 0.8]])

    @property
    def gauge_positions(self) -> NDArray[np.float64]:
        """Return gauge coordinates at the wells, in the same order."""
        return self.well_positions

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        """Return colocated gauge names in west/east/south/north well order."""
        return ("west", "east", "south", "north")

    @property
    def rates(self) -> NDArray[np.float64]:
        """Expand [start, end) controls into balanced per-step rates; gaps have all wells off."""
        rates = np.zeros((self.total_steps, len(self.well_positions)), dtype=np.float64)
        for start_s, end_s, multipliers in self.well_controls:
            start = int(start_s / self.time_step_s)
            end = int(end_s / self.time_step_s)
            rates[start:end] = np.asarray(multipliers) * self.rate_m3_s
        return rates

    @property
    def observation_steps(self) -> NDArray[np.int64]:
        return np.arange(self.sample_every, self.total_steps + 1, self.sample_every)

    @property
    def observation_times_s(self) -> NDArray[np.float64]:
        """Return the elapsed physical time of each observation sample."""
        return self.observation_steps * self.time_step_s

    @property
    def observation_ids(self) -> NDArray[np.str_]:
        """Label observations in time-major, then gauge order."""
        return np.array(
            [
                f"step_{step:04d}:{gauge}"
                for step in self.observation_steps
                for gauge in self.gauge_ids
            ]
        )

    def phase_mask(self, phase: Phase) -> NDArray[np.bool_]:
        """Select observations for conditioning, validation, or final evaluation."""
        times_s = np.repeat(self.observation_times_s, len(self.gauge_ids))
        match phase:
            case Phase.CONDITIONING:
                return times_s <= self.conditioning_end_s
            case Phase.VALIDATION:
                return (times_s > self.conditioning_end_s) & (times_s <= self.validation_end_s)
            case Phase.FINAL:
                return times_s > self.validation_end_s

    def settings(self) -> dict[str, object]:
        """Return JSON-ready case settings, including explicit well-control intervals."""
        settings = asdict(self)
        settings["well_controls"] = [
            [start_s, end_s, list(multipliers)]
            for start_s, end_s, multipliers in self.well_controls
        ]
        return settings

    def specification(self) -> dict[str, object]:
        """Record case settings, units, controls, observation order, and cutoffs."""
        return {
            "schema_version": 2,
            "case_id": "pressure_diffusion_v1",
            "settings": self.settings(),
            "boundary_condition": "zero_flux",
            "pressure_unit": "Pa",
            "permeability_unit": "m2",
            "rate_unit": "m3/s",
            "cell_order": "C: y rows, x columns",
            "observation_order": "time, then gauge",
            "observation_ids": self.observation_ids.tolist(),
            "well_positions_fraction": self.well_positions.tolist(),
            "gauge_positions_fraction": self.gauge_positions.tolist(),
            "rates_m3_s": self.rates.tolist(),
            "conditioning_end_step": self.conditioning_end_step,
            "validation_end_step": int(self.validation_end_s / self.time_step_s),
            "final_end_step": self.total_steps,
            # Keep the fixed label: load_case validates it as part of the artifact format.
            "observation_error": "independent Gaussian, fixed standard deviation",
        }
