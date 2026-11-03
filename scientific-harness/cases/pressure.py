"""Define pressure-case geometry, controls, observations, and data cutoffs."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite

import numpy as np
from numpy.typing import NDArray


class Phase(StrEnum):
    """Separate conditioning, validation, and protected final observations."""

    CONDITIONING = "conditioning"
    VALIDATION = "validation"
    FINAL = "final"


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
    segment_steps: int = 8
    sample_every: int = 2
    rate_m3_s: float = 1e-4
    observation_std_pa: float = 1000.0

    def __post_init__(self) -> None:
        for name in ("grid_size", "segment_steps", "sample_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.grid_size < 4 or self.grid_size % 2:
            raise ValueError("grid_size must be even and at least four")
        if self.segment_steps % self.sample_every:
            raise ValueError("sample_every must divide segment_steps")

    @property
    def total_steps(self) -> int:
        return 8 * self.segment_steps

    @property
    def well_positions(self) -> NDArray[np.float64]:
        """Return west/east and south/north well coordinates as domain fractions."""
        return np.array([[0.2, 0.25], [0.8, 0.25], [0.25, 0.2], [0.25, 0.8]])

    @property
    def gauge_positions(self) -> NDArray[np.float64]:
        """Return gauge coordinates as domain fractions."""
        return np.array([[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]])

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        return ("south_west", "south_east", "north_west", "north_east")

    @property
    def rates(self) -> NDArray[np.float64]:
        """Return well rates: positive adds, negative removes fluid; each step sums to zero."""
        segments = np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [-0.6, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.4, -0.4, -0.8, 0.8],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return np.repeat(segments * self.rate_m3_s, self.segment_steps, axis=0)

    @property
    def observation_steps(self) -> NDArray[np.int64]:
        return np.arange(self.sample_every, self.total_steps + 1, self.sample_every)

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
        steps = np.repeat(self.observation_steps, len(self.gauge_ids))
        conditioning_end = 4 * self.segment_steps
        validation_end = 6 * self.segment_steps
        match phase:
            case Phase.CONDITIONING:
                return steps <= conditioning_end
            case Phase.VALIDATION:
                return (steps > conditioning_end) & (steps <= validation_end)
            case Phase.FINAL:
                return steps > validation_end

    def specification(self) -> dict[str, object]:
        """Record case settings, units, controls, observation order, and cutoffs."""
        return {
            "schema_version": 1,
            "case_id": "pressure_diffusion_v1",
            "settings": asdict(self),
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
            "conditioning_end_step": 4 * self.segment_steps,
            "validation_end_step": 6 * self.segment_steps,
            "final_end_step": self.total_steps,
            # Keep the fixed label: load_case validates it as part of the artifact format.
            "observation_error": "independent Gaussian, fixed standard deviation",
        }
