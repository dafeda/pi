"""Execute labeled ensemble members serially and account for every attempted simulator call."""

import numpy as np
from numpy.typing import NDArray

from controller.progress import logger
from numerical.geology import BandGeometry, Family, parameter_names, permeability_field
from numerical.pressure import PressureModel


class MemberFailure(RuntimeError):
    """Identify a failed member; its candidate must stop without dropping or zero-filling it."""

    def __init__(self, family: Family, member_id: str, reason: str) -> None:
        super().__init__(f"{family.value}/{member_id}: {reason}")
        self.member_id = member_id
        self.family = family


class LocalExecutor:
    """Own serial ensemble execution and a shared simulator-call budget."""

    def __init__(self, model: PressureModel, max_simulations: int) -> None:
        self.model = model
        self.max_simulations = max_simulations
        self.simulations = 0
        self.max_balance_error_m3 = 0.0

    def batch(
        self,
        family: Family,
        parameters: NDArray[np.float64],
        steps: int,
        geometry: BandGeometry | None = None,
    ) -> NDArray[np.float64]:
        """Return observations by members, or fail the candidate with a stable member ID."""
        if parameters.ndim != 2 or parameters.shape[0] != len(parameter_names(family)):
            raise ValueError("parameter array rows do not match the requested pattern type")
        observations = steps // self.model.case.sample_every * len(self.model.case.gauge_ids)
        members = parameters.shape[1]
        responses = np.empty((observations, members))
        milestone = max(1, (members + 4) // 5)
        for member, values in enumerate(parameters.T):
            member_id = f"member_{member:04d}"
            try:
                field = permeability_field(family, values, self.model.case, geometry)
                if self.simulations >= self.max_simulations:
                    raise RuntimeError("search simulator-call budget exhausted")
                self.simulations += 1
                result = self.model.simulate(field, steps)
                responses[:, member] = result.responses_pa
                self.max_balance_error_m3 = max(
                    self.max_balance_error_m3, result.max_balance_error_m3
                )
            except (ValueError, FloatingPointError, RuntimeError) as error:
                raise MemberFailure(family, member_id, str(error)) from error
            completed = member + 1
            if completed % milestone == 0 or completed == members:
                logger.info(
                    "%s batch: %d/%d realizations complete; candidate simulator calls=%d/%d",
                    family.value,
                    completed,
                    members,
                    self.simulations,
                    self.max_simulations,
                )
        return responses
