"""Check allowed permeability patterns and identify proposals covering the same grid cells."""

import hashlib
import json
from dataclasses import dataclass
from math import isfinite

import numpy as np

from cases.pressure import PressureCase
from numerical.geology import BandGeometry, Family, band_cells


@dataclass(frozen=True)
class Hypothesis:
    """Select a reviewed family and fixed geometry without changing inference policy."""

    family: Family
    geometry: BandGeometry | None = None

    def specification(self) -> dict[str, object]:
        if self.family == Family.HOMOGENEOUS:
            return {"kind": "homogeneous"}
        return {
            "kind": "band",
            "orientation": "vertical" if self.family == Family.VERTICAL_BAND else "horizontal",
            "center_fraction": self.geometry.center_fraction,
            "width_fraction": self.geometry.width_fraction,
        }

    def identifier(self, case: PressureCase) -> str:
        """Reuse equivalent rasterized geometries, not just identical submitted JSON."""
        identity = {"family": self.family.value}
        if self.geometry is not None:
            identity["cells"] = np.flatnonzero(band_cells(case, self.geometry)).tolist()
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return "h_" + hashlib.sha256(encoded).hexdigest()[:24]


def parse_hypothesis(value: object, case: PressureCase) -> Hypothesis:
    """Reject code, unsupported fields, and invalid band geometry at the proposal boundary."""
    if not isinstance(value, dict):
        raise ValueError("hypothesis must be an object")
    if value.get("kind") == "homogeneous":
        if set(value) != {"kind"}:
            raise ValueError("homogeneous hypothesis only accepts kind")
        return Hypothesis(Family.HOMOGENEOUS)
    if value.get("kind") != "band":
        raise ValueError("supported hypothesis kinds are homogeneous and band")
    if set(value) != {"kind", "orientation", "center_fraction", "width_fraction"}:
        raise ValueError("band requires orientation, center_fraction, and width_fraction only")
    if value["orientation"] not in ("vertical", "horizontal"):
        raise ValueError("band orientation must be vertical or horizontal")
    for key in ("center_fraction", "width_fraction"):
        if type(value[key]) not in (int, float):
            raise ValueError(f"{key} must be a finite number")
    geometry = BandGeometry(
        center_fraction=float(value["center_fraction"]),
        width_fraction=float(value["width_fraction"]),
    )
    validate_band_geometry(geometry, case)
    family = Family.VERTICAL_BAND if value["orientation"] == "vertical" else Family.HORIZONTAL_BAND
    return Hypothesis(family, geometry)


def validate_band_geometry(geometry: BandGeometry, case: PressureCase) -> None:
    """Reject non-finite, out-of-domain, or insufficiently resolved band geometry."""
    center = geometry.center_fraction
    width = geometry.width_fraction
    if not (isfinite(center) and isfinite(width)):
        raise ValueError("center_fraction and width_fraction must be finite numbers")
    if not 2 / case.grid_size - 1e-12 <= width <= 0.5:
        raise ValueError(f"band width must lie in [{2 / case.grid_size}, 0.5]")
    if not width / 2 <= center <= 1 - width / 2:
        raise ValueError("band must lie inside the domain")
    cells = np.flatnonzero(band_cells(case, geometry))
    if cells.size < 2 or cells[0] == 0 or cells[-1] == case.grid_size - 1:
        raise ValueError(
            "band must span at least two grid rows/columns with background material on both sides"
        )
